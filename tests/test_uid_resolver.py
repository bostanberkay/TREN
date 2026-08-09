# tests/test_uid_resolver.py
"""Tests for the EXPERIMENTAL, offline-only uid_resolver.py.

Not exercising any production code path -- uid_resolver.py is not imported
by cs_annotator_app.py or reranker_integration.py (see the last test class
below, which asserts exactly that). Uses the same bypass-__init__ Annotator
convention already established in tests/test_cs_pipeline.py,
tests/test_mixed_reranker.py, and tests/test_reranker_integration.py.
"""

import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs_pipeline import Annotator, DEFAULTS
import uid_resolver as ur


def _make_annotator(turkish_top=(), turkish_all=(), english_words=()):
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set(turkish_top)
    obj.turkish_freq_all = set(turkish_all)
    obj.english_freq_words = set(english_words)
    return obj


# ---------------------------------------------------------------------------
# Hard exclusion gates (check_eligibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token,expected_reason", [
    ("https://example.com/x", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
    ("@someuser", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
    ("#trending", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
    ("1234", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
    ("...", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
    ("😀😀", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
    ("A7K-204", "other_token (url/mention/hashtag/number/punctuation/code/emoji)"),
])
def test_eligibility_rejects_other_token_shapes(token, expected_reason):
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility(token, "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == expected_reason


def test_eligibility_rejects_email():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("foo@bar.com", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "email"


def test_eligibility_rejects_apostrophe_bearing_token():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("kitab'ın", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "apostrophe"


def test_eligibility_rejects_all_caps_acronym():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("NASA", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "all_caps_acronym"


def test_eligibility_rejects_alphanumeric_identifier():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("iphone11", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "alphanumeric_identifier"


def test_eligibility_rejects_probable_proper_name():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("Ankara", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "probable_proper_name_or_protected_ne"


def test_eligibility_rejects_non_uid_label():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("meyler", "TR", obj, DEFAULTS)
    assert eligible is False
    assert reason == "label_not_uid"


def test_eligibility_rejects_too_short_token():
    obj = _make_annotator()
    eligible, reason = ur.check_eligibility("a", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "too_short"


def test_eligibility_rejects_strong_direct_english_lexicon_match():
    obj = _make_annotator(english_words={"meyler"})
    eligible, reason = ur.check_eligibility("meyler", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "strong_direct_english_lexicon_match"


def test_eligibility_rejects_english_root_plus_turkish_suffix_analysis():
    # "uploadın" = stem "upload" (in english_words) + Turkish genitive "ın" --
    # exactly the shape the frozen reranker/residual stage would treat as a
    # MIXED candidate. This resolver must never contest that.
    obj = _make_annotator(english_words={"upload"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        eligible, reason = ur.check_eligibility("uploadın", "UID", obj, DEFAULTS)
    assert eligible is False
    assert reason == "english_root_turkish_suffix_analysis"


def test_eligibility_accepts_a_plausible_native_turkish_uid_token():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        eligible, reason = ur.check_eligibility("meyler", "UID", obj, DEFAULTS)
    assert eligible is True
    assert reason == "eligible"


# ---------------------------------------------------------------------------
# Evidence extraction / scoring
# ---------------------------------------------------------------------------

def test_extract_evidence_trusted_lexicon_whole_token():
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        evidence = ur.extract_evidence("meyler", obj, DEFAULTS)
    signals = {e.signal for e in evidence}
    assert "trusted_lexicon" in signals


def test_extract_evidence_recovered_stem_lexicon_signal():
    # "gellerin": nominal split with stem "gel" in the trusted lexicon.
    obj = _make_annotator(turkish_all={"gel"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        evidence = ur.extract_evidence("gellerin", obj, DEFAULTS)
    lexicon_items = [e for e in evidence if e.signal == "trusted_lexicon"]
    assert len(lexicon_items) == 1
    assert "recovered Turkish stem" in lexicon_items[0].description


def test_extract_evidence_fasttext_strong_signal_present_above_threshold():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        evidence = ur.extract_evidence("meyler", obj, DEFAULTS)
    assert any(e.signal == "fasttext_strong" for e in evidence)


def test_extract_evidence_fasttext_below_threshold_no_signal():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.5)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        evidence = ur.extract_evidence("meyler", obj, DEFAULTS)
    assert not any(e.signal == "fasttext_strong" for e in evidence)


def test_extract_evidence_orthographic_signal():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        evidence = ur.extract_evidence("güzelöş", obj, DEFAULTS)
    assert any(e.signal == "orthographic" for e in evidence)


def test_extract_evidence_matrix_language_auxiliary_signal_only_when_tr():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        evidence_tr = ur.extract_evidence("meyler", obj, DEFAULTS, matrix_lang="TR")
        evidence_en = ur.extract_evidence("meyler", obj, DEFAULTS, matrix_lang="EN")
    assert any(e.signal == "matrix_language" for e in evidence_tr)
    assert not any(e.signal == "matrix_language" for e in evidence_en)


def test_matrix_language_alone_never_reaches_threshold():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        decision = ur.decide("meyler", "UID", obj, DEFAULTS, matrix_lang="TR")
    assert decision.promote is False


def test_lexicon_alone_never_reaches_threshold():
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        decision = ur.decide("meyler", "UID", obj, DEFAULTS)
    assert decision.promote is False
    assert ur._primary_signal_count(decision.evidence) < ur.MIN_PRIMARY_SIGNALS


def test_two_primary_signals_without_matrix_bonus_stay_below_threshold():
    # trusted_lexicon(3) + fasttext_strong(3) = 6 < PROMOTION_THRESHOLD(7):
    # exactly the "two signals is not automatically enough" conservative case.
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        decision = ur.decide("meyler", "UID", obj, DEFAULTS)
    assert ur._primary_signal_count(decision.evidence) == 2
    assert decision.score == 6
    assert decision.promote is False


# ---------------------------------------------------------------------------
# decide() end to end
# ---------------------------------------------------------------------------

def test_decide_safe_promotion_with_three_primary_signals():
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True):
        decision = ur.decide("meyler", "UID", obj, DEFAULTS)
    assert decision.promote is True
    assert decision.proposed_label == "TR"
    assert decision.score == 9
    assert ur._primary_signal_count(decision.evidence) == 3


def test_decide_safe_promotion_with_two_primary_signals_plus_matrix_bonus():
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        decision = ur.decide("meyler", "UID", obj, DEFAULTS, matrix_lang="TR")
    assert decision.score == 7
    assert decision.promote is True
    assert decision.proposed_label == "TR"


def test_decide_ambiguous_token_retained_as_uid():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        decision = ur.decide("zzqxwv", "UID", obj, DEFAULTS)
    assert decision.promote is False
    assert decision.proposed_label == "UID"
    assert decision.reason == "ambiguous_or_insufficient_evidence"


def test_decide_english_root_turkish_suffix_retained_as_uid():
    obj = _make_annotator(english_words={"upload"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        decision = ur.decide("uploadın", "UID", obj, DEFAULTS)
    assert decision.promote is False
    assert decision.proposed_label == "UID"
    assert decision.reason == "english_root_turkish_suffix_analysis"


def test_decide_is_deterministic():
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True):
        d1 = ur.decide("meyler", "UID", obj, DEFAULTS)
        d2 = ur.decide("meyler", "UID", obj, DEFAULTS)
    assert d1 == d2


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------

def test_explain_shape_for_promotion():
    obj = _make_annotator(turkish_all={"meyler"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True):
        decision = ur.decide("meyler", "UID", obj, DEFAULTS)
    text = ur.explain(decision)
    assert text.startswith("Token: meyler\nCurrent: UID\nProposed: TR\nScore: 9\nEvidence:\n")
    assert "Decision: promote (promoted)" in text


def test_explain_shape_for_retention():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        decision = ur.decide("zzqxwv", "UID", obj, DEFAULTS)
    text = ur.explain(decision)
    assert "Decision: retain (ambiguous_or_insufficient_evidence)" in text
    assert "- (none)" in text


# ---------------------------------------------------------------------------
# apply_uid_to_tr_resolver() -- block-level application
# ---------------------------------------------------------------------------

def _block(sent_id, token_label_pairs, matrix=None, embed=None):
    lines = [f"SentenceID\t{sent_id}"]
    for item, label in token_label_pairs:
        lines.append(f"{item}\t{label}")
    if matrix is not None:
        lines.append(f"MatrixLang\t{matrix}")
    if embed is not None:
        lines.append(f"EmbedLang\t{embed}")
    return "\n".join(lines)


def test_apply_resolver_promotes_eligible_uid_and_recomputes_matrix_embed():
    obj = _make_annotator(turkish_all={"meyler"})
    text = _block(1, [("ben", "TR"), ("meyler", "UID")], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True):
        new_text, decisions = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    assert "meyler\tTR" in new_text
    assert "UID" not in new_text
    assert len(decisions) == 1
    assert decisions[0].promote is True


def test_apply_resolver_leaves_block_byte_identical_when_no_promotion():
    obj = _make_annotator()
    text = _block(1, [("ben", "TR"), ("zzqxwv", "UID")], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=False):
        new_text, decisions = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    assert new_text == text
    assert len(decisions) == 1
    assert decisions[0].promote is False


@pytest.mark.parametrize("label", ["MIXED", "NE", "OTHER", "LANG3", "TR", "EN"])
def test_apply_resolver_never_touches_non_uid_labels(label):
    obj = _make_annotator(turkish_all={"meyler"})
    text = _block(1, [("meyler", label)], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True):
        new_text, decisions = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    assert new_text == text
    assert decisions == []


def test_apply_resolver_multiple_blocks_independent():
    obj = _make_annotator(turkish_all={"meyler"})
    block_a = _block(1, [("meyler", "UID")], matrix="TR", embed="-")
    block_b = _block(2, [("zzqxwv", "UID")], matrix="EN", embed="-")
    text = block_a + "\n\n" + block_b
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis") as mock_suffix:
        mock_suffix.side_effect = lambda tok_l: tok_l == "meyler"
        new_text, decisions = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    parts = new_text.split("\n\n")
    assert "meyler\tTR" in parts[0]
    assert "zzqxwv\tUID" in parts[1]
    assert len(decisions) == 2


def test_apply_resolver_recomputes_embed_after_promotion_changes_it():
    # A sentence that is otherwise all-TR: EmbedLang starts "-" (no EN/MIXED).
    # After the lone UID token promotes to TR, MatrixLang/EmbedLang must be
    # recomputed via the real, unmodified _decide_matrix_embed -- still TR/-,
    # but recomputed (not just left stale) is the point of this test.
    obj = _make_annotator(turkish_all={"meyler"})
    text = _block(1, [("ben", "TR"), ("meyler", "UID")], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True), \
         mock.patch.object(obj, "_decide_matrix_embed", wraps=obj._decide_matrix_embed) as spy:
        new_text, _ = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    spy.assert_called_once()
    assert "MatrixLang\tTR" in new_text
    assert "EmbedLang\t-" in new_text


def test_apply_resolver_is_deterministic():
    obj = _make_annotator(turkish_all={"meyler"})
    text = _block(1, [("meyler", "UID")], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)), \
         mock.patch.object(obj, "_has_valid_turkish_nominal_analysis", return_value=True):
        result1, _ = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
        result2, _ = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    assert result1 == result2


def test_apply_resolver_never_raises_on_evaluation_failure():
    obj = _make_annotator(turkish_all={"meyler"})
    text = _block(1, [("meyler", "UID")], matrix="TR", embed="-")
    with mock.patch.object(ur, "decide", side_effect=RuntimeError("boom")):
        new_text, decisions = ur.apply_uid_to_tr_resolver(text, obj, DEFAULTS)
    assert new_text == text
    assert decisions == []


# ---------------------------------------------------------------------------
# Integration boundary: as of the UID->TR resolver's production integration
# (see reranker_integration.py's module docstring / CHANGELOG), this module
# IS imported by reranker_integration.py, and only there -- cs_pipeline.py
# (the core rule-based engine) and cs_annotator_app.py (the GUI) must not
# duplicate or directly reference this module's logic; the single
# integration point is reranker_integration.apply_reranker(), exactly
# mirroring where the residual verbal detector was already integrated.
# ---------------------------------------------------------------------------

def test_uid_resolver_not_referenced_by_cs_pipeline_or_gui():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ("cs_annotator_app.py", "cs_pipeline.py"):
        path = os.path.join(repo_root, fname)
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "uid_resolver" not in source, (
            f"{fname} must not reference uid_resolver.py -- the single "
            f"production integration point is reranker_integration.py")


def test_uid_resolver_is_the_single_integration_point_in_reranker_integration():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "reranker_integration.py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "import uid_resolver as ur" in source
    assert "ur.decide(" in source
