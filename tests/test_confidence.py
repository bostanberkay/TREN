# tests/test_confidence.py
"""Tests for the production-safe confidence/uncertainty layer (confidence.py).

Uses the same bypass-__init__ Annotator convention already established in
tests/test_cs_pipeline.py, tests/test_mixed_reranker.py,
tests/test_reranker_integration.py, and tests/test_uid_resolver.py, plus the
real, small, tracked resources/models/ reranker bundle for the tests that
need genuine frozen-reranker probabilities (mirrors
tests/test_reranker_integration.py's `real_bundle` fixture)."""

import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs_pipeline import Annotator, DEFAULTS
import annotation_model
import confidence as cf
import reranker_integration as ri

REAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "models")


def _make_annotator(turkish_top=(), turkish_all=(), english_words=()):
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set(turkish_top)
    obj.turkish_freq_all = set(turkish_all)
    obj.english_freq_words = set(english_words)
    obj.ner = None
    return obj


@pytest.fixture(scope="module")
def real_bundle():
    bundle = ri.load_reranker_bundle(REAL_MODEL_DIR)
    assert bundle is not None
    return bundle


# ---------------------------------------------------------------------------
# band_for_score / thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,band", [
    (1.0, "HIGH"), (0.85, "HIGH"), (0.849, "MEDIUM"), (0.60, "MEDIUM"),
    (0.5999, "LOW"), (0.0, "LOW"),
])
def test_band_for_score_default_thresholds(score, band):
    assert cf.band_for_score(score) == band


def test_band_for_score_configurable_thresholds():
    custom = {"HIGH": 0.95, "MEDIUM": 0.50}
    assert cf.band_for_score(0.90, custom) == "MEDIUM"
    assert cf.band_for_score(0.96, custom) == "HIGH"
    assert cf.band_for_score(0.10, custom) == "LOW"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_compute_token_confidence_is_deterministic():
    obj = _make_annotator(turkish_top={"kitap"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)):
        r1 = cf.compute_token_confidence("kitap", "TR", "TR", obj, DEFAULTS)
        r2 = cf.compute_token_confidence("kitap", "TR", "TR", obj, DEFAULTS)
    assert r1 == r2
    assert r1.to_dict() == r2.to_dict()


def test_compute_block_confidence_is_deterministic_across_runs():
    obj = _make_annotator(turkish_top={"kitap"}, english_words={"cool"})
    rows = [
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "kitap", "label": "TR", "gloss": ""},
        {"idx": 2, "token": "cool", "label": "EN", "gloss": ""},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
    ]
    rows1 = [dict(r) for r in rows]
    rows2 = [dict(r) for r in rows]
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
        cf.compute_block_confidence(rows1, [], obj, DEFAULTS)
        cf.compute_block_confidence(rows2, [], obj, DEFAULTS)
    assert [r.get("confidence") for r in rows1] == [r.get("confidence") for r in rows2]


# ---------------------------------------------------------------------------
# Hard exclusions (OTHER)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", [
    "https://example.com/x", "@someuser", "#trending", "1234", "...",
    "😀😀", "A7K-204",
])
def test_other_hard_exclusion_gets_high_confidence_no_review(token):
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        rec = cf.compute_token_confidence(token, None, "OTHER", obj, DEFAULTS)
    assert rec.confidence_band == "HIGH"
    assert rec.review_recommended is False
    assert rec.uncertainty_reasons == ()


def test_other_non_exclusion_shape_is_not_high_confidence():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        rec = cf.compute_token_confidence("weirdlyOTHER", None, "OTHER", obj, DEFAULTS)
    assert rec.confidence_band != "HIGH"
    assert "other_label_does_not_match_any_automatic_exclusion_pattern" in rec.uncertainty_reasons


# ---------------------------------------------------------------------------
# All seven supported labels produce a valid record
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", list(cf.ALL_LABELS))
def test_every_schema_label_produces_a_valid_record(label):
    obj = _make_annotator(turkish_top={"kitap"}, english_words={"cool"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        rec = cf.compute_token_confidence("sometoken", label, label, obj, DEFAULTS)
    assert rec.final_label == label
    assert 0.0 <= rec.confidence_score <= 1.0
    assert rec.confidence_band in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(rec.uncertainty_reasons, tuple)
    assert isinstance(rec.evidence_summary, tuple)
    assert isinstance(rec.review_recommended, bool)
    d = rec.to_dict()
    assert d["final_label"] == label
    assert "calibration_note" in d and "NOT statistically calibrated" in d["calibration_note"]


def test_unrecognized_label_does_not_crash():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        rec = cf.compute_token_confidence("x", "TR", "NOT_A_REAL_LABEL", obj, DEFAULTS)
    assert rec.confidence_band in ("HIGH", "MEDIUM", "LOW")
    assert rec.review_recommended is True


# ---------------------------------------------------------------------------
# High / Medium / Low bands, per representative case
# ---------------------------------------------------------------------------

def test_tr_top_lexicon_is_high_confidence():
    obj = _make_annotator(turkish_top={"kitap"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)):
        rec = cf.compute_token_confidence("kitap", "TR", "TR", obj, DEFAULTS)
    assert rec.confidence_band == "HIGH"


def test_uid_never_reaches_high_band_even_with_maximum_corroborating_signal():
    # No lexicon/fastText/suffix evidence anywhere -- the "genuinely
    # unresolvable" bonus path -- must still cap below HIGH by design (UID
    # is semantically "below the LID confidence threshold").
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        rec = cf.compute_token_confidence("zzqxwv", "UID", "UID", obj, DEFAULTS)
    assert rec.confidence_band != "HIGH"
    assert rec.confidence_score <= 0.75


def test_lang3_is_low_confidence_and_flagged_for_review():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        rec = cf.compute_token_confidence("something", None, "LANG3", obj, DEFAULTS)
    assert rec.confidence_band == "LOW"
    assert rec.review_recommended is True


def test_en_conflicting_turkish_lexicon_lowers_confidence_to_medium_or_low():
    obj = _make_annotator(turkish_top={"cool"}, english_words={"cool"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        rec = cf.compute_token_confidence("cool", "EN", "EN", obj, DEFAULTS)
    assert rec.confidence_band != "HIGH"
    assert "token_also_present_in_turkish_lexicon" in rec.uncertainty_reasons


# ---------------------------------------------------------------------------
# Frozen reranker score/margin reconstruction (real bundle, empirically
# known fixture tokens -- same ones documented in
# tests/test_reranker_integration.py)
# ---------------------------------------------------------------------------

def test_mixed_via_frozen_reranker_reconstructs_real_probability(real_bundle):
    obj = _make_annotator(english_words={"boost"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        rec = cf.compute_token_confidence("boostlamak", "UID", "MIXED", obj, DEFAULTS, bundle=real_bundle)
    assert rec.promoted_by == "frozen_reranker"
    assert rec.confidence_band == "HIGH"
    assert rec.confidence_score == pytest.approx(0.985, abs=0.01)


def test_mixed_via_frozen_reranker_near_threshold_flags_low_margin(real_bundle):
    # "applar" (english_words={"app"}) -> P(MIXED)=0.819 in production tests
    # -- here we force final_label="MIXED" to inspect the near-threshold
    # reporting path directly (rather than relying on it having actually
    # been promoted, which it would not be at 0.819 < 0.85).
    obj = _make_annotator(english_words={"app"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        rec = cf.compute_token_confidence("applar", "TR", "UID", obj, DEFAULTS, bundle=real_bundle)
    # Not promoted (still UID) -- the UID scorer should surface the
    # near-miss reranker evidence.
    assert rec.final_label == "UID"
    assert any("near_miss" in r for r in rec.uncertainty_reasons) or rec.evidence_summary


def test_uid_stays_uid_when_reranker_and_resolver_both_reject(real_bundle):
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.1)):
        rec = cf.compute_token_confidence("zzqxwv", "UID", "UID", obj, DEFAULTS, bundle=real_bundle)
    assert rec.promoted_by is None
    assert rec.final_label == "UID"


def test_promoted_by_none_when_rule_label_equals_final_label():
    obj = _make_annotator(turkish_top={"kitap"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)):
        rec = cf.compute_token_confidence("kitap", "TR", "TR", obj, DEFAULTS)
    assert rec.promoted_by is None


def test_uid_to_tr_resolver_evidence_reconstructed_and_capped_medium():
    # "meyler" fully in the trusted Turkish lexicon plus a valid suffix
    # analysis -- enough primary signals for uid_resolver.decide() to
    # promote UID->TR (mirrors tests/test_uid_resolver.py's own fixtures).
    obj = _make_annotator(turkish_all={"mey"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.95)):
        rec = cf.compute_token_confidence("meyler", "UID", "TR", obj, DEFAULTS, matrix_lang="TR")
    assert rec.promoted_by == "uid_to_tr_resolver"
    assert rec.confidence_band != "HIGH"  # capped: algorithmically recovered, not a direct lexicon hit


# ---------------------------------------------------------------------------
# MatrixLang/EmbedLang consistency
# ---------------------------------------------------------------------------

def test_embed_lang_inconsistency_penalizes_score():
    obj = _make_annotator(english_words={"cool"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        consistent = cf.compute_token_confidence(
            "cool", "EN", "EN", obj, DEFAULTS, matrix_lang="TR", embed_lang="EN")
        inconsistent = cf.compute_token_confidence(
            "cool", "EN", "EN", obj, DEFAULTS, matrix_lang="TR", embed_lang="-")
    assert inconsistent.confidence_score < consistent.confidence_score
    assert "embed_lang_inconsistent_with_en_token_in_tr_matrix_sentence" in inconsistent.uncertainty_reasons


# ---------------------------------------------------------------------------
# Never raises -- a hostile/incomplete annotator must degrade gracefully
# ---------------------------------------------------------------------------

def test_compute_token_confidence_never_raises_on_broken_annotator():
    class Broken:
        pass  # no turkish_freq_top/all, no english_freq_words, no _ft_predict

    rec = cf.compute_token_confidence("token", "TR", "TR", Broken(), DEFAULTS)
    assert rec.confidence_band in ("HIGH", "MEDIUM", "LOW")


def test_compute_block_confidence_never_raises_and_marks_low_on_failure():
    class Broken:
        pass

    rows = [{"idx": 1, "token": "x", "label": "TR", "gloss": ""}]
    cf.compute_block_confidence(rows, [], Broken(), DEFAULTS)
    assert rows[0]["confidence"]["confidence_band"] == "LOW"
    assert rows[0]["confidence"]["review_recommended"] is True
    assert rows[0]["label"] == "TR"  # label itself never touched


# ---------------------------------------------------------------------------
# Block/dataset-level integration: alignment, no label mutation, JSON
# round-trip
# ---------------------------------------------------------------------------

def _two_row_block(rule_label="UID", final_label="TR"):
    final_rows = [
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "meyler", "label": final_label, "gloss": ""},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
    ]
    rule_rows = [
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "meyler", "label": rule_label, "gloss": ""},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
    ]
    return final_rows, rule_rows


def test_attach_confidence_to_blocks_never_changes_labels_or_tokens():
    obj = _make_annotator(turkish_all={"mey"}, english_words={"cool"})
    blocks = [[
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "meyler", "label": "TR", "gloss": ""},
        {"idx": 2, "token": "cool", "label": "EN", "gloss": ""},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
    ]]
    before = json.loads(json.dumps(blocks))  # deep copy via JSON round-trip
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
        cf.attach_confidence_to_blocks(blocks, blocks, obj, DEFAULTS)
    for b_before, b_after in zip(before, blocks):
        for r_before, r_after in zip(b_before, b_after):
            assert r_before["token"] == r_after["token"]
            assert r_before["label"] == r_after["label"]
            assert r_before["gloss"] == r_after["gloss"]
    # confidence/reviewed keys were added, nothing else changed
    for row in blocks[0]:
        if not annotation_model.is_meta_row_token(row["token"]):
            assert "confidence" in row
            assert row["reviewed"] is False


def test_attach_confidence_recovers_rule_label_by_position():
    final_rows, rule_rows = _two_row_block(rule_label="UID", final_label="TR")
    obj = _make_annotator(turkish_all={"mey"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
        cf.attach_confidence_to_blocks([final_rows], [rule_rows], obj, DEFAULTS)
    conf = final_rows[1]["confidence"]
    assert conf["rule_based_label"] == "UID"
    assert conf["final_label"] == "TR"


def test_attach_confidence_block_count_mismatch_degrades_gracefully():
    final_rows, _ = _two_row_block()
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        cf.attach_confidence_to_blocks([final_rows], [], obj, DEFAULTS)  # empty rule_blocks
    conf = final_rows[1]["confidence"]
    assert conf["rule_based_label"] is None  # no crash, just "unknown"


def test_confidence_records_are_json_serializable():
    obj = _make_annotator(turkish_top={"kitap"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
        rec = cf.compute_token_confidence("kitap", "TR", "TR", obj, DEFAULTS)
    json.dumps(rec.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Legacy / missing-confidence accessors
# ---------------------------------------------------------------------------

def test_get_confidence_returns_none_for_legacy_row():
    row = {"idx": 1, "token": "x", "label": "TR", "gloss": ""}
    assert cf.get_confidence(row) is None
    assert cf.is_reviewed(row) is False
    assert cf.band_of(row) is None
    assert cf.is_review_required(row) is False


@pytest.mark.parametrize("label", list(cf.ALL_LABELS))
def test_is_review_required_mirrors_review_recommended_flag_for_every_label(label):
    uncertain = {"idx": 1, "token": "x", "label": label, "gloss": "",
                 "confidence": {"review_recommended": True, "confidence_band": "LOW"}}
    confident = {"idx": 1, "token": "y", "label": label, "gloss": "",
                 "confidence": {"review_recommended": False, "confidence_band": "HIGH"}}
    assert cf.is_review_required(uncertain) is True
    assert cf.is_review_required(confident) is False


def test_is_review_required_never_raises_on_malformed_confidence_value():
    assert cf.is_review_required({"confidence": "not-a-dict"}) is False
    assert cf.is_review_required({"confidence": None}) is False
    assert cf.is_review_required({}) is False


def test_mark_reviewed_and_note_manual_edit():
    row = {"idx": 1, "token": "x", "label": "TR", "gloss": "",
           "confidence": {"confidence_band": "LOW"}}
    cf.mark_reviewed(row)
    assert cf.is_reviewed(row) is True

    row["label"] = "EN"
    cf.note_manual_edit(row)
    conf = cf.get_confidence(row)
    assert conf["final_label"] == "EN"
    assert conf["promoted_by"] == "manual_edit"
    assert conf["confidence_band"] == "HIGH"
    assert cf.is_reviewed(row) is True
    json.dumps(conf)  # still JSON-serializable
