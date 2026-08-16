# tests/test_tdk_parser.py
"""Pure-function tests for tdk.py's morphological parser section (formerly
tdk_parser.py -- see CLAUDE.md section 3). Uses tdk.load_lexicon_annotator()
to get a REAL (but fastText/Stanza-free) lexicon-aware annotator -- required now that
parse_token's selection policy is lexicon-rank-aware (a bare
Annotator.__new__(Annotator), with no lexicon at all, can no longer
reproduce the required root/suffix splits, e.g. "filmin" -> "film" + "in"
needs to know "film" outranks "fil" in the real frequency lexicon)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs_pipeline import Annotator
import tdk as tp

_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")


def _annotator():
    """The real lexicon-loaded annotator -- local, fast (well under a
    second), no fastText/Stanza involved."""
    return tp.load_lexicon_annotator(
        freq_tr=os.path.join(_RESOURCES, "frequent_tr_words.txt"),
        freq_en=os.path.join(_RESOURCES, "frequent_en_words.txt"),
    )


def _empty_annotator():
    """A bare, lexicon-free stand-in -- used only for tests that must work
    even with no lexicon data at all (never-raises / fallback tests)."""
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set()
    obj.turkish_freq_all = set()
    obj.english_freq_words = set()
    return obj


ANN = _annotator()


# ---------------------------------------------------------------------------
# The reported bug: "filmin" must parse as "film" + "in", never "fil" + "min"
# ---------------------------------------------------------------------------

def test_filmin_splits_as_film_plus_in_not_fil_plus_min():
    r = tp.parse_token("filmin", ANN)
    assert r.success is True
    assert r.root == "film"
    assert r.segments == ("in",)
    assert r.root != "fil"
    assert "min" not in r.segments


def test_filmin_does_not_treat_m_as_an_arbitrary_single_character_suffix():
    r = tp.parse_token("filmin", ANN)
    assert not any(seg == "m" for seg in r.segments)


# ---------------------------------------------------------------------------
# The second reported bug: "sürdü" must parse as "sür" + "dü" (one atomic
# past-tense suffix), never "sürd" + "ü". Zero-marked 3rd-person-singular:
# no invented agreement suffix.
# ---------------------------------------------------------------------------

def test_surdu_splits_as_sur_plus_du_not_surd_plus_u():
    r = tp.parse_token("sürdü", ANN)
    assert r.success is True
    assert r.root == "sür"
    assert r.segments == ("dü",)


def test_du_is_never_split_into_d_plus_u():
    r = tp.parse_token("sürdü", ANN)
    assert "d" not in r.segments and "ü" not in r.segments


@pytest.mark.parametrize("token,root,seg", [
    ("geldi", "gel", "di"),
    ("baktı", "bak", "tı"),
    ("gördü", "gör", "dü"),
    ("yazıyor", "yaz", "ıyor"),
    ("gelecek", "gel", "ecek"),
])
def test_verb_forms_split_via_atomic_tense_suffix_not_character_by_character(token, root, seg):
    r = tp.parse_token(token, ANN)
    assert r.success is True
    assert r.root == root
    assert r.segments == (seg,)
    assert r.part_of_speech == "verb"
    # never a single leftover character masquerading as a suffix
    assert not any(len(s) == 1 for s in r.segments)


def test_verb_hierarchy_never_invents_a_zero_marked_third_person_suffix():
    # "sür" + "dü" is the COMPLETE 3sg past form; nothing should be
    # appended after "dü" for a bare 3rd-person-singular reading.
    r = tp.parse_token("sürdü", ANN)
    assert r.segments == ("dü",)


# ---------------------------------------------------------------------------
# Noun hierarchy: noun_root -> derivational -> plural -> possessive -> case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token,root,segs", [
    ("filmin", "film", ("in",)),
    ("filmlerimizden", "film", ("ler", "imiz", "den")),
    ("evlerimizde", "ev", ("ler", "imiz", "de")),
    ("cloudumuza", "cloud", ("umuz", "a")),
])
def test_noun_hierarchy_examples(token, root, segs):
    r = tp.parse_token(token, ANN)
    assert r.success is True
    assert r.root == root
    assert r.segments == segs
    assert r.part_of_speech == "noun"


def test_kitaplarda_prefers_the_genuinely_minimal_root_over_a_longer_attested_prefix():
    """"kitaplar" is itself an attested corpus-frequency surface form (it's
    "kitap" + plural), but it is NOT a root -- the fuller decomposition
    "kitap" + "lar" + "da" must win over "kitaplar" + "da"."""
    r = tp.parse_token("kitaplarda", ANN)
    assert r.success is True
    assert r.root == "kitap"
    assert r.segments == ("lar", "da")


def test_evler_single_case_suffix():
    r = tp.parse_token("evler", ANN)
    assert r.success is True
    assert r.root == "ev"
    assert "ler" in r.segments


# ---------------------------------------------------------------------------
# Longest-root preference / shorter-root rejection / arbitrary-letter
# rejection -- explicit regression coverage beyond the two headline bugs.
# ---------------------------------------------------------------------------

def test_prefers_longest_valid_root_over_shorter_coincidental_root():
    r = tp.parse_token("filmin", ANN)
    assert len(r.root) == len("film")


def test_rejects_arbitrary_single_character_as_a_verb_tense_suffix():
    r = tp.parse_token("sürdü", ANN)
    for seg in r.segments:
        assert len(seg) > 1 or seg in ("e", "a", "ı", "i", "u", "ü")  # legit 1-char nominal suffixes only


def test_full_token_tdk_style_lexical_match_priority():
    """A base dictionary word with no valid suffix decomposition at all
    ("masa", table) must be returned unsplit, category full lexical item,
    never forced into a spurious split."""
    r = tp.parse_token("masa", ANN)
    assert r.success is True
    assert r.segments == ()
    assert r.category == tp.CATEGORY_FULL_LEXICAL


# ---------------------------------------------------------------------------
# Ambiguous candidates: genuine competing readings close in score --
# category must say so rather than confidently pick one.
# ---------------------------------------------------------------------------

def test_ambiguous_candidate_category_for_close_competing_readings():
    # "kalem" (pen, a base word) vs "kale" (fortress) + "m" (1sg
    # possessive) -- both lexicon-plausible, genuinely ambiguous without a
    # TDK lookup.
    r = tp.parse_token("kalem", ANN)
    assert r.category == tp.CATEGORY_AMBIGUOUS


def test_never_produces_a_spurious_short_stem_split_for_a_base_noun():
    """Regression for a real bug found during development: reranking.py's
    (formerly mixed_reranker.py's) permissive "verbal" fallback table
    proposed "ka" + "le" + "m" for "kalem" -- a 2-character coincidental
    stem. tdk.py's parser (formerly tdk_parser.py) must never surface that:
    whichever candidate wins, it must not be a 2-character "verbal"-sourced
    stem."""
    r = tp.parse_token("kalem", ANN)
    assert not (len(r.root) < 3 and r.source == "verbal")


# ---------------------------------------------------------------------------
# Invalid suffix sequences / vowel harmony
# ---------------------------------------------------------------------------

def test_saatler_loanword_vowel_harmony_exception_still_splits():
    # "saat" (front-vowel-final by rule but a documented loanword
    # exception) + "ler" (front plural) -- harmony is a soft signal, never
    # a hard rejection.
    r = tp.parse_token("saatler", ANN)
    assert r.success is True
    assert r.root == "saat"
    assert r.segments == ("ler",)


def test_vowel_harmony_consistent_soft_signal_never_raises():
    assert tp.vowel_harmony_consistent("ev", "e") is True
    assert tp.vowel_harmony_consistent("", "") is None
    assert tp.vowel_harmony_consistent("xyz", "e") is None  # no vowel in stem


# ---------------------------------------------------------------------------
# Apostrophe / punctuation handling (parser sees whatever token text it is
# given -- normalization/edge-punctuation stripping is the caller's job in
# annotation_model.py, not this module's; this just documents that
# parse_token never raises on punctuation-bearing input).
# ---------------------------------------------------------------------------

def test_parse_token_with_apostrophe_never_raises():
    r = tp.parse_token("kitab'ın", ANN)
    assert r is not None
    assert r.token == "kitab'ın"


# ---------------------------------------------------------------------------
# The parser never assigns a language label based on this analysis alone.
# ---------------------------------------------------------------------------

def test_parse_result_never_carries_a_label_field():
    r = tp.parse_token("filmin", ANN)
    assert "label" not in r.to_dict()
    assert not hasattr(r, "label")


# ---------------------------------------------------------------------------
# Parser failure / no valid split -- whole-token fallback
# ---------------------------------------------------------------------------

def test_parse_token_no_suffix_chain_falls_back_to_whole_token():
    r = tp.parse_token("zzqxwv", ANN)
    assert r.success is False
    assert r.source == "whole_token_fallback"
    assert r.root == "zzqxwv"
    assert r.segments == ()
    assert r.category == tp.CATEGORY_INVALID


def test_parse_token_empty_token():
    r = tp.parse_token("", ANN)
    assert r.success is False
    assert r.root == ""
    assert "empty" in r.reason


def test_parse_token_whitespace_only():
    r = tp.parse_token("   ", ANN)
    assert r.success is False


def test_parse_token_none_never_raises():
    r = tp.parse_token(None, ANN)
    assert r.success is False


def test_parse_token_never_raises_on_broken_annotator():
    class Broken:
        pass
    r = tp.parse_token("cloudumuza", Broken())
    assert r.success is False
    # a broken annotator degrades _lexicon_flags to all-False (getattr
    # defaults), so candidate generation still runs -- the whole-token
    # pseudo-candidate is simply unconfirmed, same as any genuinely
    # unrecognized token.
    assert r.source == "whole_token_fallback"


def test_parse_token_with_empty_lexicon_annotator_never_raises():
    r = tp.parse_token("cloudumuza", _empty_annotator())
    assert r is not None
    assert f"{r.root}{r.suffix}".lower() == "cloudumuza"


# ---------------------------------------------------------------------------
# Root extraction / suffix reconstruction invariant
# ---------------------------------------------------------------------------

def test_parse_token_root_plus_suffix_always_reconstructs_token():
    for token in ("cloudumuza", "kitaplarda", "evde", "arabalarımızdan", "filmin", "sürdü"):
        r = tp.parse_token(token, ANN)
        assert f"{r.root}{r.suffix}".lower() == token.lower()


def test_parse_result_suffix_property_joins_segments():
    r = tp.parse_token("cloudumuza", ANN)
    assert r.suffix == "umuza"


def test_parse_result_to_dict_json_serializable():
    import json
    r = tp.parse_token("cloudumuza", ANN)
    json.dumps(r.to_dict())


# ---------------------------------------------------------------------------
# Segment explanations: every segment of a successful automatic split
# carries a human-readable rule string (the "explain how suffixes were
# found" requirement).
# ---------------------------------------------------------------------------

def test_segment_explanations_present_for_every_automatic_segment():
    r = tp.parse_token("filmin", ANN)
    assert len(r.segment_explanations) == len(r.segments)
    assert all(e.rule for e in r.segment_explanations)


def test_segment_explanations_never_claim_a_bare_letter_is_a_tense_suffix():
    r = tp.parse_token("sürdü", ANN)
    for seg, expl in zip(r.segments, r.segment_explanations):
        if len(seg) == 1:
            assert False, "no single-character verb tense suffix should ever be proposed"
        assert expl.valid is True


# ---------------------------------------------------------------------------
# Malformed segmentation handling (segments_from_text / reparse_with_manual_root)
# ---------------------------------------------------------------------------

def test_segments_from_text_valid_correction():
    r = tp.segments_from_text("cloudumuza", "cloud", "umuz + a")
    assert r.success is True
    assert r.root == "cloud"
    assert r.segments == ("umuz", "a")
    assert r.source == "manual"


def test_segments_from_text_accepts_hyphen_or_whitespace_separators():
    r1 = tp.segments_from_text("cloudumuza", "cloud", "umuz-a")
    r2 = tp.segments_from_text("cloudumuza", "cloud", "umuz a")
    assert r1.segments == r2.segments == ("umuz", "a")


def test_segments_from_text_malformed_does_not_reconstruct_token():
    r = tp.segments_from_text("cloudumuza", "cloud", "wrongsuffix")
    assert r.success is False
    assert "do not reconstruct" in r.reason
    # still returns the user's input for display, never silently discarded
    assert r.root == "cloud"
    assert r.segments == ("wrongsuffix",)


def test_segments_from_text_empty_root_is_malformed():
    r = tp.segments_from_text("cloudumuza", "", "umuz + a")
    assert r.success is False


def test_segments_from_text_empty_segments_text_is_valid_if_root_equals_token():
    r = tp.segments_from_text("kitap", "kitap", "")
    assert r.success is True
    assert r.segments == ()


def test_reparse_with_manual_root_valid():
    r = tp.reparse_with_manual_root("cloudumuza", "cloud")
    assert r.success is True
    assert r.root == "cloud"
    assert r.segments == ("umuza",)
    assert r.source == "manual"


def test_reparse_with_manual_root_not_a_prefix():
    r = tp.reparse_with_manual_root("cloudumuza", "xyz")
    assert r.success is False
    assert r.root == "cloudumuza"  # unchanged, no-op fallback


def test_reparse_with_manual_root_empty_token():
    r = tp.reparse_with_manual_root("", "cloud")
    assert r.success is False


def test_reparse_with_manual_root_whole_token_as_root_yields_no_segments():
    r = tp.reparse_with_manual_root("kitap", "kitap")
    assert r.success is True
    assert r.segments == ()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_parse_token_is_deterministic():
    a1, a2 = _annotator(), _annotator()
    r1 = tp.parse_token("filmin", a1)
    r2 = tp.parse_token("filmin", a2)
    assert r1 == r2


# ---------------------------------------------------------------------------
# load_lexicon_annotator: local, offline, no fastText/Stanza dependency.
# ---------------------------------------------------------------------------

def test_load_lexicon_annotator_reads_real_resource_files():
    obj = _annotator()
    assert "film" in obj.turkish_freq_top or "film" in obj.turkish_freq_all
    assert obj.ner is None  # never loads Stanza


def test_load_lexicon_annotator_missing_files_degrade_gracefully():
    obj = tp.load_lexicon_annotator(freq_tr="/nonexistent/path.txt", freq_en="/nonexistent/other.txt")
    assert obj.turkish_freq_top == set()
    r = tp.parse_token("anything", obj)
    assert r is not None  # never raises
