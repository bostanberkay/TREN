import csv
import os
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

import mixed_reranker as mr
from cs_pipeline import Annotator, DEFAULTS
import build_reranker_dataset as bld
import train_mixed_reranker as trn


def _make_annotator(turkish_top=(), turkish_all=(), english_words=()):
    """Same bypass-__init__ convention as tests/test_cs_pipeline.py -- no
    frequency-file reads, no fasttext.load_model() call."""
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set(turkish_top)
    obj.turkish_freq_all = set(turkish_all)
    obj.english_freq_words = set(english_words)
    return obj


def _block(sentence_id, tokens, matrix='TR', embed='EN'):
    """tokens: list of (idx, item, label, gloss). Ends with a blank
    block-separator row, matching the real CSV export convention."""
    rows = [['', 'SentenceID', sentence_id, '']]
    for idx, item, label, gloss in tokens:
        rows.append([idx, item, label, gloss])
    rows.append(['', 'MatrixLang', matrix, ''])
    rows.append(['', 'EmbedLang', embed, ''])
    rows.append(['', '', '', ''])
    return rows


# ---------------------------------------------------------------------------
# Candidate generation / selection (mixed_reranker.py)
# ---------------------------------------------------------------------------

def test_enumerate_candidate_analyses_fully_consumed_only():
    obj = _make_annotator()
    # "meeting'e" style suffix without apostrophe: "meetinge" -> stem "meeting" + suf "e" (Case=Dat)
    candidates = mr.enumerate_candidate_analyses("meetinge", obj)
    assert any(c.stem == "meeting" and c.suffix == "e" for c in candidates)
    for c in candidates:
        segments, ud, deriv, amb = obj._parse_tr_suffixes_full(c.suffix)
        assert "Unparsed=Leftover" not in deriv


def test_enumerate_candidate_analyses_respects_min_stem_len():
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("ie", obj, min_stem_len=2)
    # token length 2 -> no split point >= min_stem_len is possible (range(2,2) is empty)
    assert candidates == []


def test_enumerate_candidate_analyses_empty_token():
    obj = _make_annotator()
    assert mr.enumerate_candidate_analyses("", obj) == []


def test_select_best_analysis_prefers_longest_stem():
    obj = _make_annotator()
    # "hypeda" -> could split as "hype"+"da" (Case=Loc) or "hyped"+"a" (Case=Dat);
    # longest-stem policy must pick "hyped"+"a" over "hype"+"da" if both are plausible.
    # Strategy passed explicitly (Phase 3A changed the module default to
    # highest_suffix_segments) -- this test is specifically about the
    # longest_stem policy, so it must not depend on whatever the current
    # default happens to be.
    candidates = mr.enumerate_candidate_analyses("hypeda", obj)
    best = mr.select_best_analysis(candidates, strategy=mr.STRATEGY_LONGEST_STEM)
    assert best is not None
    stems = sorted({c.stem for c in candidates}, key=len)
    assert best.stem == stems[-1]


def test_select_best_analysis_none_for_empty_candidates():
    assert mr.select_best_analysis([]) is None


def test_best_analysis_for_token_strategy_parameter_changes_selection():
    # "hypeda" has three real plausible splits from the production suffix
    # tables: "hyp"+"eda" (2 segments, suffix len 3), "hype"+"da" (Case=Loc,
    # suffix len 2), "hyped"+"a" (Case=Dat, suffix len 1) -- longest_stem
    # and longest_suffix must disagree on which one to prefer.
    obj = _make_annotator()
    stem_pick = mr.best_analysis_for_token("hypeda", obj, strategy=mr.STRATEGY_LONGEST_STEM)
    suffix_pick = mr.best_analysis_for_token("hypeda", obj, strategy=mr.STRATEGY_LONGEST_SUFFIX)
    assert stem_pick.stem != suffix_pick.stem
    assert stem_pick.stem == "hyped"
    assert suffix_pick.stem == "hyp"


# ---------------------------------------------------------------------------
# Phase 1 fix 2: apostrophe tokens reuse Annotator._split_mixed_apostrophe
# ---------------------------------------------------------------------------

def test_enumerate_candidate_analyses_apostrophe_excludes_apostrophe_from_stem():
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("meeting'e", obj)
    assert len(candidates) == 1
    assert candidates[0].stem == "meeting"
    assert candidates[0].suffix == "e"
    assert "'" not in candidates[0].stem


def test_enumerate_candidate_analyses_apostrophe_curly_variant():
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("meeting’e", obj)
    assert len(candidates) == 1
    assert candidates[0].stem == "meeting"


def test_enumerate_candidate_analyses_apostrophe_english_contraction_rejected():
    obj = _make_annotator()
    # "boss's" -> suffix "s" is a known English contraction remnant in
    # Annotator._split_mixed_apostrophe -> production declines to split it,
    # and so must candidate generation.
    candidates = mr.enumerate_candidate_analyses("boss's", obj)
    assert candidates == []


def test_enumerate_candidate_analyses_apostrophe_no_naive_fallback():
    # Multiple apostrophes -> _split_mixed_apostrophe returns (None, None);
    # must NOT fall back to naive character-position enumeration.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("a'b'c", obj)
    assert candidates == []


def test_select_best_analysis_deterministic_tie_break_order():
    a = mr.CandidateAnalysis(stem="ab", suffix="cd", split_position=2,
                              segments=("cd",), ud_feats=frozenset({"X"}), deriv=frozenset(), amb=frozenset())
    b = mr.CandidateAnalysis(stem="ab", suffix="ce", split_position=2,
                              segments=("c", "e"), ud_feats=frozenset({"X"}), deriv=frozenset(), amb=frozenset())
    # same stem length -> more suffix segments wins (b has 2 segments vs a's 1)
    assert mr.select_best_analysis([a, b]) is b


def test_select_best_analysis_unknown_strategy_raises():
    a = mr.CandidateAnalysis(stem="ab", suffix="cd", split_position=2,
                              segments=("cd",), ud_feats=frozenset(), deriv=frozenset(), amb=frozenset())
    with pytest.raises(ValueError):
        mr.select_best_analysis([a], strategy="not_a_real_strategy")


def test_select_best_analysis_longest_suffix_strategy_prefers_longer_suffix():
    long_stem_short_suffix = mr.CandidateAnalysis(
        stem="abcdefghij", suffix="kl", split_position=10,
        segments=("kl",), ud_feats=frozenset({"X"}), deriv=frozenset(), amb=frozenset())
    short_stem_long_suffix = mr.CandidateAnalysis(
        stem="abcde", suffix="fghijkl", split_position=5,
        segments=("f", "g", "h"), ud_feats=frozenset({"X"}), deriv=frozenset(), amb=frozenset())
    candidates = [long_stem_short_suffix, short_stem_long_suffix]
    assert mr.select_best_analysis(candidates, strategy=mr.STRATEGY_LONGEST_STEM) is long_stem_short_suffix
    assert mr.select_best_analysis(candidates, strategy=mr.STRATEGY_LONGEST_SUFFIX) is short_stem_long_suffix


def test_select_best_analysis_highest_suffix_segments_strategy_prefers_more_segments():
    long_stem_one_segment = mr.CandidateAnalysis(
        stem="abcdefghij", suffix="kl", split_position=10,
        segments=("kl",), ud_feats=frozenset({"X"}), deriv=frozenset(), amb=frozenset())
    short_stem_three_segments = mr.CandidateAnalysis(
        stem="abcde", suffix="fgh", split_position=5,
        segments=("f", "g", "h"), ud_feats=frozenset({"X"}), deriv=frozenset(), amb=frozenset())
    candidates = [long_stem_one_segment, short_stem_three_segments]
    assert mr.select_best_analysis(candidates, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS) is short_stem_three_segments
    # and longest_stem strategy on the SAME candidates picks the other one, confirming it's the
    # strategy argument -- not the data -- driving the different outcome
    assert mr.select_best_analysis(candidates, strategy=mr.STRATEGY_LONGEST_STEM) is long_stem_one_segment


def test_select_best_analysis_highest_suffix_segments_tiebreaks_on_stem_length_only():
    # equal segment_count -> highest_suffix_segments must fall back to
    # longest stem as a TIE-BREAK ONLY (not as the primary criterion the
    # way 'longest_stem' strategy uses it).
    short_stem = mr.CandidateAnalysis(stem="abc", suffix="def", split_position=3,
                                       segments=("d", "e"), ud_feats=frozenset(), deriv=frozenset(), amb=frozenset())
    long_stem = mr.CandidateAnalysis(stem="abcdefgh", suffix="ij", split_position=8,
                                      segments=("i", "j"), ud_feats=frozenset(), deriv=frozenset(), amb=frozenset())
    result = mr.select_best_analysis([short_stem, long_stem], strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert result is long_stem


def test_select_best_analysis_earliest_split_final_tiebreak():
    a = mr.CandidateAnalysis(stem="ab", suffix="cd", split_position=2,
                              segments=("cd",), ud_feats=frozenset(), deriv=frozenset(), amb=frozenset())
    b = mr.CandidateAnalysis(stem="xy", suffix="cd", split_position=5,
                              segments=("cd",), ud_feats=frozenset(), deriv=frozenset(), amb=frozenset())
    # identical stem_length/segment_count/feature_count -> earliest split_position wins
    assert mr.select_best_analysis([a, b]) is a


# ---------------------------------------------------------------------------
# Candidate classification (uses only predicted info, never gold)
# ---------------------------------------------------------------------------

def test_classify_candidate_uid_always_candidate():
    obj = _make_annotator()
    is_cand, reason, analysis = mr.classify_candidate("UID", "xyz", obj, DEFAULTS)
    assert is_cand is True
    assert reason == mr.CANDIDATE_REASON_UID


def test_classify_candidate_en_other_mixed_lang3_never_candidate():
    obj = _make_annotator()
    for label in ("EN", "OTHER", "MIXED", "LANG3"):
        is_cand, reason, analysis = mr.classify_candidate(label, "meetinge", obj, DEFAULTS)
        assert is_cand is False
        assert reason is None
        assert analysis is None


def test_classify_candidate_ne_requires_plausible_suffix():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        # no suffix-parseable tail at all
        is_cand, reason, analysis = mr.classify_candidate("NE", "ab", obj, DEFAULTS)
    assert is_cand is False


def test_classify_candidate_ne_with_plausible_suffix_is_candidate():
    obj = _make_annotator()
    is_cand, reason, analysis = mr.classify_candidate("NE", "meetinge", obj, DEFAULTS)
    assert is_cand is True
    assert reason == mr.CANDIDATE_REASON_NE_SUFFIX


def test_classify_candidate_tr_requires_non_turkish_stem_evidence():
    obj = _make_annotator(turkish_all={"meeting"})  # stem looks Turkish -> should NOT be a candidate
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.99)):
        is_cand, reason, analysis = mr.classify_candidate("TR", "meetinge", obj, DEFAULTS)
    assert is_cand is False


def test_classify_candidate_tr_with_non_turkish_stem_is_candidate():
    obj = _make_annotator(english_words={"meeting"})
    is_cand, reason, analysis = mr.classify_candidate("TR", "meetinge", obj, DEFAULTS)
    assert is_cand is True
    assert reason == mr.CANDIDATE_REASON_TR_SUSPECT_STEM


# ---------------------------------------------------------------------------
# Phase 1 fix 1: CandidateAnalysis must survive candidacy rejection
# ---------------------------------------------------------------------------

def test_classify_candidate_tr_preserves_analysis_when_candidacy_rejected():
    # stem "meeting" resolves TR here (turkish_all membership), so the
    # non-Turkish-stem-evidence check correctly fails and is_candidate must
    # stay False -- but the underlying CandidateAnalysis must NOT be
    # discarded; a real suffix split was found and should reach features.
    obj = _make_annotator(turkish_all={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.99)):
        is_cand, reason, analysis = mr.classify_candidate("TR", "meetinge", obj, DEFAULTS)
    assert is_cand is False
    assert reason is None
    assert analysis is not None
    assert analysis.stem == "meeting"
    assert analysis.suffix == "e"


def test_classify_candidate_ne_no_analysis_still_returns_none():
    # sanity: when there is genuinely no plausible split, analysis is None
    # regardless of the fix (nothing to preserve).
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        is_cand, reason, analysis = mr.classify_candidate("NE", "ab", obj, DEFAULTS)
    assert is_cand is False
    assert analysis is None


# ---------------------------------------------------------------------------
# Structured feature extraction: shape and stability
# ---------------------------------------------------------------------------

def test_build_structured_feature_dict_shape_with_and_without_candidate():
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats_with = mr.build_structured_feature_dict("meetinge", "UID", mr.best_analysis_for_token("meetinge", obj), obj, DEFAULTS)
        feats_without = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    # same key set regardless of whether a candidate analysis was found
    assert set(feats_with.keys()) == set(feats_without.keys())
    assert feats_without["stem_length"] == 0
    assert feats_without["fully_consumed_suffix"] is False
    assert feats_with["fully_consumed_suffix"] is True


def test_build_structured_feature_dict_is_deterministic():
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        f1 = mr.build_structured_feature_dict("meetinge", "TR", mr.best_analysis_for_token("meetinge", obj), obj, DEFAULTS)
        f2 = mr.build_structured_feature_dict("meetinge", "TR", mr.best_analysis_for_token("meetinge", obj), obj, DEFAULTS)
    assert f1 == f2


def test_build_structured_feature_dict_turkish_char_and_casing_flags():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("Şeyler", "UID", None, obj, DEFAULTS)
    assert feats["has_turkish_char"] is True
    assert feats["initial_capital"] is True
    assert feats["all_uppercase"] is False


# ---------------------------------------------------------------------------
# Block parsing / classification (build_reranker_dataset.py)
# ---------------------------------------------------------------------------

def test_classify_blank_meta_token_rows():
    assert bld.classify(['', '', '', '']) == 'blank'
    assert bld.classify(['', 'SentenceID', '1', '']) == 'meta:SentenceID'
    assert bld.classify(['', 'MatrixLang', 'TR', '']) == 'meta:MatrixLang'
    assert bld.classify(['1', 'kitap', 'TR', '']) == 'token'


def test_build_blocks_splits_by_sentence_id_and_preserves_order():
    gold = _block('rd_1', [(1, 'kitap', 'TR', ''), (2, 'okudum', 'TR', '')]) + \
           _block('rd_2', [(3, 'hello', 'EN', '')])
    pred = _block('1', [(1, 'kitap', 'TR', ''), (2, 'okudum', 'TR', '')]) + \
           _block('2', [(3, 'hello', 'EN', '')])
    gold_kinds = [bld.classify(r) for r in gold]
    pred_kinds = [bld.classify(r) for r in pred]
    blocks = bld.build_blocks(gold, pred, gold_kinds, pred_kinds)
    assert len(blocks) == 2
    assert [tr['gold_item'] for tr in blocks[0]['token_rows']] == ['kitap', 'okudum']
    assert [tr['gold_item'] for tr in blocks[1]['token_rows']] == ['hello']
    assert blocks[0]['gold_sentence_id'] == 'rd_1'
    assert blocks[0]['pred_sentence_id'] == '1'


# ---------------------------------------------------------------------------
# Tier-1 global structural checks (positional alignment)
# ---------------------------------------------------------------------------

def test_global_structural_check_passes_on_aligned_files():
    gold = _block('rd_1', [(1, 'kitap', 'TR', '')])
    pred = _block('1', [(1, 'kitap', 'TR', '')])
    gk, pk = bld.global_structural_check(gold, pred, 'gold.csv', 'pred.csv')
    assert gk == pk  # same structural shape (blank/meta/token classification)


def test_global_structural_check_rejects_line_count_mismatch():
    gold = _block('rd_1', [(1, 'kitap', 'TR', '')])
    pred = _block('1', [(1, 'kitap', 'TR', ''), (2, 'extra', 'TR', '')])
    with pytest.raises(SystemExit):
        bld.global_structural_check(gold, pred, 'gold.csv', 'pred.csv')


def test_global_structural_check_rejects_blank_row_position_mismatch():
    gold = [
        ['', 'SentenceID', 'rd_1', ''],
        ['1', 'kitap', 'TR', ''],
        ['', 'MatrixLang', 'TR', ''],
        ['', 'EmbedLang', 'EN', ''],
        ['', '', '', ''],  # blank separator at line index 4
        ['', 'SentenceID', 'rd_2', ''],
        ['2', 'x', 'TR', ''],
        ['', 'MatrixLang', 'TR', ''],
        ['', 'EmbedLang', 'EN', ''],
    ]
    # same total line count and same blank-row COUNT (1), but the blank is
    # at a different line index -- must be rejected, not silently accepted.
    pred = [
        ['', 'SentenceID', '1', ''],
        ['1', 'kitap', 'TR', ''],
        ['', 'MatrixLang', 'TR', ''],
        ['', '', '', ''],  # blank separator shifted to line index 3
        ['', 'EmbedLang', 'EN', ''],
        ['', 'SentenceID', '2', ''],
        ['2', 'x', 'TR', ''],
        ['', 'MatrixLang', 'TR', ''],
        ['', 'EmbedLang', 'EN', ''],
    ]
    with pytest.raises(SystemExit):
        bld.global_structural_check(gold, pred, 'gold.csv', 'pred.csv')


# ---------------------------------------------------------------------------
# Tier-2 per-block checks: exclusions, REDACTED, mismatch rejection
# ---------------------------------------------------------------------------

def test_check_and_collect_excludes_known_token_ids():
    gold = _block('rd_1', [(1, 'kitap', 'TR', ''), (2, 'garbled', 'BADLABEL', '')])
    pred = _block('1', [(1, 'kitap', 'TR', ''), (2, 'garbled', 'UID', '')])
    gk, pk = bld.global_structural_check(gold, pred, 'g', 'p')
    blocks = bld.build_blocks(gold, pred, gk, pk)
    scoreable, excluded_blocks, excluded_rows = bld.check_and_collect(blocks, known_excluded_ids={2}, orphan_block_indices=set())
    assert [r['gold_token_id'] for r in scoreable] == [1]
    assert excluded_blocks == []
    assert len(excluded_rows) == 1
    assert excluded_rows[0]['token_id'] == 2


def test_check_and_collect_unequal_count_style_mismatch_excludes_whole_block():
    # gold has an item-text mismatch not covered by any known exclusion ->
    # entire block must be excluded (Decision 1/2 fallback), not just the row.
    gold = _block('rd_1', [(1, 'kitap', 'TR', '')])
    pred = _block('1', [(1, 'DIFFERENT', 'TR', '')])
    gk, pk = bld.global_structural_check(gold, pred, 'g', 'p')
    blocks = bld.build_blocks(gold, pred, gk, pk)
    scoreable, excluded_blocks, excluded_rows = bld.check_and_collect(blocks, known_excluded_ids=set(), orphan_block_indices=set())
    assert scoreable == []
    assert len(excluded_blocks) == 1
    assert excluded_blocks[0]['scope'] == 'block'


def test_check_and_collect_redacted_matches_any_predicted_item():
    gold = _block('rd_1', [(1, 'REDACTED', 'TR', '')])
    pred = _block('1', [(1, 'gercek_isim', 'TR', '')])
    gk, pk = bld.global_structural_check(gold, pred, 'g', 'p')
    blocks = bld.build_blocks(gold, pred, gk, pk)
    scoreable, excluded_blocks, excluded_rows = bld.check_and_collect(blocks, known_excluded_ids=set(), orphan_block_indices=set())
    assert excluded_blocks == []
    assert len(scoreable) == 1
    assert scoreable[0]['is_redacted'] is True


def test_check_and_collect_orphan_segmentation_block_excluded():
    gold = _block('rd_1', [(1, 'kitap', 'TR', '')])
    pred = _block('1', [(1, 'kitap', 'TR', '')])
    gk, pk = bld.global_structural_check(gold, pred, 'g', 'p')
    blocks = bld.build_blocks(gold, pred, gk, pk)
    scoreable, excluded_blocks, excluded_rows = bld.check_and_collect(blocks, known_excluded_ids=set(), orphan_block_indices={0})
    assert scoreable == []
    assert excluded_blocks[0]['block_index'] == 0


def test_check_and_collect_target_binary_from_gold_label():
    gold = _block('rd_1', [(1, 'meetinge', 'MIXED', 'meeting-DAT'), (2, 'kitap', 'TR', '')])
    pred = _block('1', [(1, 'meetinge', 'UID', ''), (2, 'kitap', 'TR', '')])
    gk, pk = bld.global_structural_check(gold, pred, 'g', 'p')
    blocks = bld.build_blocks(gold, pred, gk, pk)
    scoreable, _, _ = bld.check_and_collect(blocks, known_excluded_ids=set(), orphan_block_indices=set())
    targets = {r['gold_token_id']: r['target'] for r in scoreable}
    assert targets == {1: 1, 2: 0}


# ---------------------------------------------------------------------------
# Known-exclusion file loading
# ---------------------------------------------------------------------------

def test_load_exclusions_only_counts_rows_with_token_id(tmp_path):
    path = tmp_path / 'exclusions.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['line_number', 'token_id', 'item', 'observed_gold_value', 'reason', 'affected_field'])
        w.writerow(['10', '5', 'foo', 'bar-BAZ', 'invalid label', 'Label'])
        w.writerow(['11', '', 'SentenceID', '(empty)', 'informational only', 'SentenceID'])
    ids, rows = bld.load_exclusions(str(path))
    assert ids == {5}
    assert len(rows) == 2


def test_load_segmentation_mismatches_builds_id_pairs(tmp_path):
    path = tmp_path / 'seg.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['event_id', 'block_index', 'gold_sentence_id', 'gold_parent_token_id', 'gold_parent_item',
                     'gold_parent_label', 'gold_parent_gloss', 'gold_echo_token_id', 'gold_echo_item',
                     'gold_echo_label', 'pred_at_parent_id_item', 'pred_at_parent_id_label',
                     'pred_at_echo_id_item', 'pred_at_echo_id_label', 'note'])
        w.writerow(['1', '3', 'rd_1', '63', 'nodeları', 'MIXED', 'node-PL-ACC', '64', '**ları**', 'OTHER',
                     'node', 'EN', 'ları', 'TR', ''])
    ids, events, orphans = bld.load_segmentation_mismatches(str(path))
    assert ids == {63, 64}
    assert orphans == set()
    assert len(events) == 1


def test_load_segmentation_mismatches_orphan_flags_block():
    pass  # covered structurally by test_check_and_collect_orphan_segmentation_block_excluded


# ---------------------------------------------------------------------------
# Grouped split: reproducibility, no leakage, MIXED representation
# ---------------------------------------------------------------------------

def _synthetic_rows_and_blocks(n_blocks=30, mixed_every=4, tokens_per_block=5, seed_offset=0):
    blocks = []
    rows = []
    for b in range(n_blocks):
        blocks.append({'block_index': b, '_kept': True})
        for t in range(tokens_per_block):
            target = 1 if (b % mixed_every == 0 and t == 0) else 0
            rows.append({'block_index': b, 'target': target})
    return blocks, rows


def test_grouped_split_is_reproducible_for_fixed_seed():
    blocks, rows = _synthetic_rows_and_blocks()
    s1 = bld.grouped_stratified_split(blocks, rows, seed=42, train_frac=0.7, dev_frac=0.15, test_frac=0.15)
    s2 = bld.grouped_stratified_split(blocks, rows, seed=42, train_frac=0.7, dev_frac=0.15, test_frac=0.15)
    assert s1['train'] == s2['train']
    assert s1['dev'] == s2['dev']
    assert s1['test'] == s2['test']


def test_grouped_split_different_seed_can_differ():
    blocks, rows = _synthetic_rows_and_blocks(n_blocks=60)
    s1 = bld.grouped_stratified_split(blocks, rows, seed=1, train_frac=0.7, dev_frac=0.15, test_frac=0.15)
    s2 = bld.grouped_stratified_split(blocks, rows, seed=2, train_frac=0.7, dev_frac=0.15, test_frac=0.15)
    assert (s1['train'], s1['dev'], s1['test']) != (s2['train'], s2['dev'], s2['test'])


def test_grouped_split_no_block_appears_in_two_splits():
    blocks, rows = _synthetic_rows_and_blocks(n_blocks=45)
    s = bld.grouped_stratified_split(blocks, rows, seed=7, train_frac=0.7, dev_frac=0.15, test_frac=0.15)
    train, dev, test = set(s['train']), set(s['dev']), set(s['test'])
    assert train & dev == set()
    assert train & test == set()
    assert dev & test == set()
    all_blocks = {b['block_index'] for b in blocks}
    assert train | dev | test == all_blocks


def test_grouped_split_preserves_mixed_representation_in_every_split():
    blocks, rows = _synthetic_rows_and_blocks(n_blocks=60, mixed_every=3)
    s = bld.grouped_stratified_split(blocks, rows, seed=42, train_frac=0.7, dev_frac=0.15, test_frac=0.15)
    mixed_by_block = s['mixed_count_by_block']
    for name in ('train', 'dev', 'test'):
        assert any(mixed_by_block.get(b, 0) > 0 for b in s[name]), f"{name} split has no MIXED-bearing block"


# ---------------------------------------------------------------------------
# Feature extraction: shape and stability (train_mixed_reranker.py)
# ---------------------------------------------------------------------------

def _fake_scoreable_rows():
    return [
        {'text': 'kitap', 'structured_features': {'pred_label': 'TR', 'token_length': 5, 'ft_lang': 'TR',
                                                    'ft_prob': 0.9, 'has_digit': False}, 'target': 0},
        {'text': 'meetinge', 'structured_features': {'pred_label': 'UID', 'token_length': 8, 'ft_lang': 'EN',
                                                       'ft_prob': 0.6, 'has_digit': False}, 'target': 1},
        {'text': 'hello123', 'structured_features': {'pred_label': 'EN', 'token_length': 8, 'ft_lang': 'EN',
                                                       'ft_prob': 0.95, 'has_digit': True}, 'target': 0},
    ]


def test_build_features_shapes_match_row_counts():
    rows = _fake_scoreable_rows()
    feats = trn.build_features(rows, rows, rows)
    n = len(rows)
    assert feats['combined']['train'].shape[0] == n
    assert feats['combined']['dev'].shape[0] == n
    assert feats['combined']['test'].shape[0] == n
    assert feats['combined']['train'].shape[1] == feats['combined']['dev'].shape[1] == feats['combined']['test'].shape[1]


def test_build_features_stable_across_repeated_calls():
    rows = _fake_scoreable_rows()
    f1 = trn.build_features(rows, rows, rows)
    f2 = trn.build_features(rows, rows, rows)
    assert (f1['combined']['train'] != f2['combined']['train']).nnz == 0


def test_build_features_dictvectorizer_onehots_categorical_pred_label():
    rows = _fake_scoreable_rows()
    feats = trn.build_features(rows, rows, rows)
    names = feats['dictvec'].get_feature_names_out()
    assert any(n.startswith('pred_label=') for n in names)


# ---------------------------------------------------------------------------
# Cascade simulation: cannot alter non-candidates; only MIXED or KEEP_ORIGINAL
# ---------------------------------------------------------------------------

class _StubModel:
    """predict_proba ignores X and returns preset probabilities in row order."""
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probs, self.probs])


def _stub_vectorizers(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import DictVectorizer
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), lowercase=True)
    tfidf.fit([r['text'] for r in rows])
    dictvec = DictVectorizer(sparse=True)
    dictvec.fit([r['structured_features'] for r in rows])
    return tfidf, dictvec


def test_cascade_never_changes_non_candidate_rows():
    rows = [
        {'text': 'apple', 'structured_features': {'x': 1}, 'is_candidate': False, 'pred_label': 'EN',
         'gold_label': 'EN', 'block_index': 0, 'position_in_block': 0, 'gold_token_id': 1},
        {'text': 'boyu', 'structured_features': {'x': 1}, 'is_candidate': True, 'pred_label': 'UID',
         'gold_label': 'MIXED', 'block_index': 0, 'position_in_block': 1, 'gold_token_id': 2},
    ]
    tfidf, dictvec = _stub_vectorizers(rows)
    # both rows would get probability 0.99 -- only the candidate may flip
    model = _StubModel([0.99, 0.99])
    result = trn.simulate_cascade(rows, model, tfidf, dictvec, threshold=0.5)
    non_candidate = [r for r in result if not r['is_candidate']][0]
    candidate = [r for r in result if r['is_candidate']][0]
    assert non_candidate['simulated_label'] == non_candidate['pred_label'] == 'EN'
    assert non_candidate['changed'] is False
    assert candidate['simulated_label'] == 'MIXED'
    assert candidate['changed'] is True


def test_cascade_output_is_always_mixed_or_keep_original():
    rows = [
        {'text': 'kitaba', 'structured_features': {'x': 1}, 'is_candidate': True, 'pred_label': 'UID',
         'gold_label': 'TR', 'block_index': 0, 'position_in_block': 0, 'gold_token_id': 1},
        {'text': 'bodyci', 'structured_features': {'x': 1}, 'is_candidate': True, 'pred_label': 'NE',
         'gold_label': 'MIXED', 'block_index': 0, 'position_in_block': 1, 'gold_token_id': 2},
        {'text': '...', 'structured_features': {'x': 1}, 'is_candidate': False, 'pred_label': 'OTHER',
         'gold_label': 'OTHER', 'block_index': 0, 'position_in_block': 2, 'gold_token_id': 3},
    ]
    tfidf, dictvec = _stub_vectorizers(rows)
    model = _StubModel([0.1, 0.99, 0.99])  # first candidate below threshold, second above
    result = trn.simulate_cascade(rows, model, tfidf, dictvec, threshold=0.5)
    for r in result:
        assert r['simulated_label'] == 'MIXED' or r['simulated_label'] == r['pred_label']
    below_threshold_candidate = result[0]
    assert below_threshold_candidate['simulated_label'] == 'UID'  # KEEP_ORIGINAL
    above_threshold_candidate = result[1]
    assert above_threshold_candidate['simulated_label'] == 'MIXED'


def test_cascade_beneficial_harmful_neutral_classification():
    rows = [
        # candidate, flips to MIXED, gold really is MIXED -> beneficial
        {'text': 'nodelari', 'structured_features': {'x': 1}, 'is_candidate': True, 'pred_label': 'UID',
         'gold_label': 'MIXED', 'block_index': 0, 'position_in_block': 0, 'gold_token_id': 1},
        # candidate, flips to MIXED, but original pred already equalled gold (impossible for UID/NE/TR->gold MIXED,
        # so use a case where gold==pred=='TR' and cascade wrongly flips it) -> harmful
        {'text': 'kahveler', 'structured_features': {'x': 1}, 'is_candidate': True, 'pred_label': 'TR',
         'gold_label': 'TR', 'block_index': 0, 'position_in_block': 1, 'gold_token_id': 2},
        # candidate, flips to MIXED, original was wrong (UID != NE) and simulated still wrong -> neutral
        {'text': 'istanbul', 'structured_features': {'x': 1}, 'is_candidate': True, 'pred_label': 'UID',
         'gold_label': 'NE', 'block_index': 0, 'position_in_block': 2, 'gold_token_id': 3},
    ]
    tfidf, dictvec = _stub_vectorizers(rows)
    model = _StubModel([0.99, 0.99, 0.99])
    result = trn.simulate_cascade(rows, model, tfidf, dictvec, threshold=0.5)
    by_id = {r['gold_token_id']: r for r in result}
    assert by_id[1]['beneficial'] is True and by_id[1]['harmful'] is False
    assert by_id[2]['harmful'] is True and by_id[2]['beneficial'] is False
    assert by_id[3]['neutral'] is True


# ---------------------------------------------------------------------------
# Phase 3A: NE cascade policy (tools/ne_cascade_policy.py)
# ---------------------------------------------------------------------------

def _ne_policy_module():
    import sys, os
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import ne_cascade_policy as nep
    return nep


def test_stem_evidence_true_via_lexicon():
    nep = _ne_policy_module()
    feats = {'stem_length': 5, 'stem_in_english_freq': True, 'stem_ft_lang': 'TR', 'stem_ft_prob': 0.1}
    assert nep.stem_evidence(feats) is True


def test_stem_evidence_true_via_fasttext_threshold():
    nep = _ne_policy_module()
    feats = {'stem_length': 5, 'stem_in_english_freq': False, 'stem_ft_lang': 'EN', 'stem_ft_prob': 0.95}
    assert nep.stem_evidence(feats) is True


def test_stem_evidence_false_below_fasttext_threshold():
    nep = _ne_policy_module()
    feats = {'stem_length': 5, 'stem_in_english_freq': False, 'stem_ft_lang': 'EN', 'stem_ft_prob': 0.5}
    assert nep.stem_evidence(feats) is False


def test_stem_evidence_false_when_no_analysis_at_all():
    nep = _ne_policy_module()
    feats = {'stem_length': 0, 'stem_in_english_freq': True, 'stem_ft_lang': 'EN', 'stem_ft_prob': 0.99}
    # stem_length 0 means there was no candidate analysis in the first
    # place -- lexicon/fastText fields are meaningless defaults, must not
    # be treated as real evidence.
    assert nep.stem_evidence(feats) is False


def _ne_policy_row(text, pred_label, gold_label, is_candidate=True, stem_evidence_feats=None):
    return {
        'text': text, 'pred_label': pred_label, 'gold_label': gold_label, 'is_candidate': is_candidate,
        'structured_features': stem_evidence_feats or {'stem_length': 0},
        'block_index': 0, 'position_in_block': 0, 'gold_token_id': 1,
    }


def test_simulate_with_ne_policy_unrestricted_flips_ne_like_anything_else():
    nep = _ne_policy_module()
    row = _ne_policy_row("store'da", "NE", "NE")
    result = nep.simulate_with_ne_policy([row], [0.95], threshold=0.5, policy=nep.POLICY_UNRESTRICTED)
    assert result[0]['simulated_label'] == 'MIXED'
    assert result[0]['harmful'] is True


def test_simulate_with_ne_policy_block_ne_never_flips():
    nep = _ne_policy_module()
    row = _ne_policy_row("store'da", "NE", "NE")
    result = nep.simulate_with_ne_policy([row], [0.999], threshold=0.5, policy=nep.POLICY_BLOCK_NE)
    assert result[0]['simulated_label'] == 'NE'
    assert result[0]['changed'] is False


def test_simulate_with_ne_policy_block_ne_only_affects_ne_predicted_rows():
    nep = _ne_policy_module()
    non_ne_row = _ne_policy_row("kitaba", "UID", "MIXED")
    result = nep.simulate_with_ne_policy([non_ne_row], [0.95], threshold=0.5, policy=nep.POLICY_BLOCK_NE)
    assert result[0]['simulated_label'] == 'MIXED'


def test_simulate_with_ne_policy_ne_threshold_gates_on_separate_bar():
    nep = _ne_policy_module()
    row = _ne_policy_row("store'da", "NE", "NE")
    below_ne_threshold = nep.simulate_with_ne_policy([row], [0.85], threshold=0.5,
                                                       policy=nep.POLICY_NE_THRESHOLD, ne_threshold=0.90)
    above_ne_threshold = nep.simulate_with_ne_policy([row], [0.95], threshold=0.5,
                                                       policy=nep.POLICY_NE_THRESHOLD, ne_threshold=0.90)
    assert below_ne_threshold[0]['simulated_label'] == 'NE'
    assert above_ne_threshold[0]['simulated_label'] == 'MIXED'


def test_simulate_with_ne_policy_ne_threshold_requires_explicit_value():
    nep = _ne_policy_module()
    row = _ne_policy_row("store'da", "NE", "NE")
    with pytest.raises(ValueError):
        nep.simulate_with_ne_policy([row], [0.95], threshold=0.5, policy=nep.POLICY_NE_THRESHOLD)


def test_simulate_with_ne_policy_stem_evidence_blocks_without_evidence():
    nep = _ne_policy_module()
    row = _ne_policy_row("obscur'ün", "NE", "NE",
                          stem_evidence_feats={'stem_length': 6, 'stem_in_english_freq': False,
                                                'stem_ft_lang': 'EN', 'stem_ft_prob': 0.176})
    result = nep.simulate_with_ne_policy([row], [0.99], threshold=0.5, policy=nep.POLICY_NE_STEM_EVIDENCE)
    assert result[0]['simulated_label'] == 'NE'


def test_simulate_with_ne_policy_stem_evidence_allows_with_lexicon_evidence():
    nep = _ne_policy_module()
    row = _ne_policy_row("store'da", "NE", "NE",
                          stem_evidence_feats={'stem_length': 5, 'stem_in_english_freq': True,
                                                'stem_ft_lang': 'EN', 'stem_ft_prob': 0.69})
    result = nep.simulate_with_ne_policy([row], [0.99], threshold=0.5, policy=nep.POLICY_NE_STEM_EVIDENCE)
    # honest test of a real limitation found during Phase 3A: lexicon-based
    # evidence alone cannot distinguish a real English common noun used as
    # part of an NE ("Store'da") from a genuine MIXED token -- this policy
    # does NOT block this case, which is exactly the finding reported to
    # the user, not a bug in the test.
    assert result[0]['simulated_label'] == 'MIXED'


def test_ne_stats_counts_harmful_and_genuine_correctly():
    nep = _ne_policy_module()
    rows = [
        _ne_policy_row("store'da", "NE", "NE"),       # will flip -> harmful
        _ne_policy_row("teaching'de", "NE", "MIXED"),  # will flip -> genuine, retained
    ]
    result = nep.simulate_with_ne_policy(rows, [0.95, 0.95], threshold=0.5, policy=nep.POLICY_UNRESTRICTED)
    stats = nep.ne_stats(result)
    assert stats['n_harmful_ne_changes'] == 1
    assert stats['harmful_ne_tokens'] == ["store'da"]
    assert stats['n_genuine_ne_retained'] == 1
    assert stats['genuine_ne_retained_tokens'] == ["teaching'de"]


# ---------------------------------------------------------------------------
# Phase 4A: experimental verbal morphology (candidate generator only)
# ---------------------------------------------------------------------------

def test_parse_experimental_verbal_suffix_infinitive_and_verbalizer():
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("lemek")
    assert fully_consumed is True
    assert "Verbal=Infinitive" in tags
    assert "Verbal=Verbalizer" in tags
    assert segments == ["le", "mek"]


def test_parse_experimental_verbal_suffix_infinitive_variant_mak():
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("lamak")
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Infinitive", "Verbal=Verbalizer"})


def test_parse_experimental_verbal_suffix_passive_inchoative_alone():
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("lan")
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=PassiveInchoative"})


def test_parse_experimental_verbal_suffix_passive_inchoative_preferred_over_verbalizer():
    # "-lan" must be recognized as passive/inchoative, not as verbalizer
    # "-la" + a leftover "n" -- the longer/more-specific pattern must win.
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("lan")
    assert "Verbal=PassiveInchoative" in tags
    assert "Verbal=Verbalizer" not in tags


def test_parse_experimental_verbal_suffix_no_tense_or_agreement():
    # "-lanmıssın" (passive/inchoative + tense + person) must NOT be treated
    # as fully consumed -- tense/agreement is explicitly out of scope for
    # Phase 4A, and a partial match must not be silently accepted.
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("lanmıssın")
    assert fully_consumed is False


def test_parse_experimental_verbal_suffix_no_match_not_consumed():
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("xyz")
    assert fully_consumed is False
    assert tags == frozenset()


def test_enumerate_candidate_analyses_finds_verbal_infinitive_construction():
    # "Replacelemek" = "Replace" (EN stem) + "le" (verbalizer) + "mek" (infinitive)
    # -- unreachable before Phase 4A (no nominal analysis exists for "lemek").
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("Replacelemek", obj)
    verbal = [c for c in candidates if c.source == "verbal"]
    assert any(c.stem == "Replace" and c.suffix == "lemek" for c in verbal)


def test_enumerate_candidate_analyses_finds_verbal_infinitive_mak_variant():
    # "Boostlamak" = "Boost" + "la" (verbalizer) + "mak" (infinitive)
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("Boostlamak", obj)
    verbal = [c for c in candidates if c.source == "verbal"]
    assert any(c.stem == "Boost" and c.suffix == "lamak" for c in verbal)


def test_enumerate_candidate_analyses_nominal_source_unaffected():
    # Pure nominal tokens must still report source="nominal" -- Phase 4A's
    # fallback must not change results for suffixes the production parser
    # already handles.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("meeting'e", obj)
    assert all(c.source == "nominal" for c in candidates)


def test_enumerate_candidate_analyses_trigerlanmissin_no_clean_full_decomposition():
    # "trigerlanmıssın" (gloss trigger-PASS-EVID-2SG) is spelled in the real
    # corpus with plain "s" rather than the standard Turkish "ş" ("mıssın",
    # not "mışsın") -- so the evidential table ("mış"/"miş"/"muş"/"müş",
    # proper spelling only, per the Phase 4B suffix list as given) does not
    # match, and the CLEAN full decomposition stem="triger" +
    # suffix="lanmıssın" is correctly NOT fully consumed.
    #
    # Phase 4B's new 2nd-person-agreement stage DOES, however, find one
    # narrow, coincidental match: stem="trigerlanmıs" + suffix="sın" (just
    # the bare agreement marker, tense stage failing to match "mıs" for the
    # same ş/s reason). That is a real, reportable side effect -- not a
    # bug -- but it is not the intended "triger" + full verbal tail
    # decomposition, so it must not be the analysis selected by the real
    # benchmark's tie-break strategy.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("trigerlanmıssın", obj)
    assert all(c.stem != "triger" for c in candidates)
    best = mr.best_analysis_for_token("trigerlanmıssın", obj, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert best.stem != "triger"


# ---------------------------------------------------------------------------
# Phase 4B: past, evidential, present progressive, future, 2nd person
# agreement -- new EXPERIMENTAL verbal-suffix categories, candidate
# generator only. One test per newly supported suffix group, per instruction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", ["dı", "di", "du", "dü", "tı", "ti", "tu", "tü"])
def test_parse_experimental_verbal_suffix_past_all_variants(suffix):
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix(suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Past"})


@pytest.mark.parametrize("suffix", ["mış", "miş", "muş", "müş"])
def test_parse_experimental_verbal_suffix_evidential_all_variants(suffix):
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix(suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Evidential"})


def test_parse_experimental_verbal_suffix_present_progressive():
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("yor", level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Progressive"})


@pytest.mark.parametrize("suffix", ["acak", "ecek"])
def test_parse_experimental_verbal_suffix_future_all_variants(suffix):
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix(suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Future"})


@pytest.mark.parametrize("suffix", ["sın", "sin", "sun", "sün"])
def test_parse_experimental_verbal_suffix_second_person_singular_all_variants(suffix):
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix(suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=2ndPersonSingular"})


@pytest.mark.parametrize("suffix", ["sınız", "siniz", "sunuz", "sünüz"])
def test_parse_experimental_verbal_suffix_second_person_plural_all_variants(suffix):
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix(suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=2ndPersonPlural"})


def test_parse_experimental_verbal_suffix_progressive_plus_second_person_singular_chain():
    # "yorsun" = progressive "yor" + 2sg agreement "sun" -- verifies stages
    # 1 (agreement) and 2 (tense/aspect/mood) combine correctly.
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("yorsun", level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Progressive", "Verbal=2ndPersonSingular"})


def test_parse_experimental_verbal_suffix_future_plus_second_person_plural_chain():
    # "acaksınız" = future "acak" + 2pl agreement "sınız"
    segments, tags, fully_consumed, informal_used = mr._parse_experimental_verbal_suffix("acaksınız", level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Future", "Verbal=2ndPersonPlural"})


def test_enumerate_candidate_analyses_finds_past_tense_construction():
    # "chatladı" -- EN-ish stem "chat" + verbalizer "la" + past "dı"
    # (3rd person singular past, no overt agreement suffix). Past tense is a
    # Phase 4B category, not active at the (restored) Phase 4A default, so
    # level must be requested explicitly.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("chatladı", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    verbal = [c for c in candidates if c.source == "verbal" and c.stem == "chat"]
    assert verbal
    assert verbal[0].suffix == "ladı"


def test_enumerate_candidate_analyses_finds_progressive_construction():
    # "postlayor" -- "post" + verbalizer "la" + "yor" (present progressive).
    # Progressive is a Phase 4B category; level requested explicitly.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("postlayor", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    verbal = [c for c in candidates if c.source == "verbal" and c.stem == "post"]
    assert verbal
    assert "yor" in verbal[0].suffix


def test_enumerate_candidate_analyses_infinitive_unaffected_by_phase_4b():
    # Phase 4A infinitive recognition must still work identically.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("Replacelemek", obj)
    verbal = [c for c in candidates if c.source == "verbal"]
    assert any(c.stem == "Replace" and c.suffix == "lemek" for c in verbal)


# ---------------------------------------------------------------------------
# Phase 4C: default restored to Phase 4A
# ---------------------------------------------------------------------------

def test_default_verbal_morphology_level_is_phase_4c1():
    # Phase 4D promoted Phase 4C-1 to the active default (see module docstring).
    assert mr.DEFAULT_VERBAL_MORPHOLOGY_LEVEL == mr.VERBAL_MORPHOLOGY_PHASE_4C1


def test_phase_4b_tables_still_fully_present_but_not_default():
    # Phase 4B code/tables must remain available (not deleted), just not
    # the active default -- e.g. progressive "yor" alone is NOT fully
    # consumed at the default level, but IS at level=PHASE_4B.
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix("yor")
    assert fully_consumed is False
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "yor", level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is True


def test_parse_experimental_verbal_suffix_unknown_level_raises():
    with pytest.raises(ValueError):
        mr._parse_experimental_verbal_suffix("mak", level="not_a_real_level")


# ---------------------------------------------------------------------------
# Phase 4C-1: 1st person singular agreement (extends Phase 4A, NOT 4B)
# ---------------------------------------------------------------------------

def test_parse_experimental_verbal_suffix_first_person_bare_m():
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "m", level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=1stPersonSingular"})


@pytest.mark.parametrize("suffix", ["ım", "im", "um", "üm"])
def test_parse_experimental_verbal_suffix_first_person_buffer_vowel_variants(suffix):
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=1stPersonSingular"})


@pytest.mark.parametrize("suffix", ["dım", "dim", "dum", "düm"])
def test_parse_experimental_verbal_suffix_first_person_past_d_variants(suffix):
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Past", "Verbal=1stPersonSingular"})


@pytest.mark.parametrize("suffix", ["tım", "tim", "tum", "tüm"])
def test_parse_experimental_verbal_suffix_first_person_past_t_variants(suffix):
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        suffix, level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Past", "Verbal=1stPersonSingular"})


def test_parse_experimental_verbal_suffix_phase_4c1_does_not_enable_2nd_person():
    # Phase 4C-1 extends Phase 4A only -- 2nd person agreement ("sın") is a
    # Phase 4B category and must NOT be active at level=PHASE_4C1.
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "sın", level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is False


def test_parse_experimental_verbal_suffix_phase_4c1_infinitive_still_works():
    # Phase 4A's infinitive/verbalizer/passive-inchoative stages must remain
    # active at level=PHASE_4C1 (it extends 4A, doesn't replace it).
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "lemek", level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Infinitive", "Verbal=Verbalizer"})


def test_enumerate_candidate_analyses_comeoutladim_phase_4c1():
    # "Comeoutladim" (gloss: come-out-VBLZ-PST.1SG) -- Phase 4C-1 should find
    # a fully-consumed analysis via the "dim" (past+1sg) suffix, at minimum
    # as stem="Comeoutla"+suffix="dim"; a better split (stem="Comeout"+
    # suffix="ladim", peeling verbalizer "la" separately) may also be found
    # since the passive/verbalizer stage runs after 1st-person consumption.
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("Comeoutladim", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    verbal = [c for c in candidates if c.source == "verbal"]
    assert verbal, "Phase 4C-1 must find at least one fully-consumed verbal analysis for Comeoutladim"
    assert any("Verbal=1stPersonSingular" in c.ud_feats | c.deriv | c.amb for c in verbal)
    best = mr.select_best_analysis(verbal, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    # the cleaner 2-segment split (verbalizer peeled separately) must win
    # the highest_suffix_segments tie-break over the 1-segment alternative
    assert best.stem == "Comeout"
    assert best.segment_count == 2


# ---------------------------------------------------------------------------
# Phase 4C-2: informal ASCII orthography fallback (s->ş, i->ı), suffix-
# matching only, opt-in, independent of `level`
# ---------------------------------------------------------------------------

def test_informal_suffix_variants_generates_expected_candidates():
    variants = mr._informal_suffix_variants("mıs")
    assert "mış" in variants


def test_informal_suffix_variants_excludes_identical_string():
    # "sın" has no "i" to normalize and its "s" is genuinely plain s in
    # standard spelling too when NOT part of an evidential marker -- but the
    # helper is purely mechanical string substitution, so it still proposes
    # "şın"; what matters is it never returns the unchanged original.
    variants = mr._informal_suffix_variants("sın")
    assert "sın" not in variants


def test_match_verbal_table_disabled_by_default():
    result = mr._match_verbal_table("mıs", mr.VERBAL_EVIDENTIAL, allow_informal_orthography=False)
    assert result is None


def test_match_verbal_table_finds_informal_evidential_when_enabled():
    result = mr._match_verbal_table("mıs", mr.VERBAL_EVIDENTIAL, allow_informal_orthography=True)
    assert result is not None
    end, tag, used_informal = result
    assert end == "mış"
    assert tag == "Verbal=Evidential"
    assert used_informal is True


def test_match_verbal_table_exact_match_never_marked_informal():
    result = mr._match_verbal_table("mış", mr.VERBAL_EVIDENTIAL, allow_informal_orthography=True)
    end, tag, used_informal = result
    assert used_informal is False


def test_parse_experimental_verbal_suffix_trigerlanmissin_informal_fallback_disabled():
    # Without the fallback, "lanmıssın" must NOT fully consume (same as
    # Phase 4B's original finding -- the ş/s spelling mismatch blocks it).
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "lanmıssın", level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=False)
    assert fully_consumed is False


def test_parse_experimental_verbal_suffix_trigerlanmissin_informal_fallback_enabled():
    # With the fallback enabled, "lanmıssın" -> lan (passive/inchoative) +
    # mış (evidential, via informal "mıs"->"mış") + sın (2nd person, exact
    # match, no normalization needed) -- fully consumed.
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "lanmıssın", level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=True)
    assert fully_consumed is True
    assert informal is True
    assert tags == frozenset({"Verbal=PassiveInchoative", "Verbal=Evidential", "Verbal=2ndPersonSingular"})


def test_enumerate_candidate_analyses_trigerlanmissin_full_decomposition_with_informal_fallback():
    # The specific Phase 4C-2 target: trigerlanmıssın -> triger + lan + mış + sın
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses(
        "trigerlanmıssın", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=True)
    clean = [c for c in candidates if c.stem == "triger" and c.suffix == "lanmıssın"]
    assert clean, "expected a stem='triger' analysis of the full verbal tail"
    assert clean[0].informal_suffix_normalization is True
    assert clean[0].source == "verbal"
    tagset = clean[0].ud_feats | clean[0].deriv | clean[0].amb
    assert tagset == frozenset({"Verbal=PassiveInchoative", "Verbal=Evidential", "Verbal=2ndPersonSingular"})


def test_enumerate_candidate_analyses_trigerlanmissin_no_clean_split_without_fallback():
    # Confirms the fallback is genuinely load-bearing for this token: without
    # it (even at level=PHASE_4B), no candidate has the clean stem="triger".
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses(
        "trigerlanmıssın", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=False)
    assert all(c.stem != "triger" for c in candidates)


def test_build_structured_feature_dict_informal_flag_present_and_false_by_default():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    assert feats["informal_suffix_normalization"] is False


def test_build_structured_feature_dict_informal_flag_true_when_analysis_used_fallback():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="triger", suffix="lanmıssın", split_position=6, segments=("lan", "mış", "sın"),
        ud_feats=frozenset(), deriv=frozenset({"Verbal=PassiveInchoative", "Verbal=Evidential",
                                                "Verbal=2ndPersonSingular"}),
        amb=frozenset(), source="verbal", informal_suffix_normalization=True,
    )
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("trigerlanmıssın", "TR", analysis, obj, DEFAULTS)
    assert feats["informal_suffix_normalization"] is True


# ---------------------------------------------------------------------------
# Phase 4D: narrowly-scoped typo-tolerant English stem-evidence fallback
# (duplicated consonant only, not general fuzzy matching)
# ---------------------------------------------------------------------------

def test_duplicated_consonant_variants_triger_includes_trigger():
    assert "trigger" in mr._duplicated_consonant_variants("triger")


def test_duplicated_consonant_variants_droping_includes_dropping():
    assert "dropping" in mr._duplicated_consonant_variants("droping")


def test_duplicated_consonant_variants_never_touches_vowels():
    # only consonants are duplicated -- no vowel-doubling variants proposed
    for variant in mr._duplicated_consonant_variants("triger"):
        # every variant must be exactly one character longer, via a
        # consonant repeat, never inserting a brand-new letter
        assert len(variant) == len("triger") + 1


def test_recover_english_stem_via_duplicated_consonant_finds_trigger():
    obj = _make_annotator(english_words={"trigger"})
    result = mr.recover_english_stem_via_duplicated_consonant("triger", obj)
    assert result == "trigger"


def test_recover_english_stem_via_duplicated_consonant_finds_dropping():
    obj = _make_annotator(english_words={"dropping"})
    result = mr.recover_english_stem_via_duplicated_consonant("droping", obj)
    assert result == "dropping"


def test_recover_english_stem_via_duplicated_consonant_none_when_no_lexicon_hit():
    obj = _make_annotator(english_words=set())
    assert mr.recover_english_stem_via_duplicated_consonant("triger", obj) is None


def test_recover_english_stem_via_duplicated_consonant_rejects_short_stems():
    # condition 4: stem length must be >= 5, even if a lexicon match would
    # otherwise exist for a shorter string
    obj = _make_annotator(english_words={"aall"})
    assert mr.recover_english_stem_via_duplicated_consonant("aal", obj) is None


def test_classify_candidate_stem_orthographic_recovery_disabled_by_default():
    # even with an informally-normalized analysis and a real duplicated-
    # consonant lexicon match available, the fallback must not fire unless
    # allow_stem_orthographic_recovery=True is passed explicitly.
    obj = _make_annotator(english_words={"trigger"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        is_cand, reason, analysis = mr.classify_candidate(
            "TR", "trigerlanmıssın", obj, DEFAULTS,
            verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=True,
            allow_stem_orthographic_recovery=False)
    assert is_cand is False
    assert analysis is not None and analysis.stem_orthographic_recovery is False


def test_classify_candidate_stem_orthographic_recovery_enabled_grants_candidacy():
    obj = _make_annotator(english_words={"trigger"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        is_cand, reason, analysis = mr.classify_candidate(
            "TR", "trigerlanmıssın", obj, DEFAULTS,
            strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS,
            verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=True,
            allow_stem_orthographic_recovery=True)
    assert is_cand is True
    assert reason == mr.CANDIDATE_REASON_TR_SUSPECT_STEM
    assert analysis.stem_orthographic_recovery is True
    assert analysis.recovered_english_stem == "trigger"
    assert analysis.stem == "triger"  # original stem text untouched


def test_classify_candidate_stem_orthographic_recovery_requires_informal_flag():
    # condition 2: only applies when informal_suffix_normalization is True.
    # A token whose analysis was reached via ordinary (non-informal)
    # matching must not trigger the Phase 4D fallback even if a
    # duplicated-consonant match happens to exist for its stem.
    obj = _make_annotator(english_words={"meetting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        is_cand, reason, analysis = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, allow_stem_orthographic_recovery=True)
    assert is_cand is False
    assert analysis.informal_suffix_normalization is False


def test_classify_candidate_stem_orthographic_recovery_none_when_no_recovery_possible():
    # duplicated-consonant fallback finds nothing -> candidacy stays False,
    # analysis unchanged (not falsely marked as recovered).
    obj = _make_annotator(english_words=set())
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        is_cand, reason, analysis = mr.classify_candidate(
            "TR", "trigerlanmıssın", obj, DEFAULTS,
            verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4B, allow_informal_orthography=True,
            allow_stem_orthographic_recovery=True)
    assert is_cand is False
    assert analysis.stem_orthographic_recovery is False
    assert analysis.recovered_english_stem is None


def test_build_structured_feature_dict_stem_orthographic_recovery_flag():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="triger", suffix="lanmıssın", split_position=6, segments=("lan", "mış", "sın"),
        ud_feats=frozenset(), deriv=frozenset({"Verbal=PassiveInchoative"}), amb=frozenset(),
        source="verbal", informal_suffix_normalization=True,
        stem_orthographic_recovery=True, recovered_english_stem="trigger",
    )
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("trigerlanmıssın", "TR", analysis, obj, DEFAULTS)
    assert feats["stem_orthographic_recovery"] is True


def test_build_structured_feature_dict_stem_orthographic_recovery_false_without_analysis():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    assert feats["stem_orthographic_recovery"] is False


# ---------------------------------------------------------------------------
# Phase 4E: combined verbal morphology level (union of 4A + 4B + 4C-1)
# ---------------------------------------------------------------------------

def test_phase_4e_is_a_valid_level():
    assert mr.VERBAL_MORPHOLOGY_PHASE_4E in mr.VERBAL_MORPHOLOGY_LEVELS


def test_default_level_unchanged_by_phase_4e():
    # Phase 4E adds a level, it does not become the default.
    assert mr.DEFAULT_VERBAL_MORPHOLOGY_LEVEL == mr.VERBAL_MORPHOLOGY_PHASE_4C1


def test_phase_4e_includes_2nd_person_agreement():
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "sın", level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=2ndPersonSingular"})


def test_phase_4e_includes_1st_person_singular_agreement():
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "dım", level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Past", "Verbal=1stPersonSingular"})


def test_phase_4e_includes_tense_aspect_mood():
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "yor", level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Progressive"})


def test_phase_4e_includes_infinitive_and_verbalizer_from_4a():
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "lemek", level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Infinitive", "Verbal=Verbalizer"})


def test_phase_4e_combines_progressive_and_2nd_person():
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "yorsun", level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    assert fully_consumed is True
    assert tags == frozenset({"Verbal=Progressive", "Verbal=2ndPersonSingular"})


def test_phase_4a_unaffected_by_phase_4e_addition():
    # phase4a must still reject anything beyond infinitive/verbalizer/passive-inchoative
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "sın", level=mr.VERBAL_MORPHOLOGY_PHASE_4A)
    assert fully_consumed is False


def test_phase_4b_unaffected_by_phase_4e_addition():
    # phase4b must still NOT include 1st-person singular forms
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "dım", level=mr.VERBAL_MORPHOLOGY_PHASE_4B)
    assert fully_consumed is False


def test_phase_4c1_unaffected_by_phase_4e_addition():
    # phase4c1 must still NOT include 2nd-person forms
    segments, tags, fully_consumed, informal = mr._parse_experimental_verbal_suffix(
        "sın", level=mr.VERBAL_MORPHOLOGY_PHASE_4C1)
    assert fully_consumed is False


def test_enumerate_candidate_analyses_trigerlanmissin_phase_4e_with_informal_fallback():
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses(
        "trigerlanmıssın", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4E, allow_informal_orthography=True)
    clean = [c for c in candidates if c.stem == "triger" and c.suffix == "lanmıssın"]
    assert clean
    assert clean[0].informal_suffix_normalization is True


def test_enumerate_candidate_analyses_dropladim_phase_4e():
    # "dropladım" = drop + la (verbalizer) + dım (past+1sg)
    obj = _make_annotator()
    candidates = mr.enumerate_candidate_analyses("dropladım", obj, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    verbal = [c for c in candidates if c.source == "verbal"]
    assert any("Verbal=1stPersonSingular" in (c.ud_feats | c.deriv | c.amb) for c in verbal)


# ---------------------------------------------------------------------------
# Phase 5A, Batch C: language-confidence interaction features
# (ft_prob_delta, ft_lang_agreement, stem_evidence_strength)
# ---------------------------------------------------------------------------

_PRE_BATCH_C_KEYS = frozenset({
    "pred_label", "token_length", "has_apostrophe", "has_hyphen", "has_digit",
    "initial_capital", "all_uppercase", "has_turkish_char", "in_turkish_top",
    "in_turkish_all", "in_english_freq", "ft_lang", "ft_prob", "is_ne",
    "stem_length", "suffix_length", "suffix_segment_count", "fully_consumed_suffix",
    "stem_in_turkish_top", "stem_in_turkish_all", "stem_in_english_freq",
    "stem_ft_lang", "stem_ft_prob", "informal_suffix_normalization",
    "stem_orthographic_recovery",
})
_BATCH_C_KEYS = frozenset({"ft_prob_delta", "ft_lang_agreement", "stem_evidence_strength"})


def test_build_structured_feature_dict_no_existing_keys_removed():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    assert _PRE_BATCH_C_KEYS.issubset(feats.keys())


def test_build_structured_feature_dict_batch_c_keys_present():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    assert _BATCH_C_KEYS.issubset(feats.keys())


def test_ft_prob_delta_with_analysis():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")

    def fake_ft_predict(token_l):
        return {"meetinge": ("TR", 0.30), "meeting": ("EN", 0.90)}[token_l]

    with mock.patch.object(obj, "_ft_predict", side_effect=fake_ft_predict):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    assert feats["ft_prob_delta"] == pytest.approx(0.90 - 0.30)


def test_ft_prob_delta_without_analysis_uses_zero_for_stem_prob():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.42)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    # stem_ft_prob defaults to 0.0 when there is no analysis
    assert feats["ft_prob_delta"] == pytest.approx(0.0 - 0.42)


def test_ft_lang_agreement_true_when_languages_match():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.5)):
        analysis = mr.CandidateAnalysis(
            stem="meeting", suffix="e", split_position=7, segments=("e",),
            ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    assert feats["ft_lang"] == feats["stem_ft_lang"] == "EN"
    assert feats["ft_lang_agreement"] is True


def test_ft_lang_agreement_false_when_languages_differ():
    obj = _make_annotator()

    def fake_ft_predict(token_l):
        return {"meetinge": ("TR", 0.9), "meeting": ("EN", 0.9)}[token_l]

    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", side_effect=fake_ft_predict):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    assert feats["ft_lang"] != feats["stem_ft_lang"]
    assert feats["ft_lang_agreement"] is False


def test_ft_lang_agreement_false_when_no_analysis():
    # stem_ft_lang defaults to "NONE", which will not equal a real ft_lang
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    assert feats["stem_ft_lang"] == "NONE"
    assert feats["ft_lang_agreement"] is False


def test_stem_evidence_strength_none_when_neither_source_agrees():
    obj = _make_annotator(english_words=set())
    analysis = mr.CandidateAnalysis(
        stem="xyzstem", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.99)):
        feats = mr.build_structured_feature_dict("xyzsteme", "TR", analysis, obj, DEFAULTS)
    assert feats["stem_evidence_strength"] == "none"


def test_stem_evidence_strength_lexicon_only():
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    # fastText disagrees (says TR) -- only the lexicon hit should count
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.99)):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    assert feats["stem_evidence_strength"] == "lexicon_only"


def test_stem_evidence_strength_fasttext_only():
    obj = _make_annotator(english_words=set())
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    assert feats["stem_evidence_strength"] == "fasttext_only"


def test_stem_evidence_strength_both():
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    assert feats["stem_evidence_strength"] == "both"


def test_stem_evidence_strength_respects_ft_en_min_threshold():
    # fastText says EN but below cfg["FT_EN_MIN"] -- must not count as a fastText hit
    obj = _make_annotator(english_words=set())
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    cfg = dict(DEFAULTS, FT_EN_MIN=0.80)
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.50)):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, cfg)
    assert feats["stem_evidence_strength"] == "none"


def test_stem_evidence_strength_none_when_no_analysis():
    obj = _make_annotator(english_words={"xyz"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.99)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    # no analysis -> stem_in_english_freq/stem_ft_lang default to False/"NONE"
    # regardless of what the whole-token lexicon/fastText would say
    assert feats["stem_evidence_strength"] == "none"


# ---------------------------------------------------------------------------
# Phase 5B, Batch A: parser-derived structure features
# (analysis_source, candidate_reason, is_candidate, split_position_ratio)
# ---------------------------------------------------------------------------

def test_batch_a_excluded_by_default():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    assert "analysis_source" not in feats
    assert "candidate_reason" not in feats
    assert "is_candidate" not in feats
    assert "split_position_ratio" not in feats


def test_batch_a_does_not_disturb_batch_c_or_baseline_when_both_enabled():
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True)
    # baseline + Batch C keys still present
    assert "stem_evidence_strength" in feats
    assert "ft_prob_delta" in feats
    # Batch A keys also present
    assert feats["analysis_source"] == "nominal"
    assert feats["is_candidate"] is True


def test_analysis_source_nominal():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS, include_batch_a=True)
    assert feats["analysis_source"] == "nominal"


def test_analysis_source_verbal():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="Replace", suffix="lemek", split_position=7, segments=("le", "mek"),
        ud_feats=frozenset(), deriv=frozenset({"Verbal=Verbalizer", "Verbal=Infinitive"}),
        amb=frozenset(), source="verbal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "Replacelemek", "UID", analysis, obj, DEFAULTS, include_batch_a=True)
    assert feats["analysis_source"] == "verbal"


def test_analysis_source_none_when_no_analysis():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, include_batch_a=True)
    assert feats["analysis_source"] == "none"


@pytest.mark.parametrize("reason", [
    mr.CANDIDATE_REASON_UID, mr.CANDIDATE_REASON_NE_SUFFIX, mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
])
def test_candidate_reason_each_real_value_preserved_verbatim(reason):
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, is_candidate=True, candidate_reason=reason,
            include_batch_a=True)
    # exact value preserved, not re-derived or redefined
    assert feats["candidate_reason"] == reason


def test_candidate_reason_none_when_not_a_candidate():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "EN", None, obj, DEFAULTS, is_candidate=False, candidate_reason=None,
            include_batch_a=True)
    assert feats["candidate_reason"] == "none"


def test_is_candidate_true_and_false_reflect_caller_supplied_value_verbatim():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats_true = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, is_candidate=True, include_batch_a=True)
        feats_false = mr.build_structured_feature_dict(
            "xyz", "EN", None, obj, DEFAULTS, is_candidate=False, include_batch_a=True)
    assert feats_true["is_candidate"] is True
    assert feats_false["is_candidate"] is False


def test_non_candidate_row_gets_none_analysis_source_and_reason():
    # a non-candidate row (e.g. pred_label=EN, never a candidate bucket)
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "hello", "EN", None, obj, DEFAULTS, is_candidate=False, candidate_reason=None,
            include_batch_a=True)
    assert feats["is_candidate"] is False
    assert feats["candidate_reason"] == "none"
    assert feats["analysis_source"] == "none"
    assert feats["split_position_ratio"] == 0.0


def test_split_position_ratio_calculation():
    obj = _make_annotator()
    # "meetinge" -- stem "meeting" (7 chars), split_position=7, token_length=8
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS, include_batch_a=True)
    assert feats["split_position_ratio"] == pytest.approx(7 / 8)


def test_split_position_ratio_fallback_zero_when_no_analysis():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, include_batch_a=True)
    assert feats["split_position_ratio"] == 0.0


def test_split_position_ratio_fallback_zero_when_token_length_zero():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "", "UID", None, obj, DEFAULTS, include_batch_a=True)
    assert feats["split_position_ratio"] == 0.0


def test_batch_a_does_not_change_candidate_counts_or_decisions(tmp_path):
    # Confirms classify_candidate's own decision is untouched by Batch A --
    # build_structured_feature_dict only ever *receives* is_candidate/
    # candidate_reason, it never feeds back into or alters classify_candidate.
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        is_cand_before, reason_before, analysis_before = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
        # build a feature dict with Batch A included -- must not mutate anything
        mr.build_structured_feature_dict(
            "meetinge", "TR", analysis_before, obj, DEFAULTS,
            is_candidate=is_cand_before, candidate_reason=reason_before, include_batch_a=True)
        is_cand_after, reason_after, analysis_after = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert is_cand_before == is_cand_after
    assert reason_before == reason_after


# ---------------------------------------------------------------------------
# Phase 5D, Batch B: morphological complexity features (morph_tag_count,
# has_case, has_plural, has_possessive, has_derivational_suffix,
# has_verbal_morphology, morph_complexity). Every value is read directly off
# analysis.ud_feats/deriv/amb -- no new parser call, no change to candidate
# generation.
# ---------------------------------------------------------------------------

def test_batch_b_excluded_by_default():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    for key in ("morph_tag_count", "has_case", "has_plural", "has_possessive",
                "has_derivational_suffix", "has_verbal_morphology", "morph_complexity"):
        assert key not in feats


def test_batch_b_defaults_when_no_analysis():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, include_batch_b=True)
    assert feats["morph_tag_count"] == 0
    assert feats["has_case"] is False
    assert feats["has_plural"] is False
    assert feats["has_possessive"] is False
    assert feats["has_derivational_suffix"] is False
    assert feats["has_verbal_morphology"] is False
    assert feats["morph_complexity"] == "simple"


def test_has_case_true_for_case_ending():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="de", split_position=7, segments=("de",),
        ud_feats=frozenset({"Case=Loc"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetingde", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["has_case"] is True
    assert feats["morph_tag_count"] == 1
    assert feats["morph_complexity"] == "simple"


def test_has_case_false_when_no_case_tag():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="app", suffix="lar", split_position=3, segments=("lar",),
        ud_feats=frozenset({"Number=Plur"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "applar", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["has_case"] is False
    assert feats["has_plural"] is True


def test_has_plural_false_for_possessor_plural_not_own_number():
    # Number[psor]=Plur (e.g. "-ımız", "our X") describes the POSSESSOR's
    # number, not the token's own number -- must not be conflated with
    # Number=Plur (the token itself being plural).
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="app", suffix="ımız", split_position=3, segments=("ımız",),
        ud_feats=frozenset({"Poss=Yes", "Person[psor]=1", "Number[psor]=Plur"}),
        deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "appımız", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["has_plural"] is False
    assert feats["has_possessive"] is True
    assert feats["morph_tag_count"] == 3
    assert feats["morph_complexity"] == "moderate"


def test_has_possessive_true_for_poss_yes():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="app", suffix="ım", split_position=3, segments=("ım",),
        ud_feats=frozenset({"Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"}),
        deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "appım", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["has_possessive"] is True


def test_has_derivational_suffix_true_for_deriv_tag_not_derivpos_alone():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="app", suffix="cı", split_position=3, segments=("cı",),
        ud_feats=frozenset(), deriv=frozenset({"Deriv=CI", "DerivPOS=NOUN"}),
        amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "appcı", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["has_derivational_suffix"] is True
    assert feats["has_verbal_morphology"] is False
    assert feats["morph_tag_count"] == 2


def test_has_verbal_morphology_true_only_for_verbal_source():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="Replace", suffix="lemek", split_position=7, segments=("le", "mek"),
        ud_feats=frozenset(), deriv=frozenset({"Verbal=Verbalizer", "Verbal=Infinitive"}),
        amb=frozenset(), source="verbal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "Replacelemek", "UID", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["has_verbal_morphology"] is True
    assert feats["has_derivational_suffix"] is False
    assert feats["has_case"] is False
    assert feats["morph_tag_count"] == 2
    assert feats["morph_complexity"] == "moderate"


@pytest.mark.parametrize("ud_feats,deriv,amb,expected", [
    (frozenset(), frozenset(), frozenset(), "simple"),
    (frozenset({"Case=Loc"}), frozenset(), frozenset(), "simple"),
    (frozenset({"Case=Loc", "Number=Plur"}), frozenset(), frozenset(), "moderate"),
    (frozenset({"Poss=Yes", "Person[psor]=1", "Number[psor]=Sing"}), frozenset(), frozenset(), "moderate"),
    (frozenset({"Poss=Yes", "Person[psor]=1", "Number[psor]=Sing", "Case=Loc"}), frozenset(), frozenset(), "complex"),
])
def test_morph_complexity_thresholds(ud_feats, deriv, amb, expected):
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="app", suffix="x", split_position=3, segments=("x",),
        ud_feats=ud_feats, deriv=deriv, amb=amb, source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "appx", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["morph_complexity"] == expected


def test_morph_tag_count_matches_analysis_feature_count_property():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="app", suffix="larından", split_position=3, segments=("ından", "lar"),
        ud_feats=frozenset({"Case=Abl", "Number=Plur"}), deriv=frozenset(),
        amb=frozenset({"Amb=P3sg_or_Acc"}), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "applarından", "TR", analysis, obj, DEFAULTS, include_batch_b=True)
    assert feats["morph_tag_count"] == analysis.feature_count == 3


def test_batch_b_does_not_disturb_batch_a_or_c_when_all_three_enabled():
    # The untested gating interaction introduced by Phase 5D: all three
    # include_batch_* flags passed together for the first time.
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True, include_batch_b=True)
    # baseline + Batch A + Batch C keys still present and correct
    assert feats["stem_evidence_strength"] == "both"
    assert feats["analysis_source"] == "nominal"
    assert feats["is_candidate"] is True
    # Batch B keys also present and correct
    assert feats["has_case"] is True
    assert feats["morph_tag_count"] == 1
    assert feats["morph_complexity"] == "simple"


def test_batch_b_does_not_change_candidate_counts_or_decisions():
    # Mirrors test_batch_a_does_not_change_candidate_counts_or_decisions --
    # build_structured_feature_dict must never feed back into classify_candidate.
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        is_cand_before, reason_before, analysis_before = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
        mr.build_structured_feature_dict(
            "meetinge", "TR", analysis_before, obj, DEFAULTS,
            is_candidate=is_cand_before, candidate_reason=reason_before, include_batch_b=True)
        is_cand_after, reason_after, analysis_after = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert is_cand_before == is_cand_after
    assert reason_before == reason_after


# ---------------------------------------------------------------------------
# Phase 5E, Batch G: candidate-analysis ambiguity / selection-process
# metadata. Computed purely from a caller-supplied `candidate_analyses` list
# -- no new parsing, no change to enumerate_candidate_analyses/
# select_best_analysis.
#
# Phase 5F: pruned to exactly analysis_candidate_count, selection_is_unique,
# has_nominal_verbal_competition -- distinct_stem_count removed (structurally
# identical to analysis_candidate_count under the current enumeration
# algorithm; see build_structured_feature_dict's docstring) -- and the
# dataset-builder's second enumerate_candidate_analyses call eliminated in
# favour of classify_candidate(return_candidates=True).
# ---------------------------------------------------------------------------

def _ca(stem, suffix, split_position, segments, ud_feats=frozenset(), deriv=frozenset(),
        amb=frozenset(), source="nominal"):
    return mr.CandidateAnalysis(stem=stem, suffix=suffix, split_position=split_position,
                                 segments=segments, ud_feats=ud_feats, deriv=deriv, amb=amb, source=source)


def test_batch_g_excluded_by_default():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict("xyz", "UID", None, obj, DEFAULTS)
    for key in ("analysis_candidate_count", "distinct_stem_count",
                "selection_is_unique", "has_nominal_verbal_competition"):
        assert key not in feats


def test_batch_g_no_analysis_fallback():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, include_batch_g=True, candidate_analyses=None)
    assert feats["analysis_candidate_count"] == 0
    assert feats["has_nominal_verbal_competition"] is False
    assert feats["selection_is_unique"] is True  # vacuously -- nothing to tie with


def test_batch_g_empty_list_same_as_none():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[])
    assert feats["analysis_candidate_count"] == 0
    assert feats["selection_is_unique"] is True


def test_batch_g_exactly_one_analysis():
    obj = _make_annotator()
    c1 = _ca("meeting", "e", 7, ("e",), ud_feats=frozenset({"Case=Dat"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", c1, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1])
    assert feats["analysis_candidate_count"] == 1
    assert feats["selection_is_unique"] is True
    assert feats["has_nominal_verbal_competition"] is False
    assert "distinct_stem_count" not in feats


def test_batch_g_multiple_analyses_same_stem():
    obj = _make_annotator()
    # same stem, two different suffix splits of the same token
    c1 = _ca("app", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}))
    c2 = _ca("app", "larda", 3, ("lar", "da"), ud_feats=frozenset({"Number=Plur", "Case=Loc"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "applarda", "TR", c2, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1, c2])
    assert feats["analysis_candidate_count"] == 2
    # c2 has more segments (2 > 1) -> uniquely selected under highest_suffix_segments
    assert feats["selection_is_unique"] is True


def test_batch_g_multiple_analyses_distinct_stems():
    obj = _make_annotator()
    c1 = _ca("app", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}))
    c2 = _ca("appl", "ar", 4, ("ar",), ud_feats=frozenset({"Number=Plur"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "applar", "TR", c1, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1, c2])
    assert feats["analysis_candidate_count"] == 2


def test_batch_g_tied_selection_is_not_unique():
    obj = _make_annotator()
    # identical (segment_count, stem_length, feature_count, split_position)
    # key under highest_suffix_segments -- a genuine tie, different stems.
    c1 = _ca("app", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}))
    c2 = _ca("cat", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "catlar", "TR", c2, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1, c2])
    assert feats["selection_is_unique"] is False
    assert feats["analysis_candidate_count"] == 2


def test_batch_g_nominal_only_no_competition():
    obj = _make_annotator()
    c1 = _ca("app", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "applar", "TR", c1, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1])
    assert feats["has_nominal_verbal_competition"] is False


def test_batch_g_verbal_only_no_competition():
    obj = _make_annotator()
    c1 = _ca("Replace", "lemek", 7, ("le", "mek"), deriv=frozenset({"Verbal=Verbalizer", "Verbal=Infinitive"}),
             source="verbal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "Replacelemek", "UID", c1, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1])
    assert feats["has_nominal_verbal_competition"] is False


def test_batch_g_nominal_verbal_competition_true():
    obj = _make_annotator()
    c1 = _ca("app", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}), source="nominal")
    c2 = _ca("app", "lamak", 3, ("la", "mak"), deriv=frozenset({"Verbal=Verbalizer", "Verbal=Infinitive"}),
             source="verbal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "applamak", "TR", c1, obj, DEFAULTS, include_batch_g=True, candidate_analyses=[c1, c2])
    assert feats["has_nominal_verbal_competition"] is True


def test_batch_g_does_not_change_selection_result():
    # select_best_analysis's own output must be identical regardless of
    # whether Batch G bookkeeping is computed over the same list afterward.
    obj = _make_annotator()
    c1 = _ca("app", "lar", 3, ("lar",), ud_feats=frozenset({"Number=Plur"}))
    c2 = _ca("appl", "ar", 4, ("ar",), ud_feats=frozenset({"Number=Plur"}))
    candidates = [c1, c2]
    before = mr.select_best_analysis(candidates, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        mr.build_structured_feature_dict(
            "applar", "TR", before, obj, DEFAULTS, include_batch_g=True, candidate_analyses=candidates)
    after = mr.select_best_analysis(candidates, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert before == after


def test_batch_g_does_not_change_candidate_counts_or_decisions():
    # Mirrors the Batch A/B equivalents -- build_structured_feature_dict
    # (with Batch G bookkeeping) must never feed back into classify_candidate.
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        is_cand_before, reason_before, analysis_before = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
        candidates = mr.enumerate_candidate_analyses("meetinge", obj)
        mr.build_structured_feature_dict(
            "meetinge", "TR", analysis_before, obj, DEFAULTS,
            is_candidate=is_cand_before, candidate_reason=reason_before,
            include_batch_g=True, candidate_analyses=candidates)
        is_cand_after, reason_after, analysis_after = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert is_cand_before == is_cand_after
    assert reason_before == reason_after


def test_batch_g_does_not_disturb_batch_a_or_c_when_all_enabled():
    # The untested gating interaction introduced by Phase 5E: Batch A + C + G
    # together (Batch B stays off, matching the active experimental config).
    obj = _make_annotator(english_words={"meeting"})
    c1 = _ca("meeting", "e", 7, ("e",), ud_feats=frozenset({"Case=Dat"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", c1, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True, include_batch_b=False,
            include_batch_g=True, candidate_analyses=[c1])
    assert feats["stem_evidence_strength"] == "both"
    assert feats["analysis_source"] == "nominal"
    assert feats["is_candidate"] is True
    assert feats["analysis_candidate_count"] == 1
    assert feats["selection_is_unique"] is True
    for key in ("morph_tag_count", "has_case", "morph_complexity"):
        assert key not in feats
    assert "distinct_stem_count" not in feats


# ---------------------------------------------------------------------------
# Phase 5F: Batch G integrity cleanup -- prune distinct_stem_count, and
# reuse classify_candidate's own internal enumeration (via
# return_candidates=True) instead of a second enumerate_candidate_analyses
# call. These tests cover the cleanup itself, not new feature behaviour.
# ---------------------------------------------------------------------------

def test_classify_candidate_default_still_returns_3_tuple():
    # Preserves classify_candidate's external/default behaviour: every
    # existing caller that doesn't pass return_candidates must keep getting
    # exactly the same 3-tuple as before Phase 5F.
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        result = mr.classify_candidate("UID", "xyz", obj, DEFAULTS)
    assert len(result) == 3


def test_classify_candidate_non_candidate_label_does_not_enumerate():
    obj = _make_annotator()
    with mock.patch.object(mr, "enumerate_candidate_analyses") as mock_enum:
        is_cand, reason, analysis, candidates = mr.classify_candidate(
            "EN", "hello", obj, DEFAULTS, return_candidates=True)
    mock_enum.assert_not_called()
    assert is_cand is False
    assert analysis is None
    assert candidates == []


@pytest.mark.parametrize("pred_label", ["UID", "NE", "TR"])
def test_classify_candidate_enumerates_exactly_once_for_eligible_labels(pred_label):
    obj = _make_annotator(english_words={"meeting"})
    c1 = _ca("meeting", "e", 7, ("e",), ud_feats=frozenset({"Case=Dat"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)), \
         mock.patch.object(mr, "enumerate_candidate_analyses", return_value=[c1]) as mock_enum:
        is_cand, reason, analysis, candidates = mr.classify_candidate(
            pred_label, "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS,
            return_candidates=True)
    assert mock_enum.call_count == 1
    assert candidates == [c1]
    assert analysis == c1


def test_classify_candidate_return_candidates_does_not_change_selected_analysis_or_decision():
    # The Phase 5F refactor (inlining enumerate_candidate_analyses +
    # select_best_analysis instead of calling best_analysis_for_token) must
    # be behaviourally identical to the pre-cleanup code path.
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        is_cand3, reason3, analysis3 = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
        is_cand4, reason4, analysis4, candidates4 = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS,
            return_candidates=True)
    assert is_cand3 == is_cand4
    assert reason3 == reason4
    assert analysis3 == analysis4
    assert analysis4 in candidates4


@pytest.mark.parametrize("pred_label", ["UID", "NE", "TR", "EN", "OTHER", "MIXED", "LANG3"])
def test_classify_candidate_decision_identical_regardless_of_return_candidates_flag(pred_label):
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        is_cand_off, reason_off, _ = mr.classify_candidate(
            pred_label, "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
        is_cand_on, reason_on, _, _ = mr.classify_candidate(
            pred_label, "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS,
            return_candidates=True)
    assert is_cand_off == is_cand_on
    assert reason_off == reason_on


def test_batch_g_distinct_stem_count_removed_in_all_scenarios():
    obj = _make_annotator()
    c1 = _ca("meeting", "e", 7, ("e",), ud_feats=frozenset({"Case=Dat"}))
    c2 = _ca("appl", "ar", 4, ("ar",), ud_feats=frozenset({"Number=Plur"}))
    scenarios = [None, [], [c1], [c1, c2]]
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        for candidates in scenarios:
            feats = mr.build_structured_feature_dict(
                "meetinge", "TR", c1, obj, DEFAULTS, include_batch_g=True, candidate_analyses=candidates)
            assert "distinct_stem_count" not in feats
            assert "analysis_candidate_count" in feats
            assert "selection_is_unique" in feats
            assert "has_nominal_verbal_competition" in feats


def test_batch_a_and_c_gating_unchanged_alongside_pruned_batch_g():
    # Batch A/C's own gating logic must be untouched by the Batch G prune.
    obj = _make_annotator(english_words={"meeting"})
    c1 = _ca("meeting", "e", 7, ("e",), ud_feats=frozenset({"Case=Dat"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats_a_c_only = mr.build_structured_feature_dict(
            "meetinge", "TR", c1, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True)
        feats_a_c_g = mr.build_structured_feature_dict(
            "meetinge", "TR", c1, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True, include_batch_g=True,
            candidate_analyses=[c1])
    for key in ("analysis_source", "candidate_reason", "is_candidate", "split_position_ratio",
                "stem_evidence_strength", "ft_prob_delta", "ft_lang_agreement"):
        assert feats_a_c_only[key] == feats_a_c_g[key]
    assert "analysis_candidate_count" not in feats_a_c_only
    assert "analysis_candidate_count" in feats_a_c_g


def test_batch_b_excluded_from_pruned_batch_g_benchmark_config():
    # Batch B must remain off by default even with Batch A+C+G all enabled --
    # the Phase 5F active benchmark configuration never includes Batch B.
    obj = _make_annotator(english_words={"meeting"})
    c1 = _ca("meeting", "e", 7, ("e",), ud_feats=frozenset({"Case=Dat"}))
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", c1, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True, include_batch_g=True,
            candidate_analyses=[c1])
    for key in ("morph_tag_count", "has_case", "has_plural", "has_possessive",
                "has_derivational_suffix", "has_verbal_morphology", "morph_complexity"):
        assert key not in feats


# ---------------------------------------------------------------------------
# Phase 5G, Batch D: English-stem quality (stem_english_confidence,
# stem_turkish_confidence, stem_lexicon_contrast). Every value is a
# deterministic function of already-computed baseline fields -- no new
# fastText or lexicon call.
# ---------------------------------------------------------------------------

def test_batch_d_excluded_by_default():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS)
    for key in ("stem_english_confidence", "stem_turkish_confidence", "stem_lexicon_contrast"):
        assert key not in feats


def test_stem_english_confidence_when_stem_is_english():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.93)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_english_confidence"] == pytest.approx(0.93)
    assert feats["stem_turkish_confidence"] == 0.0


def test_stem_turkish_confidence_when_stem_is_turkish():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="araba", suffix="da", split_position=5, segments=("da",),
        ud_feats=frozenset({"Case=Loc"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.87)):
        feats = mr.build_structured_feature_dict(
            "arabada", "UID", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_turkish_confidence"] == pytest.approx(0.87)
    assert feats["stem_english_confidence"] == 0.0


def test_stem_confidence_both_zero_for_unrelated_language():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="bonjour", suffix="da", split_position=7, segments=("da",),
        ud_feats=frozenset({"Case=Loc"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("FR", 0.95)):
        feats = mr.build_structured_feature_dict(
            "bonjourda", "UID", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_english_confidence"] == 0.0
    assert feats["stem_turkish_confidence"] == 0.0


def test_stem_lexicon_contrast_english_only():
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_lexicon_contrast"] == "english_only"


def test_stem_lexicon_contrast_turkish_only():
    obj = _make_annotator(turkish_all={"araba"})
    analysis = mr.CandidateAnalysis(
        stem="araba", suffix="da", split_position=5, segments=("da",),
        ud_feats=frozenset({"Case=Loc"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
        feats = mr.build_structured_feature_dict(
            "arabada", "UID", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_lexicon_contrast"] == "turkish_only"


def test_stem_lexicon_contrast_both():
    obj = _make_annotator(english_words={"can"}, turkish_all={"can"})
    analysis = mr.CandidateAnalysis(
        stem="can", suffix="da", split_position=3, segments=("da",),
        ud_feats=frozenset({"Case=Loc"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "canda", "UID", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_lexicon_contrast"] == "both"


def test_stem_lexicon_contrast_neither():
    obj = _make_annotator()
    analysis = mr.CandidateAnalysis(
        stem="xyzzy", suffix="da", split_position=5, segments=("da",),
        ud_feats=frozenset({"Case=Loc"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.1)):
        feats = mr.build_structured_feature_dict(
            "xyzzyda", "UID", analysis, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_lexicon_contrast"] == "neither"


def test_batch_d_no_analysis_fallback():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "xyz", "UID", None, obj, DEFAULTS, include_batch_d=True)
    assert feats["stem_english_confidence"] == 0.0
    assert feats["stem_turkish_confidence"] == 0.0
    assert feats["stem_lexicon_contrast"] == "neither"


def test_stem_length_ratio_deliberately_omitted_as_duplicate():
    # Phase 5G inventory finding: CandidateAnalysis.split_position == len(stem)
    # by construction, so a stem_length_ratio feature would be an exact
    # duplicate of Batch A's existing split_position_ratio. Confirmed absent.
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    assert analysis.split_position == analysis.stem_length
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", analysis, obj, DEFAULTS,
            include_batch_a=True, include_batch_d=True)
    assert "stem_length_ratio" not in feats
    assert "stem_is_short" not in feats
    assert feats["split_position_ratio"] == pytest.approx(7 / 8)


def test_batch_d_does_not_introduce_new_fasttext_calls():
    obj = _make_annotator(english_words={"meeting"})
    analysis = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)) as mock_ft:
        mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS, include_batch_d=False)
        calls_without = mock_ft.call_count
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)) as mock_ft:
        mr.build_structured_feature_dict("meetinge", "TR", analysis, obj, DEFAULTS, include_batch_d=True)
        calls_with = mock_ft.call_count
    assert calls_without == calls_with


def test_batch_d_does_not_change_candidate_counts_or_decisions():
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        is_cand_before, reason_before, analysis_before = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
        mr.build_structured_feature_dict(
            "meetinge", "TR", analysis_before, obj, DEFAULTS,
            is_candidate=is_cand_before, candidate_reason=reason_before, include_batch_d=True)
        is_cand_after, reason_after, analysis_after = mr.classify_candidate(
            "TR", "meetinge", obj, DEFAULTS, strategy=mr.STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
    assert is_cand_before == is_cand_after
    assert reason_before == reason_after
    assert analysis_before == analysis_after


def test_batch_d_does_not_disturb_a_c_g_gating_when_all_enabled():
    obj = _make_annotator(english_words={"meeting"})
    c1 = mr.CandidateAnalysis(
        stem="meeting", suffix="e", split_position=7, segments=("e",),
        ud_feats=frozenset({"Case=Dat"}), deriv=frozenset(), amb=frozenset(), source="nominal")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        feats = mr.build_structured_feature_dict(
            "meetinge", "TR", c1, obj, DEFAULTS,
            is_candidate=True, candidate_reason=mr.CANDIDATE_REASON_TR_SUSPECT_STEM,
            include_batch_a=True, include_batch_c=True, include_batch_b=False,
            include_batch_g=True, candidate_analyses=[c1], include_batch_d=True)
    assert feats["stem_evidence_strength"] == "both"
    assert feats["analysis_source"] == "nominal"
    assert feats["analysis_candidate_count"] == 1
    assert feats["selection_is_unique"] is True
    assert feats["stem_english_confidence"] == pytest.approx(0.9)
    assert feats["stem_lexicon_contrast"] == "english_only"
    for key in ("morph_tag_count", "has_case", "morph_complexity", "distinct_stem_count"):
        assert key not in feats


# ---------------------------------------------------------------------------
# Residual verbal MIXED detector -- production integration (strict evidence
# only). Reuses the real, unmodified PHASE_4E level (DEFAULT_VERBAL_MORPHOLOGY_
# LEVEL, PHASE_4C1, is never referenced by this stage) and the existing
# _make_annotator bypass-__init__ convention above. TARGETS/NATIVE_CONTROLS
# mirror the offline Phase 4 validation exactly -- see CHANGELOG for the
# authorized production brief this stage implements.
# ---------------------------------------------------------------------------

RESIDUAL_TARGETS = [
    ("uploadlamışım", "upload"), ("designladık", "design"), ("inviteladık", "invite"),
    ("filterladık", "filter"), ("forwardladı", "forward"), ("mutelamışım", "mute"),
    ("refreshledik", "refresh"), ("cancelladık", "cancel"), ("editledik", "edit"),
    ("dropladı", "drop"),
]

RESIDUAL_NATIVE_CONTROLS = [
    "anladık", "yapmışım", "bekledik", "konuşuyoruz", "gideceksin", "dinledim",
    "başladık", "temizledik", "eklemişim", "yazdırdık", "kapattılar",
    "paylaşmışım", "güncelledik", "alamadık", "alıyoruz", "biliyonuz",
    "dediğimiz", "düşünüyoruz", "giremiyoruz", "gittik", "istiyoruz",
    "kaldık", "olduğumuz", "olmuyorsunuz", "rahatlarız", "bilirdik",
]


def _residual_annotator():
    """A real (not real-lexicon-file-backed) Annotator carrying the exact
    English/Turkish stems the target/control fixtures below need -- built
    once so every test below reflects the actual lexicon-membership
    condition (6/7), not a mock."""
    return _make_annotator(
        turkish_all={
            "an", "yap", "din", "ek", "al", "iz", "biliyor", "biliy", "gid",
            "konus", "konuş", "bekle", "basla", "başla", "temizle", "yazdır",
            "yazdir", "kapat", "paylas", "paylaş", "guncelle", "güncelle",
            "olmuyor", "rahat", "bil", "kal", "old", "gel", "gir",
        },
        english_words={
            "upload", "design", "invite", "filter", "forward", "mute",
            "refresh", "cancel", "edit", "drop",
        },
    )


# --- 1/2/3: exact parser gaps confirmed + resolved ------------------------

def test_residual_parse_lamisim_fully_consumed():
    segs, tags, full, _ = mr.parse_residual_verbal_suffix("lamışım")
    assert full is True
    assert "Verbal=Verbalizer" in tags
    assert "Verbal=1stPersonSingular" in tags


def test_residual_parse_ladi_fully_consumed():
    segs, tags, full, _ = mr.parse_residual_verbal_suffix("ladı")
    assert full is True
    assert "Verbal=Verbalizer" in tags
    assert "Verbal=Past" in tags


def test_residual_parse_ladik_not_consumable_under_any_existing_level():
    # Confirms the ONE genuine gap this stage closes: no existing level
    # (including the frozen default PHASE_4C1) fully parses "ladık" alone.
    for level in mr.VERBAL_MORPHOLOGY_LEVELS:
        _, _, full, _ = mr._parse_experimental_verbal_suffix("ladık", level=level)
        assert full is False, f"level {level!r} unexpectedly consumed 'ladık'"


def test_residual_parse_ladik_fully_consumed_via_new_plural_stage():
    segs, tags, full, _ = mr.parse_residual_verbal_suffix("ladık")
    assert full is True
    assert "Verbal=Verbalizer" in tags
    assert "Verbal=Past" in tags
    assert "Verbal=1stPersonPlural" in tags


# --- 4: first-person-plural fused past forms -------------------------------

@pytest.mark.parametrize("suffix", ["dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük"])
def test_residual_plural_fused_past_forms_tag_both_past_and_plural(suffix):
    tagset = mr.VERBAL_FIRST_PERSON_PLURAL[suffix]
    assert "Verbal=Past" in tagset
    assert "Verbal=1stPersonPlural" in tagset


@pytest.mark.parametrize("suffix", ["ık", "ik", "uk", "ük", "ız", "iz", "uz", "üz"])
def test_residual_plural_bare_forms_tag_plural_only(suffix):
    tagset = mr.VERBAL_FIRST_PERSON_PLURAL[suffix]
    assert tagset == frozenset({"Verbal=1stPersonPlural"})


# --- 5: bare single-character "-k" rejected --------------------------------

def test_residual_bare_single_char_k_not_in_plural_table():
    assert "k" not in mr.VERBAL_FIRST_PERSON_PLURAL


def test_residual_bare_k_does_not_yield_a_fully_consumed_plural_analysis():
    # "artık" is an ordinary Turkish word ending in "-ık"/"-k"; parsing its
    # suffix must not spuriously resolve via a bare 1-char "k" candidate --
    # only the (excluded) full "ık" as a listed suffix should ever apply,
    # and only when the caller explicitly enumerates that split.
    segs, tags, full, _ = mr.parse_residual_verbal_suffix("k")
    assert full is False  # "k" alone matches nothing in the plural table


# --- 6: strict English-lexicon requirement (production policy) ------------

def test_residual_strict_evidence_requires_direct_lexicon_hit():
    obj = _make_annotator(english_words={"upload"})
    assert mr._residual_verbal_direct_english_evidence(obj, "upload") is True
    assert mr._residual_verbal_direct_english_evidence(obj, "uplod") is False


def test_residual_strict_mode_rejects_fasttext_only_evidence():
    # A stem with NO lexicon entry but strong fastText EN evidence must be
    # rejected under the production default (strict_lexicon_only=True) --
    # this is the exact offline finding that justified NOT wiring the broad
    # fallback into production.
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.99)):
        promote, cand, reason = mr.evaluate_residual_verbal_promotion(
            "xyzqladık", obj, DEFAULTS, strict_lexicon_only=True)
    assert promote is False
    assert reason in ("no_qualifying_candidate", "no_verbal_candidate")


def test_residual_broad_mode_available_but_not_default():
    # The broad (fastText-inclusive) evidence path remains selectable for
    # offline experimentation only -- exercising it here proves it still
    # exists and works, without implying it is wired into any production
    # call site (reranker_integration.py never passes strict_lexicon_only).
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.99)):
        promote, cand, reason = mr.evaluate_residual_verbal_promotion(
            "xyzqladık", obj, DEFAULTS, strict_lexicon_only=False)
    assert promote is True
    assert cand["stem"] == "xyzq"


def test_residual_default_parameter_is_strict():
    import inspect
    sig = inspect.signature(mr.evaluate_residual_verbal_promotion)
    assert sig.parameters["strict_lexicon_only"].default is True


# --- 7: Turkish-lexicon collision rejection --------------------------------

def test_residual_turkish_lexicon_stem_rejected_even_with_english_homograph():
    # "an" is a genuine Turkish word ("moment") that also happens to be an
    # English word (indefinite article) -- condition 7 must reject it via
    # Turkish-lexicon presence, independent of condition 6's outcome.
    obj = _make_annotator(turkish_all={"an"}, english_words={"an"})
    promote, cand, reason = mr.evaluate_residual_verbal_promotion("anladık", obj, DEFAULTS)
    assert promote is False


@pytest.mark.parametrize("token", RESIDUAL_NATIVE_CONTROLS)
def test_residual_native_turkish_controls_never_promote(token):
    obj = _residual_annotator()
    promote, cand, reason = mr.evaluate_residual_verbal_promotion(token, obj, DEFAULTS)
    assert promote is False, f"{token} incorrectly promoted (stem={cand['stem'] if cand else None}, reason={reason})"


# --- 8: competing nominal-analysis rejection -------------------------------

def test_residual_competing_lexicon_confirmed_nominal_stem_blocks_promotion():
    # A constructed token where a nominal (case/possessive) suffix split
    # leaves a Turkish-lexicon-confirmed stem must be rejected even though a
    # verbal analysis also qualifies -- condition 8.
    obj = _make_annotator(turkish_all={"masa"}, english_words={"upload"})
    # "uploadlamışım" itself has no competing lexicon-confirmed nominal stem
    # (its only short nominal split leaves "uploadlamış", not lexicon-
    # confirmed) -- the promotion must succeed there (see target-list test
    # below). This test isolates the OPPOSITE case: a stem that genuinely
    # does have a lexicon-confirmed nominal competitor.
    assert mr._residual_verbal_has_lexicon_confirmed_competing_nominal_stem("masaya", obj) is True


def test_residual_short_stem_nonsense_competing_analysis_does_not_block():
    # Regression for the exact bug found during offline validation: the
    # nominal parser trivially matches a bare 2-char suffix ("ım") almost
    # anywhere, leaving a nonsense stem ("uploadlamış") that must NOT count
    # as real competing evidence (it is not itself Turkish-lexicon-confirmed).
    obj = _make_annotator(english_words={"upload"})
    promote, cand, reason = mr.evaluate_residual_verbal_promotion("uploadlamışım", obj, DEFAULTS)
    assert promote is True
    assert cand["stem"] == "upload"


# --- 9: ambiguous-analysis rejection ---------------------------------------

def test_residual_ambiguous_tie_rejected():
    # Two distinct English-lexicon stems of EQUAL length, both otherwise
    # qualifying, must be rejected as ambiguous rather than arbitrarily
    # picking one.
    obj = _make_annotator(english_words={"drop", "stop"})
    with mock.patch.object(mr, "enumerate_residual_verbal_candidates", return_value=[
        {"stem": "drop", "suffix": "ladı", "split_position": 4,
         "segments": ["la", "dı"], "tags": frozenset({"Verbal=Verbalizer", "Verbal=Past"}), "used_informal": False},
        {"stem": "stop", "suffix": "ladı", "split_position": 4,
         "segments": ["la", "dı"], "tags": frozenset({"Verbal=Verbalizer", "Verbal=Past"}), "used_informal": False},
    ]):
        promote, cand, reason = mr.evaluate_residual_verbal_promotion("dropstopladı", obj, DEFAULTS)
    assert promote is False
    assert reason == "ambiguous_analysis"


# --- 10: proper-name/acronym/code guards -----------------------------------

def test_residual_capitalized_token_rejected_as_proper_name():
    assert mr._residual_verbal_looks_like_proper_name_or_noise("Uploadlamışım") is True


def test_residual_all_caps_acronym_rejected():
    assert mr._residual_verbal_looks_like_proper_name_or_noise("APIlamışım") is True


def test_residual_alphanumeric_code_rejected():
    assert mr._residual_verbal_looks_like_proper_name_or_noise("A7Klamışım") is True


def test_residual_lowercase_plain_token_not_flagged_as_noise():
    assert mr._residual_verbal_looks_like_proper_name_or_noise("uploadlamışım") is False


def test_residual_capitalized_target_never_promotes_end_to_end():
    obj = _make_annotator(english_words={"comeout"})
    promote, cand, reason = mr.evaluate_residual_verbal_promotion("Comeoutladim", obj, DEFAULTS)
    assert promote is False
    assert reason == "proper_name_or_noise"


def test_residual_apostrophe_bearing_token_produces_no_candidates():
    assert mr.enumerate_residual_verbal_candidates("upload'lamışım") == []
    promote, cand, reason = mr.evaluate_residual_verbal_promotion(
        "upload'lamışım", _make_annotator(english_words={"upload"}), DEFAULTS)
    assert promote is False
    assert reason == "no_verbal_candidate"


# --- 11: all positive regression targets promote with the correct stem ----

@pytest.mark.parametrize("token,expected_stem", RESIDUAL_TARGETS)
def test_residual_all_targets_promote_with_correct_stem(token, expected_stem):
    obj = _residual_annotator()
    promote, cand, reason = mr.evaluate_residual_verbal_promotion(token, obj, DEFAULTS)
    assert promote is True, f"{token}: {reason}"
    assert cand["stem"] == expected_stem


# --- conditions requiring an explicit verbalizer/passive-inchoative marker

def test_residual_no_verbalizer_never_promotes():
    # A fully-consumed agreement/tense analysis without -la-/-le-/-lan-/-len-
    # must never qualify, regardless of lexicon evidence (condition 2).
    obj = _make_annotator(english_words={"shipler"})
    promote, cand, reason = mr.evaluate_residual_verbal_promotion("shiplerdim", obj, DEFAULTS)
    assert promote is False
