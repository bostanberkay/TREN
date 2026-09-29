"""MIXED-token reranking: candidate generation and features, frozen-model integration (apply_reranker), and the UID->TR resolver."""

import json
import os
import re
import sys
from dataclasses import dataclass, field, replace as _dataclass_replace
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

import scipy.sparse as sp

import annotation_model


# ---------------------------------------------------------------------------
# MIXED-candidate generation and feature engineering
#
# Candidates are right-to-left splits whose tail parses as a suffix chain
# (nominal parser first, then the experimental verbal tables). The frozen model
# uses feature Batches A + C + pruned G; B and D are off (see README).
# ---------------------------------------------------------------------------


MIN_STEM_LEN = 2

TURKISH_CHARS = set("çÇğĞıİöÖşŞüÜ")

SCHEMA_LABELS = ("TR", "EN", "MIXED", "UID", "NE", "LANG3", "OTHER")

# Derived only from the predicted annotation, never the gold label.
CANDIDATE_REASON_UID = "UID"
CANDIDATE_REASON_NE_SUFFIX = "NE_SUFFIX"
CANDIDATE_REASON_TR_SUSPECT_STEM = "TR_SUSPECT_STEM"

NON_CANDIDATE_LABELS = frozenset({"EN", "OTHER", "MIXED", "LANG3"})


@dataclass(frozen=True)
class CandidateAnalysis:
    """One plausible stem/suffix split of a token; split_position == len(stem).

    `source` is "nominal" if the production nominal parser fully consumed the
    suffix, "verbal" if only the verbal fallback did.
    `informal_suffix_normalization` is True only if the analysis was reached
    via the informal-orthography fallback.
    """
    stem: str
    suffix: str
    split_position: int
    segments: Tuple[str, ...]
    ud_feats: FrozenSet[str]
    deriv: FrozenSet[str]
    amb: FrozenSet[str]
    source: str = "nominal"
    informal_suffix_normalization: bool = False
    # Set only when TR-bucket candidacy came from duplicated-consonant recovery;
    # `stem` is never modified, the recovered form is `recovered_english_stem`.
    stem_orthographic_recovery: bool = False
    recovered_english_stem: Optional[str] = None

    @property
    def stem_length(self) -> int:
        return len(self.stem)

    @property
    def suffix_length(self) -> int:
        return len(self.suffix)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def feature_count(self) -> int:
        """UD + derivational + ambiguity tag count; a selection tie-break."""
        return len(self.ud_feats) + len(self.deriv) + len(self.amb)


# Experimental verbal-suffix tables; nominal parsing in cs_pipeline is untouched.
# Stage sets per VERBAL_MORPHOLOGY_* level are listed in
# _parse_experimental_verbal_suffix.
VERBAL_INFINITIVE = {"mak": "Verbal=Infinitive", "mek": "Verbal=Infinitive"}
VERBAL_VERBALIZER = {"la": "Verbal=Verbalizer", "le": "Verbal=Verbalizer"}
VERBAL_PASSIVE_INCHOATIVE = {"lan": "Verbal=PassiveInchoative", "len": "Verbal=PassiveInchoative"}

VERBAL_PAST = {
    "dı": "Verbal=Past", "di": "Verbal=Past", "du": "Verbal=Past", "dü": "Verbal=Past",
    "tı": "Verbal=Past", "ti": "Verbal=Past", "tu": "Verbal=Past", "tü": "Verbal=Past",
}
VERBAL_EVIDENTIAL = {
    "mış": "Verbal=Evidential", "miş": "Verbal=Evidential", "muş": "Verbal=Evidential", "müş": "Verbal=Evidential",
}
VERBAL_PROGRESSIVE = {"yor": "Verbal=Progressive"}
VERBAL_FUTURE = {"acak": "Verbal=Future", "ecek": "Verbal=Future"}
# Mutually exclusive on a finite verb, so tried together, longest key first.
VERBAL_TENSE_ASPECT_MOOD = {**VERBAL_PAST, **VERBAL_EVIDENTIAL, **VERBAL_PROGRESSIVE, **VERBAL_FUTURE}

VERBAL_SECOND_PERSON = {
    "sınız": "Verbal=2ndPersonPlural", "siniz": "Verbal=2ndPersonPlural",
    "sunuz": "Verbal=2ndPersonPlural", "sünüz": "Verbal=2ndPersonPlural",
    "sın": "Verbal=2ndPersonSingular", "sin": "Verbal=2ndPersonSingular",
    "sun": "Verbal=2ndPersonSingular", "sün": "Verbal=2ndPersonSingular",
}

# Fused past+1sg forms carry both Past and 1stPersonSingular although matched as one string.
VERBAL_FIRST_PERSON_SINGULAR = {
    "m": frozenset({"Verbal=1stPersonSingular"}),
    "ım": frozenset({"Verbal=1stPersonSingular"}), "im": frozenset({"Verbal=1stPersonSingular"}),
    "um": frozenset({"Verbal=1stPersonSingular"}), "üm": frozenset({"Verbal=1stPersonSingular"}),
    "dım": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "dim": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "dum": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "düm": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "tım": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "tim": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "tum": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
    "tüm": frozenset({"Verbal=Past", "Verbal=1stPersonSingular"}),
}

# Union of VERBAL_SECOND_PERSON (values wrapped as singleton frozensets) and
# VERBAL_FIRST_PERSON_SINGULAR, so one lookup serves both.
VERBAL_AGREEMENT_COMBINED_4E = {
    **{k: frozenset({v}) for k, v in VERBAL_SECOND_PERSON.items()},
    **VERBAL_FIRST_PERSON_SINGULAR,
}

VERBAL_MORPHOLOGY_PHASE_4A = "phase4a"
VERBAL_MORPHOLOGY_PHASE_4B = "phase4b"
VERBAL_MORPHOLOGY_PHASE_4C1 = "phase4c1"
# 4A (always active) + 4B + 4C-1 combined; the other levels are unchanged.
VERBAL_MORPHOLOGY_PHASE_4E = "phase4e"
VERBAL_MORPHOLOGY_LEVELS = (VERBAL_MORPHOLOGY_PHASE_4A, VERBAL_MORPHOLOGY_PHASE_4B,
                            VERBAL_MORPHOLOGY_PHASE_4C1, VERBAL_MORPHOLOGY_PHASE_4E)
# Matched or beat PHASE_4A on the precision>=0.75 threshold policy.
DEFAULT_VERBAL_MORPHOLOGY_LEVEL = VERBAL_MORPHOLOGY_PHASE_4C1

# Informal-orthography fallback (s->ş, i->ı) for verbal-suffix matching only:
# respells the local suffix substring, never the token or stem, and only after
# literal matching failed. Off unless allow_informal_orthography=True.
INFORMAL_ORTHOGRAPHY_MAP = {"s": "ş", "i": "ı"}


def _informal_suffix_variants(s: str) -> List[str]:
    """Respellings of `s` that differ from it; the caller always tries the
    literal `s` first."""
    variants = []
    seen = {s}
    for candidate in (
        s.replace("s", "ş"),
        s.replace("i", "ı"),
        s.replace("s", "ş").replace("i", "ı"),
    ):
        if candidate not in seen:
            variants.append(candidate)
            seen.add(candidate)
    return variants


def _match_verbal_table(s: str, table: dict, allow_informal_orthography: bool):
    """Match the end of `s` against `table`, longest key first. Returns
    (matched_end, tag_or_tagset, used_informal) or None. The caller strips
    len(matched_end) from the ORIGINAL `s`; safe because every
    INFORMAL_ORTHOGRAPHY_MAP entry is a same-length substitution.
    """
    items = sorted(table.items(), key=lambda kv: -len(kv[0]))
    for end, tag in items:
        if s.endswith(end):
            return end, tag, False
    if allow_informal_orthography:
        for variant in _informal_suffix_variants(s):
            for end, tag in items:
                if variant.endswith(end):
                    return end, tag, True
    return None


def _parse_experimental_verbal_suffix(suffix: str, level: str = DEFAULT_VERBAL_MORPHOLOGY_LEVEL,
                                       allow_informal_orthography: bool = False
                                       ) -> Tuple[List[str], FrozenSet[str], bool, bool]:
    """Experimental verbal-suffix parser (candidate generation only). Peels
    right to left: agreement (1/1b), tense/aspect/mood (2), infinitive (3, only
    if no finite stage matched), passive/inchoative else verbalizer (4).

    Levels: 4A = stages 3-4; 4B adds 1 (2nd person) and 2; 4C1 adds only 1b
    (1st person singular); 4E = 4B + 4C1. Returns (segments, tags,
    fully_consumed, used_informal_orthography); partial matches are rejected
    by the caller.
    """
    if level not in VERBAL_MORPHOLOGY_LEVELS:
        raise ValueError(f"unknown verbal morphology level: {level!r} (valid: {VERBAL_MORPHOLOGY_LEVELS})")

    s = suffix.lower()
    segments_rev: List[str] = []
    tags = set()
    used_informal = False
    matched_finite = False  # True if ANY finite (agreement/tense/1st-person) stage matched

    if level == VERBAL_MORPHOLOGY_PHASE_4B:
        m = _match_verbal_table(s, VERBAL_SECOND_PERSON, allow_informal_orthography)
        if m is not None:
            end, tag, informal = m
            segments_rev.append(end)
            tags.add(tag)
            s = s[:-len(end)]
            matched_finite = True
            used_informal = used_informal or informal

        m = _match_verbal_table(s, VERBAL_TENSE_ASPECT_MOOD, allow_informal_orthography)
        if m is not None:
            end, tag, informal = m
            segments_rev.append(end)
            tags.add(tag)
            s = s[:-len(end)]
            matched_finite = True
            used_informal = used_informal or informal

    elif level == VERBAL_MORPHOLOGY_PHASE_4C1:
        m = _match_verbal_table(s, VERBAL_FIRST_PERSON_SINGULAR, allow_informal_orthography)
        if m is not None:
            end, tagset, informal = m
            segments_rev.append(end)
            tags.update(tagset)
            s = s[:-len(end)]
            matched_finite = True
            used_informal = used_informal or informal

    elif level == VERBAL_MORPHOLOGY_PHASE_4E:
        # Combined agreement stage: 2nd person + 1st person singular, longest key first.
        m = _match_verbal_table(s, VERBAL_AGREEMENT_COMBINED_4E, allow_informal_orthography)
        if m is not None:
            end, tagset, informal = m
            segments_rev.append(end)
            tags.update(tagset)
            s = s[:-len(end)]
            matched_finite = True
            used_informal = used_informal or informal

        m = _match_verbal_table(s, VERBAL_TENSE_ASPECT_MOOD, allow_informal_orthography)
        if m is not None:
            end, tag, informal = m
            segments_rev.append(end)
            tags.add(tag)
            s = s[:-len(end)]
            matched_finite = True
            used_informal = used_informal or informal

    if not matched_finite:
        m = _match_verbal_table(s, VERBAL_INFINITIVE, allow_informal_orthography)
        if m is not None:
            end, tag, informal = m
            segments_rev.append(end)
            tags.add(tag)
            s = s[:-len(end)]
            used_informal = used_informal or informal

    m = _match_verbal_table(s, VERBAL_PASSIVE_INCHOATIVE, allow_informal_orthography)
    if m is None:
        m = _match_verbal_table(s, VERBAL_VERBALIZER, allow_informal_orthography)
    if m is not None:
        end, tag, informal = m
        segments_rev.append(end)
        tags.add(tag)
        s = s[:-len(end)]
        used_informal = used_informal or informal

    fully_consumed = (s == "")
    return list(reversed(segments_rev)), frozenset(tags), fully_consumed, used_informal


def _analysis_from_suffix(stem: str, suffix: str, split_position: int, annotator,
                           verbal_level: str = DEFAULT_VERBAL_MORPHOLOGY_LEVEL,
                           allow_informal_orthography: bool = False) -> Optional[CandidateAnalysis]:
    """CandidateAnalysis for one (stem, suffix) pair if the suffix passes the
    plausibility checks, else None. The production nominal parser is tried
    first, then the verbal parser at `verbal_level`.
    """
    if not suffix:
        return None
    segments, ud_feats, deriv, amb = annotator._parse_tr_suffixes_full(suffix)
    if "Unparsed=Leftover" not in deriv and (ud_feats or deriv or amb):
        return CandidateAnalysis(
            stem=stem, suffix=suffix, split_position=split_position,
            segments=tuple(segments), ud_feats=frozenset(ud_feats),
            deriv=frozenset(deriv), amb=frozenset(amb), source="nominal",
        )

    v_segments, v_tags, fully_consumed, used_informal = _parse_experimental_verbal_suffix(
        suffix, level=verbal_level, allow_informal_orthography=allow_informal_orthography)
    if fully_consumed and v_tags:
        return CandidateAnalysis(
            stem=stem, suffix=suffix, split_position=split_position,
            segments=tuple(v_segments), ud_feats=frozenset(), deriv=frozenset(v_tags),
            amb=frozenset(), source="verbal", informal_suffix_normalization=used_informal,
        )
    return None


def enumerate_candidate_analyses(token: str, annotator, min_stem_len: int = MIN_STEM_LEN,
                                  verbal_level: str = DEFAULT_VERBAL_MORPHOLOGY_LEVEL,
                                  allow_informal_orthography: bool = False) -> List[CandidateAnalysis]:
    """Plausible stem/suffix splits of `token`.

    Apostrophe tokens use only Annotator._split_mixed_apostrophe (per-position
    splits kept the apostrophe in every stem and broke lexicon lookups); if it
    declines there are deliberately no candidates.
    """
    candidates: List[CandidateAnalysis] = []
    if not token:
        return candidates

    if "'" in token or "’" in token:
        base, suffix = annotator._split_mixed_apostrophe(token)
        if base and suffix and len(base) >= min_stem_len:
            analysis = _analysis_from_suffix(base, suffix, len(base), annotator,
                                              verbal_level, allow_informal_orthography)
            if analysis is not None:
                candidates.append(analysis)
        return candidates

    n = len(token)
    for split_position in range(min_stem_len, n):
        stem = token[:split_position]
        suffix = token[split_position:]
        analysis = _analysis_from_suffix(stem, suffix, split_position, annotator,
                                          verbal_level, allow_informal_orthography)
        if analysis is not None:
            candidates.append(analysis)
    return candidates


STRATEGY_LONGEST_STEM = "longest_stem"
STRATEGY_LONGEST_SUFFIX = "longest_suffix"
STRATEGY_HIGHEST_SUFFIX_SEGMENTS = "highest_suffix_segments"
CANDIDATE_STRATEGIES = (STRATEGY_LONGEST_STEM, STRATEGY_LONGEST_SUFFIX, STRATEGY_HIGHEST_SUFFIX_SEGMENTS)
# Chosen over STRATEGY_LONGEST_STEM because it matched or beat both other
# strategies on every dev/test metric (artifacts/mixed_reranker/strategy_comparison/).
DEFAULT_CANDIDATE_STRATEGY = STRATEGY_HIGHEST_SUFFIX_SEGMENTS

_STRATEGY_KEYS = {
    # Each key function ranks candidates from a `max()` call, so higher
    # tuple values win; -split_position turns "earliest split" into a max.
    STRATEGY_LONGEST_STEM: lambda c: (c.stem_length, c.segment_count, c.feature_count, -c.split_position),
    STRATEGY_LONGEST_SUFFIX: lambda c: (c.suffix_length, c.segment_count, c.feature_count, -c.split_position),
    STRATEGY_HIGHEST_SUFFIX_SEGMENTS: lambda c: (c.segment_count, c.stem_length, c.feature_count, -c.split_position),
}


def select_best_analysis(candidates: List[CandidateAnalysis], strategy: str = DEFAULT_CANDIDATE_STRATEGY) -> Optional[CandidateAnalysis]:
    """Deterministic choice among plausible analyses. Sort keys, in priority order:

      'longest_stem':            stem length, suffix segments, feature count, earliest split
      'longest_suffix':          suffix length, suffix segments, feature count, earliest split
      'highest_suffix_segments': suffix segments, stem length, feature count, earliest split

    'longest_stem' differs from the production MIXED detector, which walks
    suffixes longest-first (shortest stem first); 'longest_suffix' mirrors it.
    Raises ValueError for an unknown strategy.
    """
    if not candidates:
        return None
    if strategy not in _STRATEGY_KEYS:
        raise ValueError(f"unknown candidate-selection strategy: {strategy!r} (valid: {CANDIDATE_STRATEGIES})")
    return max(candidates, key=_STRATEGY_KEYS[strategy])


def best_analysis_for_token(token: str, annotator, min_stem_len: int = MIN_STEM_LEN,
                             strategy: str = DEFAULT_CANDIDATE_STRATEGY,
                             verbal_level: str = DEFAULT_VERBAL_MORPHOLOGY_LEVEL,
                             allow_informal_orthography: bool = False) -> Optional[CandidateAnalysis]:
    return select_best_analysis(
        enumerate_candidate_analyses(token, annotator, min_stem_len, verbal_level, allow_informal_orthography),
        strategy,
    )


def fasttext_predict_raw(annotator, token: str) -> Tuple[str, float]:
    """Raw (language, confidence) from Annotator._ft_predict on the lowercased
    token; ("", 0.0) for an empty token."""
    if not token:
        return "", 0.0
    lang, prob = annotator._ft_predict(token.lower())
    return lang, float(prob)


def is_non_turkish_stem_evidence(annotator, stem: str, cfg) -> bool:
    """Mirrors the stem test in Annotator._detect_mixed_no_apostrophe:
    English-lexicon membership, or fastText EN at >= cfg["FT_EN_MIN"]."""
    stem_l = stem.lower()
    if stem_l in annotator.english_freq_words:
        return True
    lang, prob = fasttext_predict_raw(annotator, stem_l)
    return lang == "EN" and prob >= cfg["FT_EN_MIN"]


# Typo-tolerant English-stem lookup limited to one edit type: inserting a
# duplicated consonant ("triger" -> "trigger"). Returns a separate recovered
# lexicon string; the token and stem are never modified.
MIN_STEM_LEN_FOR_ORTHOGRAPHIC_RECOVERY = 5
_ENGLISH_VOWELS = frozenset("aeiouAEIOU")


def _duplicated_consonant_variants(stem: str) -> List[str]:
    """Strings obtained by duplicating exactly one existing consonant in place."""
    variants = []
    for i, ch in enumerate(stem):
        if ch.isalpha() and ch not in _ENGLISH_VOWELS:
            variants.append(stem[:i + 1] + ch + stem[i + 1:])
    return variants


def recover_english_stem_via_duplicated_consonant(stem: str, annotator) -> Optional[str]:
    """First English-lexicon hit among the duplicated-consonant variants of
    `stem`, else None. Returns None for stems shorter than
    MIN_STEM_LEN_FOR_ORTHOGRAPHIC_RECOVERY; does not consult fastText."""
    if len(stem) < MIN_STEM_LEN_FOR_ORTHOGRAPHIC_RECOVERY:
        return None
    for variant in _duplicated_consonant_variants(stem):
        if variant.lower() in annotator.english_freq_words:
            return variant.lower()
    return None


def classify_candidate(pred_label: str, pred_item: str, annotator, cfg, min_stem_len: int = MIN_STEM_LEN,
                        strategy: str = DEFAULT_CANDIDATE_STRATEGY,
                        verbal_level: str = DEFAULT_VERBAL_MORPHOLOGY_LEVEL,
                        allow_informal_orthography: bool = False,
                        allow_stem_orthographic_recovery: bool = False,
                        return_candidates: bool = False):
    """(is_candidate, reason, analysis[, candidates]) from predicted
    information only, never the gold label. UID is always a candidate; NE needs
    a plausible Turkish suffix analysis; TR also needs non-Turkish stem
    evidence; EN/OTHER/MIXED/LANG3 never. `analysis` is returned even for
    non-candidates so features still see the morphology.
    """
    def _result(is_cand, reason, analysis, candidates):
        if return_candidates:
            return is_cand, reason, analysis, candidates
        return is_cand, reason, analysis

    if pred_label in NON_CANDIDATE_LABELS:
        return _result(False, None, None, [])

    if pred_label == "UID":
        candidates = enumerate_candidate_analyses(pred_item, annotator, min_stem_len, verbal_level, allow_informal_orthography)
        analysis = select_best_analysis(candidates, strategy)
        return _result(True, CANDIDATE_REASON_UID, analysis, candidates)

    if pred_label == "NE":
        candidates = enumerate_candidate_analyses(pred_item, annotator, min_stem_len, verbal_level, allow_informal_orthography)
        analysis = select_best_analysis(candidates, strategy)
        if analysis is not None:
            return _result(True, CANDIDATE_REASON_NE_SUFFIX, analysis, candidates)
        return _result(False, None, analysis, candidates)

    if pred_label == "TR":
        candidates = enumerate_candidate_analyses(pred_item, annotator, min_stem_len, verbal_level, allow_informal_orthography)
        analysis = select_best_analysis(candidates, strategy)
        if analysis is not None and is_non_turkish_stem_evidence(annotator, analysis.stem, cfg):
            return _result(True, CANDIDATE_REASON_TR_SUSPECT_STEM, analysis, candidates)

        # Only for informal-orthography analyses whose ordinary stem evidence
        # failed; the recovered form goes on a copy, never on `analysis.stem`.
        if (allow_stem_orthographic_recovery and analysis is not None
                and analysis.informal_suffix_normalization):
            recovered = recover_english_stem_via_duplicated_consonant(analysis.stem, annotator)
            if recovered is not None:
                recovered_analysis = _dataclass_replace(
                    analysis, stem_orthographic_recovery=True, recovered_english_stem=recovered)
                return _result(True, CANDIDATE_REASON_TR_SUSPECT_STEM, recovered_analysis, candidates)

        # Not a candidate, but the analysis is still returned for feature extraction.
        return _result(False, None, analysis, candidates)

    # Unexpected label: not a candidate.
    return _result(False, None, None, [])


def build_structured_feature_dict(pred_item: str, pred_label: str, analysis: Optional[CandidateAnalysis],
                                   annotator, cfg,
                                   is_candidate: bool = False,
                                   candidate_reason: Optional[str] = None,
                                   include_batch_a: bool = False,
                                   include_batch_c: bool = True,
                                   include_batch_b: bool = False,
                                   include_batch_g: bool = False,
                                   candidate_analyses: Optional[List[CandidateAnalysis]] = None,
                                   candidate_strategy: str = DEFAULT_CANDIDATE_STRATEGY,
                                   include_batch_d: bool = False) -> dict:
    """Structured feature dict for one row, from predicted item/label only.

    Opt-in batches: C (default on) evidence deltas; A parser metadata (its
    is_candidate/candidate_reason come from the caller); B morphological
    counts; G candidate ambiguity (needs `candidate_analyses`); D stem-language
    confidence. distinct_stem_count and stem_length_ratio are omitted because
    they duplicate analysis_candidate_count and split_position_ratio.
    """
    token_l = pred_item.lower()
    ft_lang, ft_prob = fasttext_predict_raw(annotator, pred_item)

    feats = {
        "pred_label": pred_label,
        "token_length": len(pred_item),
        "has_apostrophe": ("'" in pred_item) or ("’" in pred_item),
        "has_hyphen": "-" in pred_item,
        "has_digit": any(c.isdigit() for c in pred_item),
        "initial_capital": bool(pred_item[:1].isupper()) if pred_item else False,
        "all_uppercase": pred_item.isupper() if pred_item else False,
        "has_turkish_char": any(c in TURKISH_CHARS for c in pred_item),
        "in_turkish_top": token_l in annotator.turkish_freq_top,
        "in_turkish_all": token_l in annotator.turkish_freq_all,
        "in_english_freq": token_l in annotator.english_freq_words,
        "ft_lang": ft_lang or "NONE",
        "ft_prob": ft_prob,
        "is_ne": pred_label == "NE",
    }

    if analysis is not None:
        stem_l = analysis.stem.lower()
        stem_ft_lang, stem_ft_prob = fasttext_predict_raw(annotator, analysis.stem)
        feats.update({
            "stem_length": analysis.stem_length,
            "suffix_length": analysis.suffix_length,
            "suffix_segment_count": analysis.segment_count,
            "fully_consumed_suffix": True,
            "stem_in_turkish_top": stem_l in annotator.turkish_freq_top,
            "stem_in_turkish_all": stem_l in annotator.turkish_freq_all,
            "stem_in_english_freq": stem_l in annotator.english_freq_words,
            "stem_ft_lang": stem_ft_lang or "NONE",
            "stem_ft_prob": stem_ft_prob,
            "informal_suffix_normalization": analysis.informal_suffix_normalization,
            # The recovered form stays on analysis.recovered_english_stem, not a feature.
            "stem_orthographic_recovery": analysis.stem_orthographic_recovery,
        })
    else:
        feats.update({
            "stem_length": 0,
            "suffix_length": 0,
            "suffix_segment_count": 0,
            "fully_consumed_suffix": False,
            "stem_in_turkish_top": False,
            "stem_in_turkish_all": False,
            "stem_in_english_freq": False,
            "stem_ft_lang": "NONE",
            "stem_ft_prob": 0.0,
            "informal_suffix_normalization": False,
            "stem_orthographic_recovery": False,
        })

    if include_batch_c:
        lexicon_hit = feats["stem_in_english_freq"]
        fasttext_hit = feats["stem_ft_lang"] == "EN" and feats["stem_ft_prob"] >= cfg["FT_EN_MIN"]
        if lexicon_hit and fasttext_hit:
            stem_evidence_strength = "both"
        elif lexicon_hit:
            stem_evidence_strength = "lexicon_only"
        elif fasttext_hit:
            stem_evidence_strength = "fasttext_only"
        else:
            stem_evidence_strength = "none"

        feats["ft_prob_delta"] = feats["stem_ft_prob"] - feats["ft_prob"]
        feats["ft_lang_agreement"] = feats["ft_lang"] == feats["stem_ft_lang"]
        feats["stem_evidence_strength"] = stem_evidence_strength

    if include_batch_a:
        feats["analysis_source"] = analysis.source if analysis is not None else "none"
        feats["candidate_reason"] = candidate_reason if candidate_reason is not None else "none"
        feats["is_candidate"] = bool(is_candidate)
        token_length = feats["token_length"]
        if analysis is not None and token_length > 0:
            feats["split_position_ratio"] = analysis.split_position / token_length
        else:
            feats["split_position_ratio"] = 0.0

    if include_batch_b:
        if analysis is not None:
            morph_tag_count = analysis.feature_count
            has_case = any(t.startswith("Case=") for t in analysis.ud_feats)
            has_plural = "Number=Plur" in analysis.ud_feats
            has_possessive = "Poss=Yes" in analysis.ud_feats
            has_derivational_suffix = any(t.startswith("Deriv=") for t in analysis.deriv)
            has_verbal_morphology = any(t.startswith("Verbal=") for t in analysis.deriv)
        else:
            morph_tag_count = 0
            has_case = False
            has_plural = False
            has_possessive = False
            has_derivational_suffix = False
            has_verbal_morphology = False

        # Thresholds on tag count only: 0-1 simple, 2-3 moderate, 4+ complex.
        if morph_tag_count <= 1:
            morph_complexity = "simple"
        elif morph_tag_count <= 3:
            morph_complexity = "moderate"
        else:
            morph_complexity = "complex"

        feats["morph_tag_count"] = morph_tag_count
        feats["has_case"] = has_case
        feats["has_plural"] = has_plural
        feats["has_possessive"] = has_possessive
        feats["has_derivational_suffix"] = has_derivational_suffix
        feats["has_verbal_morphology"] = has_verbal_morphology
        feats["morph_complexity"] = morph_complexity

    # best_second_score_gap is deliberately omitted: the selection key is a
    # lexicographic tuple, which has no meaningful scalar gap.
    if include_batch_g:
        candidates = candidate_analyses or []
        n = len(candidates)
        sources = {c.source for c in candidates}
        has_nominal_verbal_competition = ("nominal" in sources and "verbal" in sources)
        if n == 0:
            selection_is_unique = True
        else:
            key_fn = _STRATEGY_KEYS[candidate_strategy]
            best_key = max(key_fn(c) for c in candidates)
            n_at_best = sum(1 for c in candidates if key_fn(c) == best_key)
            selection_is_unique = (n_at_best == 1)

        feats["analysis_candidate_count"] = n
        feats["selection_is_unique"] = selection_is_unique
        feats["has_nominal_verbal_competition"] = has_nominal_verbal_competition

    # stem_lexicon_contrast uses stem_in_turkish_all as the Turkish side, the
    # counterpart of the single-granularity English list.
    if include_batch_d:
        stem_english_confidence = feats["stem_ft_prob"] if feats["stem_ft_lang"] == "EN" else 0.0
        stem_turkish_confidence = feats["stem_ft_prob"] if feats["stem_ft_lang"] == "TR" else 0.0

        en_hit = feats["stem_in_english_freq"]
        tr_hit = feats["stem_in_turkish_all"]
        if en_hit and tr_hit:
            stem_lexicon_contrast = "both"
        elif en_hit:
            stem_lexicon_contrast = "english_only"
        elif tr_hit:
            stem_lexicon_contrast = "turkish_only"
        else:
            stem_lexicon_contrast = "neither"

        feats["stem_english_confidence"] = stem_english_confidence
        feats["stem_turkish_confidence"] = stem_turkish_confidence
        feats["stem_lexicon_contrast"] = stem_lexicon_contrast

    return feats


# ---------------------------------------------------------------------------
# Residual verbal MIXED detector
#
# Runs from apply_reranker() after the frozen reranker, on tokens still labeled
# UID/TR. Reuses _parse_experimental_verbal_suffix at PHASE_4E; the one gap it
# fills is 1st-person-plural agreement ("ladık"), via VERBAL_FIRST_PERSON_PLURAL.
# ---------------------------------------------------------------------------

# Structured like VERBAL_FIRST_PERSON_SINGULAR: bare forms plus the fused
# past+1pl portmanteau (-dık/-dik/...). Bare "-k" is excluded: too many
# ordinary words end in it (büyük, küçük, artık, gerek).
VERBAL_FIRST_PERSON_PLURAL = {
    "ık": frozenset({"Verbal=1stPersonPlural"}), "ik": frozenset({"Verbal=1stPersonPlural"}),
    "uk": frozenset({"Verbal=1stPersonPlural"}), "ük": frozenset({"Verbal=1stPersonPlural"}),
    "ız": frozenset({"Verbal=1stPersonPlural"}), "iz": frozenset({"Verbal=1stPersonPlural"}),
    "uz": frozenset({"Verbal=1stPersonPlural"}), "üz": frozenset({"Verbal=1stPersonPlural"}),
    "dık": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "dik": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "duk": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "dük": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "tık": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "tik": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "tuk": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
    "tük": frozenset({"Verbal=Past", "Verbal=1stPersonPlural"}),
}

RESIDUAL_VERBALIZER_TAGS = frozenset({"Verbal=Verbalizer", "Verbal=PassiveInchoative"})


def parse_residual_verbal_suffix(suffix: str, allow_informal_orthography: bool = True):
    """Peels 1st-person-plural agreement (bare or past-fused; outermost, so
    tried first), then delegates the rest to _parse_experimental_verbal_suffix
    at PHASE_4E. Returns the same (segments, tags, fully_consumed,
    used_informal) shape."""
    s = suffix.lower()
    plural_segment = None
    plural_tags = frozenset()

    items = sorted(VERBAL_FIRST_PERSON_PLURAL.items(), key=lambda kv: -len(kv[0]))
    for end, tagset in items:
        if s.endswith(end):
            plural_segment = end
            plural_tags = tagset
            s = s[:-len(end)]
            break

    rest_segments, rest_tags, fully_consumed, used_informal = _parse_experimental_verbal_suffix(
        s, level=VERBAL_MORPHOLOGY_PHASE_4E, allow_informal_orthography=allow_informal_orthography)

    segments = list(rest_segments) + ([plural_segment] if plural_segment else [])
    tags = frozenset(rest_tags | plural_tags)
    return segments, tags, fully_consumed, used_informal


def enumerate_residual_verbal_candidates(token: str, min_stem_len: int = MIN_STEM_LEN):
    """Right-to-left split enumeration, verbal-only. Apostrophe tokens are
    excluded (condition 10; handled by the production apostrophe-MIXED path).
    Returns dicts {stem, suffix, split_position, segments, tags, used_informal}.
    """
    candidates = []
    if "'" in token or "’" in token:
        return candidates
    tok_l = token.lower()
    n = len(tok_l)
    for split in range(n - min_stem_len, 0, -1):
        stem, suffix = tok_l[:split], tok_l[split:]
        if len(stem) < min_stem_len or not suffix:
            continue
        segments, tags, fully_consumed, used_informal = parse_residual_verbal_suffix(suffix)
        if not fully_consumed or not tags:
            continue
        candidates.append({
            "stem": stem, "suffix": suffix, "split_position": split,
            "segments": segments, "tags": tags, "used_informal": used_informal,
        })
    return candidates


def _residual_verbal_looks_like_proper_name_or_noise(token: str) -> bool:
    """Condition 10: probable proper names, acronyms, codes, URLs, mentions,
    hashtags and other non-lexical noise. Local import keeps cs_pipeline (and
    its stanza import) out of module load."""
    from cs_pipeline import is_other_token
    if is_other_token(token):
        return True
    if token.isupper() and len(token) > 1:
        return True  # acronym / all-caps code
    if any(c.isdigit() for c in token):
        return True  # alphanumeric identifier / product code
    if token[:1].isupper():
        return True  # capitalized -- probable proper name/brand
    return False


def _residual_verbal_direct_english_evidence(annotator, stem: str) -> bool:
    """Condition 6: a direct annotator.english_freq_words hit only. Narrower
    than is_non_turkish_stem_evidence: the fastText fallback gave
    byte-identical offline results, so production does not use it."""
    return stem.lower() in annotator.english_freq_words


def _residual_verbal_has_lexicon_confirmed_competing_nominal_stem(token_l: str, annotator) -> bool:
    """Condition 8: whether a nominal suffix split leaves a Turkish-lexicon-
    confirmed stem. Requiring confirmation avoids trivial matches (stripping
    "ım" from "uploadlamışım" leaves the nonsense stem "uploadlamış")."""
    n = len(token_l)
    for split in range(n - 2, 0, -1):
        stem, suf = token_l[:split], token_l[split:]
        if len(suf) < 2:
            continue
        segments, ud_feats, deriv, amb = annotator._parse_tr_suffixes_full(suf)
        if "Unparsed=Leftover" in deriv or not (ud_feats or deriv or amb):
            continue
        if stem in annotator.turkish_freq_all or stem in annotator.turkish_freq_top:
            return True
    return False


def evaluate_residual_verbal_promotion(token: str, annotator, cfg, strict_lexicon_only: bool = True):
    """Apply every residual-verbal promotion condition. Returns
    (should_promote, chosen_candidate_or_None, reason); reason is always set.

    `strict_lexicon_only=True` (production, never overridden) requires direct
    English-lexicon evidence; False also accepts fastText evidence and is for
    offline experiments only.
    """
    if _residual_verbal_looks_like_proper_name_or_noise(token):
        return False, None, "proper_name_or_noise"

    candidates = enumerate_residual_verbal_candidates(token)
    if not candidates:
        return False, None, "no_verbal_candidate"

    qualifying = []
    for c in candidates:
        if not (c["tags"] & RESIDUAL_VERBALIZER_TAGS):
            continue  # condition 2: explicit verbalizer/passive-inchoative required
        stem = c["stem"]
        if stem in annotator.turkish_freq_all or stem in annotator.turkish_freq_top:
            continue  # condition 7: stem must be absent from the Turkish lexicon
        has_evidence = (_residual_verbal_direct_english_evidence(annotator, stem) if strict_lexicon_only
                         else is_non_turkish_stem_evidence(annotator, stem, cfg))
        if not has_evidence:
            continue  # condition 6
        qualifying.append(c)

    if not qualifying:
        return False, None, "no_qualifying_candidate"

    if _residual_verbal_has_lexicon_confirmed_competing_nominal_stem(token.lower(), annotator):
        return False, None, "competing_nominal_analysis"  # condition 8

    # condition 9: uniqueness -- prefer the longest (most specific) stem;
    # a tie among qualifying candidates means the analysis is not unique
    # enough to act on.
    qualifying.sort(key=lambda c: -len(c["stem"]))
    best = qualifying[0]
    if len(qualifying) > 1 and len(qualifying[1]["stem"]) == len(best["stem"]):
        return False, None, "ambiguous_analysis"

    return True, best, "promoted"


# ---------------------------------------------------------------------------
# Production integration of the frozen Phase 5F reranker
# (Baseline + Batch A + Batch C + pruned Batch G, threshold 0.85)
#
# apply_reranker() runs three stages, each only on tokens still unpromoted:
#   1. frozen reranker: UID/NE/TR -> MIXED when probability >= threshold
#   2. residual verbal detector (strict English-lexicon evidence): UID/TR -> MIXED
#   3. UID->TR resolver, gated by UID_TR_RESOLVER_ENABLED: UID -> TR
# Other labels are never demoted. Loading never raises: on failure the bundle is
# None and the input text is returned unchanged.
# ---------------------------------------------------------------------------


# False disables the UID->TR stage and restores the pre-integration output exactly
# (tests/test_reranker_integration.py).
UID_TR_RESOLVER_ENABLED = True

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "models")
MODEL_FILENAME = "model.joblib"
VECTORIZER_FILENAME = "vectorizer.joblib"
METADATA_FILENAME = "metadata.json"

# Frozen Phase 5F contract: metadata.json that does not match is rejected.
EXPECTED_THRESHOLD = 0.85
EXPECTED_THRESHOLD_POLICY = "precision_0.75"
EXPECTED_MODEL_TYPE = "LogisticRegression"

# metadata.json has no explicit phase field, so Phase 5F is identified
# structurally by which batch feature groups are present.
_BATCH_A_KEYS = frozenset({"analysis_source", "candidate_reason", "is_candidate", "split_position_ratio"})
_BATCH_C_KEYS = frozenset({"stem_evidence_strength", "ft_prob_delta", "ft_lang_agreement"})
_BATCH_G_PRUNED_KEYS = frozenset({"analysis_candidate_count", "selection_is_unique", "has_nominal_verbal_competition"})
_BATCH_G_REMOVED_KEY = "distinct_stem_count"  # must be ABSENT (pruned in Phase 5F)
_BATCH_B_KEYS = frozenset({"morph_tag_count", "has_case", "has_plural", "has_possessive",
                           "has_derivational_suffix", "has_verbal_morphology", "morph_complexity"})
_BATCH_D_KEYS = frozenset({"stem_english_confidence", "stem_turkish_confidence", "stem_lexicon_contrast"})


class ReRankerBundle(NamedTuple):
    """Everything needed to run the frozen reranker at inference time (data only)."""
    model: object
    tfidf: object
    dictvec: object
    threshold: float
    metadata: dict


def _warn(message: str) -> None:
    """Best-effort stderr note; never raises. Also kept as the failure reason
    reported by the in-progress load_reranker_bundle() call."""
    global _last_warning
    _last_warning = message
    try:
        print(f"[reranker_integration] {message}", file=sys.stderr)
    except Exception:
        pass


_last_warning: Optional[str] = None

# Why the last load returned None (None after success); shown in the GUI so
# users can tell MIXED detection is running rule-based only.
last_load_failure: Optional[str] = None


def validate_metadata(metadata: dict) -> Tuple[bool, List[str]]:
    """Check that `metadata` describes the frozen configuration: Batch A, C
    and pruned G enabled (no distinct_stem_count), B and D disabled, threshold
    0.85. Returns (is_valid, reasons); `reasons` lists every failed check.
    """
    reasons: List[str] = []

    if not isinstance(metadata, dict):
        return False, ["metadata is not a JSON object"]

    feature_config = metadata.get("feature_configuration")
    if not isinstance(feature_config, dict):
        return False, ["metadata missing 'feature_configuration'"]

    structured = set(feature_config.get("structured_features") or [])
    if not structured:
        reasons.append("feature_configuration.structured_features is empty or missing")

    if not _BATCH_A_KEYS.issubset(structured):
        reasons.append(f"Batch A features missing: {sorted(_BATCH_A_KEYS - structured)}")
    if not _BATCH_C_KEYS.issubset(structured):
        reasons.append(f"Batch C features missing: {sorted(_BATCH_C_KEYS - structured)}")
    if not _BATCH_G_PRUNED_KEYS.issubset(structured):
        reasons.append(f"Batch G (pruned) features missing: {sorted(_BATCH_G_PRUNED_KEYS - structured)}")
    if _BATCH_G_REMOVED_KEY in structured:
        reasons.append(f"'{_BATCH_G_REMOVED_KEY}' present -- Batch G is not in its pruned Phase 5F form")

    batch_b_present = _BATCH_B_KEYS & structured
    if batch_b_present:
        reasons.append(f"Batch B features present (must be disabled): {sorted(batch_b_present)}")

    batch_d_present = _BATCH_D_KEYS & structured
    if batch_d_present:
        reasons.append(f"Batch D features present (must be disabled): {sorted(batch_d_present)}")

    model_type = feature_config.get("model", {}).get("type") if isinstance(feature_config.get("model"), dict) else None
    if model_type != EXPECTED_MODEL_TYPE:
        reasons.append(f"unexpected model type: {model_type!r} (expected {EXPECTED_MODEL_TYPE!r})")

    thresholds = metadata.get("selected_thresholds")
    if not isinstance(thresholds, dict):
        reasons.append("metadata missing 'selected_thresholds'")
    else:
        if thresholds.get("policy") != EXPECTED_THRESHOLD_POLICY:
            reasons.append(f"threshold policy mismatch: expected {EXPECTED_THRESHOLD_POLICY!r}, "
                            f"got {thresholds.get('policy')!r}")
        active = thresholds.get("active")
        if active != EXPECTED_THRESHOLD:
            reasons.append(f"active threshold mismatch: expected {EXPECTED_THRESHOLD!r}, got {active!r}")
        by_policy_key = thresholds.get("best_precision_ge_0.75")
        if by_policy_key != EXPECTED_THRESHOLD:
            reasons.append(f"'best_precision_ge_0.75' threshold mismatch: expected {EXPECTED_THRESHOLD!r}, "
                            f"got {by_policy_key!r}")

    return (len(reasons) == 0), reasons


def get_threshold(metadata: dict) -> Optional[float]:
    """Frozen threshold from `metadata`, or None if absent or non-numeric (no
    hardcoded fallback: a missing threshold makes the bundle unusable).
    """
    if not isinstance(metadata, dict):
        return None
    thresholds = metadata.get("selected_thresholds")
    if not isinstance(thresholds, dict):
        return None
    value = thresholds.get("best_precision_ge_0.75")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _load_json_safely(path: str) -> Optional[dict]:
    """Parsed JSON, or None on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _warn(f"could not read/parse {path!r}: {e}")
        return None


def _load_joblib_file(joblib_module, path: str):
    """Load one joblib file; None (with a warning) on failure."""
    try:
        return joblib_module.load(path)
    except Exception as e:
        _warn(f"could not load {path!r}: {e}")
        return None


def load_reranker_bundle(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[ReRankerBundle]:
    """Load and validate the bundle from `model_dir`, or return None (never
    raises) and set last_load_failure. Called once per session by the GUI."""
    global _last_warning, last_load_failure
    _last_warning = None
    try:
        bundle = _load_reranker_bundle(model_dir)
    except Exception as e:
        _warn(f"unexpected error while loading the reranker: {e}")
        bundle = None
    last_load_failure = None if bundle is not None else (_last_warning or "unknown error")
    return bundle


def _load_reranker_bundle(model_dir: str) -> Optional[ReRankerBundle]:
    try:
        import joblib  # local import: no reranker dependency at module load time
    except ImportError as e:
        _warn(f"joblib/scikit-learn not available: {e}")
        return None

    metadata_path = os.path.join(model_dir, METADATA_FILENAME)
    metadata = _load_json_safely(metadata_path)
    if metadata is None:
        return None

    is_valid, reasons = validate_metadata(metadata)
    if not is_valid:
        _warn("metadata validation failed, rejecting model: " + "; ".join(reasons))
        return None

    threshold = get_threshold(metadata)
    if threshold is None:
        _warn("could not extract a valid threshold from metadata; rejecting model")
        return None

    model = _load_joblib_file(joblib, os.path.join(model_dir, MODEL_FILENAME))
    if model is None:
        return None

    vectorizer_bundle = _load_joblib_file(joblib, os.path.join(model_dir, VECTORIZER_FILENAME))
    if vectorizer_bundle is None:
        return None

    if not isinstance(vectorizer_bundle, dict):
        _warn(f"vectorizer bundle has unexpected type: {type(vectorizer_bundle)!r}")
        return None

    tfidf = vectorizer_bundle.get("tfidf")
    dictvec = vectorizer_bundle.get("dictvec")
    if tfidf is None or dictvec is None:
        _warn("vectorizer bundle missing 'tfidf' or 'dictvec'")
        return None

    # Under scikit-learn <= 1.7 the model unpickles but every prediction raises
    # ('multi_class'), which would silently disable the reranker per token.
    try:
        probe = sp.hstack([tfidf.transform([""]), dictvec.transform([{}])]).tocsr()
        model.predict_proba(probe)
    except Exception as e:
        try:
            import sklearn
            version = sklearn.__version__
        except Exception:
            version = "unknown"
        _warn(f"model cannot run with the installed scikit-learn {version}: {e}")
        return None

    return ReRankerBundle(model=model, tfidf=tfidf, dictvec=dictvec, threshold=threshold, metadata=metadata)


# Operates on Annotator.annotate()'s raw text (blank-line-separated blocks with
# SentenceID/MatrixLang/EmbedLang meta rows and "Item\tLabel" token rows) and
# returns the same format.

# Non-candidate rows never reach classify_candidate.
_CANDIDATE_ELIGIBLE_LABELS = frozenset({"UID", "NE", "TR"})

# Labels Annotator.annotate() feeds into _decide_matrix_embed (OTHER and UID
# are not counted); reproduced to recompute the vote after a label change.
_LABELS_COUNTED_FOR_MATRIX_EMBED = frozenset({"TR", "EN", "MIXED", "NE"})


def _parse_block_lines(block_text: str) -> List[Dict]:
    """Parse one blank-line-delimited block of annotate() output into line
    records, using annotation_model.is_meta_row_token to tell meta rows from
    token rows."""
    records: List[Dict] = []
    for ln in block_text.split("\n"):
        if ln == "":
            records.append({"kind": "blank"})
            continue
        parts = ln.split("\t")
        if len(parts) != 2:
            # Not the 2-field shape annotate() emits: pass through verbatim.
            records.append({"kind": "raw", "raw": ln})
            continue
        first, second = parts
        if annotation_model.is_meta_row_token(first):
            records.append({"kind": "meta", "name": first, "value": second})
        else:
            records.append({"kind": "token", "item": first, "label": second})
    return records


def _rerank_token_label(item: str, label: str, annotator, cfg, bundle: ReRankerBundle) -> str:
    """Possibly overridden label for one token, using the frozen feature
    generation and threshold. Never raises; any failure keeps the label."""
    if label not in _CANDIDATE_ELIGIBLE_LABELS:
        return label
    try:
        is_candidate, reason, analysis, candidates = classify_candidate(
            label, item, annotator, cfg, return_candidates=True)
        if not is_candidate:
            return label

        feats = build_structured_feature_dict(
            item, label, analysis, annotator, cfg,
            is_candidate=is_candidate, candidate_reason=reason,
            include_batch_a=True, include_batch_c=True, include_batch_g=True,
            candidate_analyses=candidates, candidate_strategy=DEFAULT_CANDIDATE_STRATEGY)

        X = sp.hstack([bundle.tfidf.transform([item]), bundle.dictvec.transform([feats])]).tocsr()
        prob = bundle.model.predict_proba(X)[0, 1]
        if prob >= bundle.threshold:
            return "MIXED"
        return label
    except Exception as e:
        _warn(f"reranking failed for {item!r} (label={label!r}), keeping original: {e}")
        return label


# Residual verbal stage: UID/TR only (never NE), checked against the label after the reranker pass.
_RESIDUAL_VERBAL_ELIGIBLE_LABELS = frozenset({"UID", "TR"})


def _apply_residual_verbal_promotion(item: str, label: str, annotator, cfg: dict) -> str:
    """Possibly overridden label via the strict residual verbal detector
    (strict_lexicon_only left at its default True). Never raises; any failure
    keeps the label."""
    if label not in _RESIDUAL_VERBAL_ELIGIBLE_LABELS:
        return label
    try:
        promote, _candidate, _reason = evaluate_residual_verbal_promotion(item, annotator, cfg)
        return "MIXED" if promote else label
    except Exception as e:
        _warn(f"residual verbal detection failed for {item!r} (label={label!r}), keeping original: {e}")
        return label


# UID->TR stage: UID only, checked against the label after both earlier stages, so it never pre-empts a MIXED promotion.
_UID_TR_RESOLVER_ELIGIBLE_LABELS = frozenset({"UID"})


def _apply_uid_to_tr_resolver_stage(item: str, label: str, annotator, cfg: dict,
                                     matrix_lang: Optional[str]) -> str:
    """Possibly overridden label via the UID->TR resolver (decide()); returns
    `label` unchanged when UID_TR_RESOLVER_ENABLED is False. Never raises; any
    failure keeps the label."""
    if not UID_TR_RESOLVER_ENABLED:
        return label
    if label not in _UID_TR_RESOLVER_ELIGIBLE_LABELS:
        return label
    try:
        decision = decide(item, label, annotator, cfg, matrix_lang=matrix_lang)
        return decision.proposed_label if decision.promote else label
    except Exception as e:
        _warn(f"UID->TR resolver failed for {item!r} (label={label!r}), keeping original: {e}")
        return label


def apply_reranker(annotated_text: str, annotator, cfg: dict, bundle: Optional[ReRankerBundle]) -> str:
    """Apply the three stages to annotate() text (same format in and out).
    MatrixLang/EmbedLang are recomputed only in changed blocks; unchanged
    blocks, or everything when `bundle` is None, come back byte-for-byte.
    Never raises.
    """
    if bundle is None:
        return annotated_text

    blocks = annotated_text.split("\n\n")
    new_blocks = []
    for block in blocks:
        records = _parse_block_lines(block)
        matrix_lang = next(
            (rec["value"] for rec in records if rec["kind"] == "meta" and rec["name"] == "MatrixLang"),
            None)

        changed = False
        for rec in records:
            if rec["kind"] != "token":
                continue
            new_label = _rerank_token_label(rec["item"], rec["label"], annotator, cfg, bundle)
            if new_label != rec["label"]:
                rec["label"] = new_label
                changed = True

        for rec in records:
            if rec["kind"] != "token":
                continue
            new_label = _apply_residual_verbal_promotion(rec["item"], rec["label"], annotator, cfg)
            if new_label != rec["label"]:
                rec["label"] = new_label
                changed = True

        for rec in records:
            if rec["kind"] != "token":
                continue
            new_label = _apply_uid_to_tr_resolver_stage(rec["item"], rec["label"], annotator, cfg, matrix_lang)
            if new_label != rec["label"]:
                rec["label"] = new_label
                changed = True

        if not changed:
            new_blocks.append(block)
            continue

        labels_in_sent = [rec["label"] for rec in records
                           if rec["kind"] == "token" and rec["label"] in _LABELS_COUNTED_FOR_MATRIX_EMBED]
        matrix, embed = annotator._decide_matrix_embed(labels_in_sent, cfg)
        for rec in records:
            if rec["kind"] == "meta" and rec["name"] == "MatrixLang":
                rec["value"] = matrix
            elif rec["kind"] == "meta" and rec["name"] == "EmbedLang":
                rec["value"] = embed

        lines = []
        for rec in records:
            if rec["kind"] == "blank":
                lines.append("")
            elif rec["kind"] == "raw":
                lines.append(rec["raw"])
            elif rec["kind"] == "meta":
                lines.append(f"{rec['name']}\t{rec['value']}")
            else:  # token
                lines.append(f"{rec['item']}\t{rec['label']}")
        new_blocks.append("\n".join(lines))

    return "\n\n".join(new_blocks)


# ---------------------------------------------------------------------------
# UID->TR resolver
#
# Third stage of apply_reranker(): UID -> TR only, never UID -> EN.
# check_eligibility() excludes any token with an English-root-plus-Turkish-suffix
# analysis, so genuine MIXED candidates are never contested. cs_pipeline is
# imported lazily so importing this module does not pull in stanza.
# ---------------------------------------------------------------------------


MIN_TOKEN_LEN = 2

# local@domain.tld shape; cs_pipeline.MENTION_RE matches only a bare "@handle".
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _has_apostrophe(token: str) -> bool:
    return "'" in token or "’" in token


def _looks_like_email(token: str) -> bool:
    return bool(EMAIL_RE.match(token))


def _looks_like_all_caps_acronym(token: str) -> bool:
    return token.isupper() and len(token) > 1


def _looks_like_alphanumeric_identifier(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _looks_like_probable_proper_name(token: str) -> bool:
    """Capitalization heuristic (same as the residual verbal stage's
    proper-name guard), since this stage never sees NER output."""
    return token[:1].isupper()


def _has_strong_direct_english_match(token_l: str, annotator) -> bool:
    return token_l in annotator.english_freq_words


def _has_english_root_turkish_suffix_analysis(token: str, annotator, cfg) -> bool:
    """True if any enumerated split (nominal or verbal, at the broadest verbal
    level) has a stem with non-Turkish evidence, i.e. would be a MIXED
    candidate; such tokens are left to the MIXED stages.
    """
    candidates = enumerate_candidate_analyses(
        token, annotator, verbal_level=VERBAL_MORPHOLOGY_PHASE_4E)
    return any(is_non_turkish_stem_evidence(annotator, c.stem, cfg) for c in candidates)


def check_eligibility(token: str, label: str, annotator, cfg) -> Tuple[bool, str]:
    """Whether `token` labeled `label` may be considered. Returns
    (is_eligible, reason); reason is always populated.
    """
    if label != "UID":
        return False, "label_not_uid"
    if not token or len(token) < MIN_TOKEN_LEN:
        return False, "too_short"

    from cs_pipeline import is_other_token  # local import: avoids stanza at module load
    if is_other_token(token):
        return False, "other_token (url/mention/hashtag/number/punctuation/code/emoji)"
    if _looks_like_email(token):
        return False, "email"
    if _has_apostrophe(token):
        return False, "apostrophe"
    if _looks_like_all_caps_acronym(token):
        return False, "all_caps_acronym"
    if _looks_like_alphanumeric_identifier(token):
        return False, "alphanumeric_identifier"
    if _looks_like_probable_proper_name(token):
        return False, "probable_proper_name_or_protected_ne"

    token_l = token.lower()
    if _has_strong_direct_english_match(token_l, annotator):
        return False, "strong_direct_english_lexicon_match"
    if _has_english_root_turkish_suffix_analysis(token, annotator, cfg):
        return False, "english_root_turkish_suffix_analysis"

    return True, "eligible"


# Primary signals: independent and individually insufficient; see PROMOTION_THRESHOLD.
PRIMARY_SIGNAL_WEIGHTS: Dict[str, int] = {
    "trusted_lexicon": 3,
    "suffix_chain": 3,
    "fasttext_strong": 3,
    "orthographic": 2,
}
PRIMARY_SIGNALS = frozenset(PRIMARY_SIGNAL_WEIGHTS)

# Auxiliary signal (MatrixLang agreement): never counts toward MIN_PRIMARY_SIGNALS.
AUXILIARY_SIGNAL_WEIGHTS: Dict[str, int] = {
    "matrix_language": 1,
}

MIN_PRIMARY_SIGNALS = 2

# Two primary signals give at most 3+3=6, short of the threshold; promotion
# needs three primary signals (>= 8) or two worth >= 3 each plus the
# auxiliary bonus (== 7).
PROMOTION_THRESHOLD = 7

TURKISH_ORTHOGRAPHIC_CHARS = set("çÇğĞıİöÖşŞüÜ")


@dataclass(frozen=True)
class EvidenceItem:
    signal: str
    weight: int
    description: str


def _recovered_turkish_stem(token: str, annotator) -> Optional[str]:
    """Stem of the first nominal candidate (closed-class suffix chain only, not
    the verbal tables) confirmed in the trusted Turkish lexicon, in original
    case; None if there is none.
    """
    for candidate in enumerate_candidate_analyses(token, annotator):
        if candidate.source != "nominal":
            continue
        stem_l = candidate.stem.lower()
        if stem_l in annotator.turkish_freq_all or stem_l in annotator.turkish_freq_top:
            return candidate.stem
    return None


def extract_evidence(token: str, annotator, cfg, matrix_lang: Optional[str] = None) -> List[EvidenceItem]:
    """Explainable evidence list for `token`. Does not check eligibility;
    decide() does."""
    evidence: List[EvidenceItem] = []
    token_l = token.lower()

    if token_l in annotator.turkish_freq_all or token_l in annotator.turkish_freq_top:
        evidence.append(EvidenceItem(
            "trusted_lexicon", PRIMARY_SIGNAL_WEIGHTS["trusted_lexicon"],
            "complete token found in trusted Turkish lexicon"))
    else:
        stem = _recovered_turkish_stem(token, annotator)
        if stem is not None:
            evidence.append(EvidenceItem(
                "trusted_lexicon", PRIMARY_SIGNAL_WEIGHTS["trusted_lexicon"],
                f"recovered Turkish stem '{stem}' found in trusted lexicon"))

    if annotator._has_valid_turkish_nominal_analysis(token_l):
        evidence.append(EvidenceItem(
            "suffix_chain", PRIMARY_SIGNAL_WEIGHTS["suffix_chain"],
            "valid Turkish suffix-chain analysis"))

    ft_min = cfg.get("FT_TR_MIN", 0.80)
    lang, prob = fasttext_predict_raw(annotator, token_l)
    if lang == "TR" and prob >= ft_min:
        evidence.append(EvidenceItem(
            "fasttext_strong", PRIMARY_SIGNAL_WEIGHTS["fasttext_strong"],
            f"strong Turkish fastText support (p={prob:.2f} >= {ft_min:.2f})"))

    if any(ch in TURKISH_ORTHOGRAPHIC_CHARS for ch in token):
        evidence.append(EvidenceItem(
            "orthographic", PRIMARY_SIGNAL_WEIGHTS["orthographic"],
            "Turkish-specific orthographic character present"))

    if matrix_lang == "TR":
        evidence.append(EvidenceItem(
            "matrix_language", AUXILIARY_SIGNAL_WEIGHTS["matrix_language"],
            "sentence-level MatrixLang is TR (weak auxiliary signal only)"))

    return evidence


def score_evidence(evidence: List[EvidenceItem]) -> int:
    return sum(e.weight for e in evidence)


def _primary_signal_count(evidence: List[EvidenceItem]) -> int:
    return sum(1 for e in evidence if e.signal in PRIMARY_SIGNALS)


class ResolverDecision(NamedTuple):
    token: str
    current_label: str
    proposed_label: str
    promote: bool
    score: int
    evidence: Tuple[EvidenceItem, ...]
    reason: str


def decide(token: str, label: str, annotator, cfg, matrix_lang: Optional[str] = None) -> ResolverDecision:
    """Eligibility, evidence, scoring and promotion decision for one token.
    Deterministic."""
    eligible, reason = check_eligibility(token, label, annotator, cfg)
    if not eligible:
        return ResolverDecision(token, label, label, False, 0, (), reason)

    evidence = extract_evidence(token, annotator, cfg, matrix_lang)
    score = score_evidence(evidence)
    promote = _primary_signal_count(evidence) >= MIN_PRIMARY_SIGNALS and score >= PROMOTION_THRESHOLD

    if promote:
        return ResolverDecision(token, label, "TR", True, score, tuple(evidence), "promoted")
    return ResolverDecision(token, label, label, False, score, tuple(evidence),
                             "ambiguous_or_insufficient_evidence")


def explain(decision: ResolverDecision) -> str:
    """Human-readable explanation: Token / Current / Proposed / Score / Evidence / Decision."""
    lines = [
        f"Token: {decision.token}",
        f"Current: {decision.current_label}",
        f"Proposed: {decision.proposed_label}",
        f"Score: {decision.score}",
        "Evidence:",
    ]
    if decision.evidence:
        for e in decision.evidence:
            lines.append(f"- {e.description}")
    else:
        lines.append("- (none)")
    verdict = "promote" if decision.promote else "retain"
    lines.append(f"Decision: {verdict} ({decision.reason})")
    return "\n".join(lines)


# Only UID rows are inspected, so others never appear in `decisions`.
_RESOLVER_ELIGIBLE_LABELS = frozenset({"UID"})


def apply_uid_to_tr_resolver(annotated_text: str, annotator, cfg: dict) -> Tuple[str, List[ResolverDecision]]:
    """Offline evaluation only (production calls decide() directly):
    (new_text, decisions) with eligible UID rows promoted to TR. Never raises.
    """
    labels_counted_for_matrix_embed = _LABELS_COUNTED_FOR_MATRIX_EMBED

    blocks = annotated_text.split("\n\n")
    new_blocks = []
    all_decisions: List[ResolverDecision] = []

    for block in blocks:
        records = _parse_block_lines(block)
        matrix_lang = next(
            (rec["value"] for rec in records if rec["kind"] == "meta" and rec["name"] == "MatrixLang"),
            None)

        changed = False
        for rec in records:
            if rec["kind"] != "token":
                continue
            if rec["label"] not in _RESOLVER_ELIGIBLE_LABELS:
                continue
            try:
                decision = decide(rec["item"], rec["label"], annotator, cfg, matrix_lang=matrix_lang)
            except Exception:
                continue
            all_decisions.append(decision)
            if decision.promote:
                rec["label"] = decision.proposed_label
                changed = True

        if not changed:
            new_blocks.append(block)
            continue

        labels_in_sent = [rec["label"] for rec in records
                           if rec["kind"] == "token" and rec["label"] in labels_counted_for_matrix_embed]
        new_matrix, new_embed = annotator._decide_matrix_embed(labels_in_sent, cfg)
        for rec in records:
            if rec["kind"] == "meta" and rec["name"] == "MatrixLang":
                rec["value"] = new_matrix
            elif rec["kind"] == "meta" and rec["name"] == "EmbedLang":
                rec["value"] = new_embed

        lines = []
        for rec in records:
            if rec["kind"] == "blank":
                lines.append("")
            elif rec["kind"] == "raw":
                lines.append(rec["raw"])
            elif rec["kind"] == "meta":
                lines.append(f"{rec['name']}\t{rec['value']}")
            else:  # token
                lines.append(f"{rec['item']}\t{rec['label']}")
        new_blocks.append("\n".join(lines))

    return "\n\n".join(new_blocks), all_decisions
