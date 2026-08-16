# reranking.py
"""Consolidated reranking module: MIXED-candidate generation/feature
engineering, production integration of the frozen reranker, and the
UID->TR resolver.

This file merges the previously separate mixed_reranker.py,
reranker_integration.py, and uid_resolver.py (see CLAUDE.md section 3 and
the merge commit) into one module -- a file-organization change only. No
behavior, threshold, model-loading, or fallback logic was altered by the
merge; every function/class/constant keeps its original name. Cross-module
references between the three former files (mr./ur./ri. qualifiers) are now
intra-module and unqualified; each former module's own docstring is kept
verbatim under its own section header below for full context.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field, replace as _dataclass_replace
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

import scipy.sparse as sp

import annotation_model


# ===========================================================================
# formerly mixed_reranker.py -- MIXED-candidate generation & feature
# engineering (Phase 2+), plus the residual verbal MIXED detector.
# ===========================================================================

"""Experimental MIXED-vs-KEEP_ORIGINAL reranker support code (Phase 2+).

Isolated from the production annotation flow: nothing in this module is
imported by cs_pipeline.py or cs_annotator_app.py directly, and nothing
here alters Annotator.annotate() or any other production behaviour -- this
module's own functions are pure candidate/feature-generation logic with no
side effects on the rule-based pipeline. It reuses Annotator._parse_tr_suffixes_full
and Annotator._ft_predict as-is (read-only calls into the existing
pipeline) rather than duplicating the suffix tables or the fastText/lexicon
lookup logic.

As of Phase 6, this module's logic IS exercised on every normal annotation
run -- indirectly, via reranker_integration.py, which cs_annotator_app.py
imports and which calls classify_candidate()/build_structured_feature_dict()
from here to rerank UID/NE/TR candidates to MIXED. "Isolated" above
describes this module's own import graph and lack of side effects, not
whether its logic affects what a user sees -- see reranker_integration.py
and README.md's "MIXED-Token Reranker" section for the current, wired-in
production behaviour.

Candidate generation (enumerate_candidate_analyses / select_best_analysis)
implements a conservative stem/suffix split search: since
Annotator._parse_tr_suffixes_full only ever receives a suffix substring (see
docs/annotation-model.md), this module enumerates every right-to-left split
point of a token and asks the existing parser whether the tail parses as a
valid Turkish suffix chain.

Phase 4A/4B/4C: a SEPARATE, EXPERIMENTAL verbal-suffix table
(_parse_experimental_verbal_suffix and the VERBAL_* dicts below) has been
added, used only as a fallback when Annotator._parse_tr_suffixes_full fails
to fully consume a suffix. This does NOT modify cs_pipeline.py or its
nominal suffix tables in any way -- it is new, additional, experimental-only
logic that exists solely in this module.

Which morphology categories are actually active is controlled by the
`verbal_level` parameter (VERBAL_MORPHOLOGY_PHASE_4A/4B/4C1/4E), threaded
through enumerate_candidate_analyses / best_analysis_for_token /
classify_candidate. DEFAULT_VERBAL_MORPHOLOGY_LEVEL is PHASE_4C1 (promoted
in Phase 4D -- it matched or beat Phase 4A on the precision>=0.75 threshold
policy). Phase 4B (past/evidential/progressive/future/2nd-person) did not
improve the frozen test evaluation and was therefore NOT promoted to the
active default; its code/tables/tests remain fully available via
level=PHASE_4B. Phase 4C-1 (1st person singular agreement) extends
PHASE_4A, not PHASE_4B. Phase 4E (union of 4A+4B+4C1) is also available but
not the default.

Phase 4C-2 adds a SEPARATE, independent, opt-in fallback
(_informal_suffix_variants / _match_verbal_table's allow_informal_orthography
parameter) that normalizes ASCII/informal spelling (s->ş, i->ı) only when
matching a verbal suffix against the tables above, only after the literal
spelling failed to match. It never touches the original token, stem text,
lexical/fastText lookup input, or exported data, and is disabled by default
(allow_informal_orthography=False everywhere).

Phase 5A-5G: build_structured_feature_dict() additionally exposes several
independently-gated, opt-in structured feature groups ("batches") consumed
by tools/train_mixed_reranker.py's LogisticRegression reranker. As of Phase
5F/5G, the active experimental baseline is Batch A + Batch C + a pruned
Batch G; Batch B and Batch D were evaluated and experimentally rejected
(kept available, off by default, for reproducibility only). See the
"Experimental: MIXED-Token Reranker" section of README.md for the full
batch inventory, current benchmark numbers, and rejection rationale.
"""


MIN_STEM_LEN = 2

TURKISH_CHARS = set("çÇğĞıİöÖşŞüÜ")

# The 7-label schema, see CLAUDE.md section 2. Order fixed only for
# deterministic feature-key documentation; not semantically significant.
SCHEMA_LABELS = ("TR", "EN", "MIXED", "UID", "NE", "LANG3", "OTHER")

# Candidate buckets, defined using ONLY information available from the
# machine-generated (predicted) annotation -- never the gold label. See
# Phase 2 Decision 3/4.
CANDIDATE_REASON_UID = "UID"
CANDIDATE_REASON_NE_SUFFIX = "NE_SUFFIX"
CANDIDATE_REASON_TR_SUSPECT_STEM = "TR_SUSPECT_STEM"

# Predicted labels that can never be candidates (Decision 3: "Do not include
# predicted EN, OTHER, MIXED, or LANG3 in the candidate subset").
NON_CANDIDATE_LABELS = frozenset({"EN", "OTHER", "MIXED", "LANG3"})


@dataclass(frozen=True)
class CandidateAnalysis:
    """One plausible stem/suffix split for a token.

    split_position = len(stem), i.e. the index in the token where the
    suffix begins. Kept explicitly (rather than re-derived from stem length)
    so the "earliest split position" tie-break in select_best_analysis is
    unambiguous even if stem_length ties are possible through other means
    in the future.

    `source`: "nominal" if Annotator._parse_tr_suffixes_full (production,
    unmodified) fully consumed the suffix; "verbal" if it only fully
    consumed via the Phase 4A+ EXPERIMENTAL verbal-suffix fallback in this
    module. Defaults to "nominal" for backward compatibility with existing
    call sites/tests predating Phase 4A.

    `informal_suffix_normalization`: True if this analysis was only reached
    by applying the Phase 4C-2 EXPERIMENTAL informal-orthography fallback
    (see _informal_suffix_variants) during verbal-suffix matching. Always
    False unless that fallback was both enabled and actually needed for
    this specific analysis. Defaults to False for backward compatibility.
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
    # Phase 4D: set only when the TR-bucket candidacy check recovered a
    # real English lexicon entry from `stem` via the duplicated-consonant
    # fallback (see recover_english_stem_via_duplicated_consonant). `stem`
    # itself is NEVER modified/replaced -- `recovered_english_stem` holds
    # the separately-recovered lexicon form for inspection/feature purposes.
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
        """Total morphological feature count (UD + derivational + ambiguity
        tags combined) -- used only as a selection tie-break, not exposed
        as a standalone modeling feature beyond suffix_segment_count."""
        return len(self.ud_feats) + len(self.deriv) + len(self.amb)


# ---------------------------------------------------------------------------
# Phase 4A/4B/4C-1: EXPERIMENTAL verbal-suffix table. Exists ONLY in this
# module -- cs_pipeline.py's Annotator._parse_tr_suffixes_full is not
# modified, called, or reimplemented differently.
#
# Phase 4A added: infinitive, verbalizer, passive/inchoative.
# Phase 4B additionally added (and only added): past, evidential, present
#   progressive, future, 2nd person agreement.
# Phase 4C-1 additionally added (and only added): 1st person singular
#   agreement (bare -m, -ım/-im/-um/-üm, and the fused past+1sg forms
#   -dım/-dim/-dum/-düm/-tım/-tim/-tum/-tüm).
#
# Which of these are ACTIVE at any given call is controlled by the `level`
# parameter threaded through every function below (VERBAL_MORPHOLOGY_*
# constants). The DEFAULT level is PHASE_4A -- Phase 4B's benchmark did not
# improve the frozen test evaluation, so it was NOT promoted to the active
# experimental default; its code, tables, and tests remain fully available
# and selectable, just not used unless a caller explicitly asks for them.
# ---------------------------------------------------------------------------
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
# Combined for a single "tense/aspect/mood" peeling stage -- the four
# categories above are mutually exclusive (a finite verb carries exactly
# one), so they are tried together as one dict, longest key first.
VERBAL_TENSE_ASPECT_MOOD = {**VERBAL_PAST, **VERBAL_EVIDENTIAL, **VERBAL_PROGRESSIVE, **VERBAL_FUTURE}

VERBAL_SECOND_PERSON = {
    "sınız": "Verbal=2ndPersonPlural", "siniz": "Verbal=2ndPersonPlural",
    "sunuz": "Verbal=2ndPersonPlural", "sünüz": "Verbal=2ndPersonPlural",
    "sın": "Verbal=2ndPersonSingular", "sin": "Verbal=2ndPersonSingular",
    "sun": "Verbal=2ndPersonSingular", "sün": "Verbal=2ndPersonSingular",
}

# Phase 4C-1 ONLY. Bare/possessive-shaped forms tag as 1st-person-singular
# alone; the four fused past+1sg forms tag as BOTH Past and
# 1stPersonSingular since they genuinely encode both pieces of information
# even though matched as a single literal string (matching the exact,
# closed list given for Phase 4C-1 -- no other morphology).
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

# Phase 4E ONLY. A union of VERBAL_SECOND_PERSON (Phase 4B, values
# normalized from plain strings to singleton frozensets here so both
# tables can share one lookup) and VERBAL_FIRST_PERSON_SINGULAR (Phase
# 4C-1, values already frozensets), used together as a single combined
# "agreement" stage. Does not modify either source table.
VERBAL_AGREEMENT_COMBINED_4E = {
    **{k: frozenset({v}) for k, v in VERBAL_SECOND_PERSON.items()},
    **VERBAL_FIRST_PERSON_SINGULAR,
}

VERBAL_MORPHOLOGY_PHASE_4A = "phase4a"
VERBAL_MORPHOLOGY_PHASE_4B = "phase4b"
VERBAL_MORPHOLOGY_PHASE_4C1 = "phase4c1"
# Phase 4E: union of Phase 4A (infinitive/verbalizer/passive-inchoative,
# always active at every level) + Phase 4B (tense/evidential/progressive/
# future/2nd-person) + Phase 4C-1 (1st-person-singular). Does NOT change
# the behaviour of phase4a/phase4b/phase4c1 in any way -- each remains
# independently selectable and byte-for-byte identical to before; phase4e
# is purely an ADDITIONAL level.
VERBAL_MORPHOLOGY_PHASE_4E = "phase4e"
VERBAL_MORPHOLOGY_LEVELS = (VERBAL_MORPHOLOGY_PHASE_4A, VERBAL_MORPHOLOGY_PHASE_4B,
                            VERBAL_MORPHOLOGY_PHASE_4C1, VERBAL_MORPHOLOGY_PHASE_4E)
# Phase 4D: promoted to Phase 4C-1 (1st person singular agreement, extends
# Phase 4A) as the active experimental default -- it matched or beat Phase
# 4A on the precision>=0.75 threshold policy (see Phase 4C-1 benchmark).
# Unchanged by Phase 4E -- phase4e is available but not the default.
DEFAULT_VERBAL_MORPHOLOGY_LEVEL = VERBAL_MORPHOLOGY_PHASE_4C1

# ---------------------------------------------------------------------------
# Phase 4C-2 ONLY: EXPERIMENTAL informal-orthography fallback for verbal-
# suffix MATCHING ONLY. Never touches the original token, the stem text, any
# lexical/fastText lookup input, or exported data -- it only ever generates
# alternate spellings of the small local suffix substring being compared
# against the VERBAL_* tables above, and only when the literal/standard
# spelling failed to match. Disabled unless a caller explicitly passes
# allow_informal_orthography=True.
# ---------------------------------------------------------------------------
INFORMAL_ORTHOGRAPHY_MAP = {"s": "ş", "i": "ı"}


def _informal_suffix_variants(s: str) -> List[str]:
    """Generate candidate standard-orthography respellings of `s` for
    VERBAL SUFFIX TABLE MATCHING ONLY (Phase 4C-2). Returns only variants
    that actually differ from `s`; the literal/unmodified `s` is always
    tried by the caller first, elsewhere, before this fallback is consulted."""
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
    """Try to match the END of `s` against `table` (a {suffix_string: tag_or_tagset}
    dict), longest key first. Returns (matched_end, tag_or_tagset, used_informal)
    or None. `matched_end`'s length (not the informal variant's) is what the
    caller must strip from the ORIGINAL `s` -- safe because every mapping in
    INFORMAL_ORTHOGRAPHY_MAP is a same-length, one-to-one character
    substitution, so `end`'s length is identical in both spellings.
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
    """EXPERIMENTAL verbal-suffix parser, candidate-generator-only.

    `level` controls which stages below are active:
      - PHASE_4A: stages 3 (infinitive) and 4 (verbalizer/passive-inchoative) only.
      - PHASE_4B: adds stages 1 (2nd person agreement) and 2 (tense/aspect/mood).
      - PHASE_4C1: adds stage 1b (1st person singular agreement) instead of
        stages 1/2 -- Phase 4C-1 is defined as an extension of PHASE_4A, not
        PHASE_4B (per Phase 4C-1 instructions: "benchmark against Phase 4A"),
        so PHASE_4C1 does NOT enable 2nd person agreement or the general
        tense/aspect/mood stage; it ONLY adds the closed list of 1st-person-
        singular forms (bare and fused-with-past), tried as an alternative
        to infinitive at the same "finite vs. non-finite" branch point.
      - PHASE_4E (Phase 4E): the union of PHASE_4B and PHASE_4C1 -- 1st AND
        2nd person agreement tried together in one combined stage, then the
        same tense/aspect/mood stage as PHASE_4B. Does not alter PHASE_4B or
        PHASE_4C1's own behaviour at all; it is a wholly additional level.

    `allow_informal_orthography` (Phase 4C-2): if True, any stage below that
    fails to match the literal/standard spelling retries against informal-
    orthography variants of the CURRENT remaining substring (s -> ş, i -> ı).
    Disabled by default; independent of `level`.

    Peels, right-to-left:
      1. [PHASE_4B: 2nd person only. PHASE_4E: 2nd person + 1st person
         singular combined] agreement, longest key first.
      2. [PHASE_4B, PHASE_4E] tense/aspect/mood (past/evidential/
         progressive/future), longest key first, regardless of whether
         stage 1 matched.
      1b. [PHASE_4C1 only] 1st person singular agreement (bare or fused-
          with-past), tried instead of stages 1/2.
      3. infinitive -mak/-mek [ALL levels], but ONLY if no finite stage
         above matched anything (infinitive is non-finite, mutually
         exclusive with any finite tense/person marker).
      4. passive/inchoative -lan/-len, OR (if no match) verbalizer -la/-le
         [ALL levels] -- always attempted last.

    Returns (segments, tags, fully_consumed, used_informal_orthography).
    `fully_consumed` is True only if nothing is left over -- a partial
    match (e.g. tense+person morphology outside the active level's scope)
    returns fully_consumed=False, so _analysis_from_suffix correctly
    rejects it rather than accepting a partial parse.
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
        # Combined agreement stage: 2nd person (4B) + 1st person singular
        # (4C-1) tried together, longest key first, exactly like stage 2's
        # tense/aspect/mood combination below.
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
    """Build a CandidateAnalysis for one (stem, suffix) pair if the suffix
    passes the plausibility checks (Phase 2 Decision 6); else None.

    Tries the production nominal parser first (unmodified,
    Annotator._parse_tr_suffixes_full); if that doesn't fully consume the
    suffix, falls back to the experimental verbal parser at `verbal_level`.
    The two are mutually exclusive per call -- whichever one fully consumes
    the suffix first wins, nominal taking priority since it is the
    production-backed, better-tested path.
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
    """Enumerate stem/suffix splits of `token` and keep only the plausible
    ones, per Phase 2 Decision 6:

    - stem length >= min_stem_len
    - suffix non-empty
    - at least one morphological feature or suffix segment produced
    - the suffix substring is fully consumed by the parser (no
      "Unparsed=Leftover" in the derivational tag set)

    Does not modify or duplicate Annotator._parse_tr_suffixes_full; calls it
    once per candidate split.

    Phase 1 fix (apostrophe handling): tokens containing an apostrophe
    (straight or curly) are no longer enumerated character-position-by-
    character-position -- that treated the apostrophe as an ordinary
    character, so every candidate stem retained it verbatim (e.g.
    "placeholder'" ), which broke lexicon lookups and depressed fastText
    confidence for otherwise-clear English stems. Instead, such tokens are
    routed through Annotator._split_mixed_apostrophe -- the SAME apostrophe
    split the production pipeline already uses for apostrophe-MIXED
    detection (cs_pipeline.py) -- to get a clean (base, suffix) pair with
    the apostrophe itself excluded from both sides, and NOT the
    right-to-left enumeration used for apostrophe-free tokens. If
    _split_mixed_apostrophe declines to split (e.g. an English contraction
    remnant, or more than one apostrophe), no candidates are produced for
    that token -- there is no naive-enumeration fallback, since falling back
    would silently reintroduce the exact behaviour being fixed.
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
# Phase 3A: changed from STRATEGY_LONGEST_STEM (the Phase 1 baseline) to
# STRATEGY_HIGHEST_SUFFIX_SEGMENTS, based on the Phase 2 three-way benchmark
# (artifacts/mixed_reranker/strategy_comparison/), where it matched or beat
# both other strategies on every reported dev/test metric. All three
# strategies remain fully available and selectable via --candidate-strategy;
# this only changes which one is used when a caller doesn't specify one.
# This is an experimental-pipeline default only -- cs_pipeline.py's
# production Annotator is untouched and has no notion of "strategy".
DEFAULT_CANDIDATE_STRATEGY = STRATEGY_HIGHEST_SUFFIX_SEGMENTS

_STRATEGY_KEYS = {
    # Each key function ranks candidates from a `max()` call, so higher
    # tuple values win; -split_position turns "earliest split" into a max.
    STRATEGY_LONGEST_STEM: lambda c: (c.stem_length, c.segment_count, c.feature_count, -c.split_position),
    STRATEGY_LONGEST_SUFFIX: lambda c: (c.suffix_length, c.segment_count, c.feature_count, -c.split_position),
    STRATEGY_HIGHEST_SUFFIX_SEGMENTS: lambda c: (c.segment_count, c.stem_length, c.feature_count, -c.split_position),
}


def select_best_analysis(candidates: List[CandidateAnalysis], strategy: str = DEFAULT_CANDIDATE_STRATEGY) -> Optional[CandidateAnalysis]:
    """Deterministic selection among multiple plausible analyses. Three
    strategies (Phase 2, configurable -- default unchanged from Phase 1):

      'longest_stem' (default, Phase 1 baseline):
        1. longest candidate stem
        2. highest number of parsed suffix segments
        3. highest total number of morphological features
        4. earliest split position (final tie-break)
        NOTE: this is an EXPERIMENTAL policy that differs from the
        production MIXED detector's own tie-break in
        Annotator._detect_mixed_no_apostrophe (cs_pipeline.py), which walks
        candidate suffixes longest-suffix-first -- i.e. effectively
        shortest-stem-first -- and takes the first one that passes all
        checks. This module does not change that production behaviour.

      'longest_suffix':
        1. longest candidate suffix (mirrors the production detector's own
           longest-suffix-first walk -- since every candidate here has
           already passed the same plausibility checks production applies
           per-candidate, "first suffix production would try that passes"
           and "longest suffix among already-valid candidates" coincide)
        2. highest number of parsed suffix segments
        3. highest total number of morphological features
        4. earliest split position (final tie-break)

      'highest_suffix_segments':
        1. highest number of parsed suffix segments
        2. longest candidate stem (tie-break only, per instruction -- NOT
           the primary criterion the way it is in 'longest_stem')
        3. highest total number of morphological features
        4. earliest split position (final tie-break)

    Raises ValueError for an unrecognized strategy name rather than
    silently falling back to a default.
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
    """Experimental adapter exposing Annotator._ft_predict's raw
    (language, confidence) output. Calls the existing, unmodified,
    lru_cached method -- does not alter fastText loading or prediction
    behaviour in any way. Lowercases the token first, matching every
    production call site's convention (_choose_label, _detect_mixed_no_apostrophe).
    Returns ("", 0.0) for an empty token (fastText cannot score it).
    """
    if not token:
        return "", 0.0
    lang, prob = annotator._ft_predict(token.lower())
    return lang, float(prob)


def is_non_turkish_stem_evidence(annotator, stem: str, cfg) -> bool:
    """Mirrors the production "is this stem plausibly English/non-Turkish"
    test used inside Annotator._detect_mixed_no_apostrophe (cs_pipeline.py):
    English-lexicon membership, or fastText EN prediction at >= cfg["FT_EN_MIN"].
    Reused here (not reimplemented differently) so the TR-bucket candidate
    definition stays consistent with the one real signal the production
    pipeline already trusts for "non-Turkish stem".
    """
    stem_l = stem.lower()
    if stem_l in annotator.english_freq_words:
        return True
    lang, prob = fasttext_predict_raw(annotator, stem_l)
    return lang == "EN" and prob >= cfg["FT_EN_MIN"]


# ---------------------------------------------------------------------------
# Phase 4D ONLY: EXPERIMENTAL, narrowly-scoped typo-tolerant English-stem
# lookup. Not general fuzzy matching -- exactly one edit type (inserting a
# duplicated consonant, e.g. "triger" -> "trigger", "droping" -> "dropping"),
# gated behind several preconditions (see recover_english_stem_via_
# duplicated_consonant and its caller in classify_candidate). Never modifies
# the original token or the stem text itself -- only ever returns a
# separate recovered lexicon string for the caller to store alongside the
# unmodified analysis.
# ---------------------------------------------------------------------------
MIN_STEM_LEN_FOR_ORTHOGRAPHIC_RECOVERY = 5
_ENGLISH_VOWELS = frozenset("aeiouAEIOU")


def _duplicated_consonant_variants(stem: str) -> List[str]:
    """All strings obtainable from `stem` by duplicating exactly one
    consonant in place (inserting a copy immediately after an existing
    occurrence). No substitutions, deletions, or insertions of a NEW
    letter -- only ever a repeat of a letter already present."""
    variants = []
    for i, ch in enumerate(stem):
        if ch.isalpha() and ch not in _ENGLISH_VOWELS:
            variants.append(stem[:i + 1] + ch + stem[i + 1:])
    return variants


def recover_english_stem_via_duplicated_consonant(stem: str, annotator) -> Optional[str]:
    """Phase 4D: try every duplicated-consonant variant of `stem` (see
    _duplicated_consonant_variants) against the English frequency lexicon;
    return the first lexicon entry found, or None. Does not consult
    fastText and does not modify `stem` -- returns a new, separate string.
    Returns None outright if `stem` is shorter than
    MIN_STEM_LEN_FOR_ORTHOGRAPHIC_RECOVERY (condition 4 of the Phase 4D
    brief), so callers can invoke this unconditionally once the other
    preconditions are already checked.
    """
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
    """Decide whether a token is a reranker candidate, using ONLY
    machine-generated information (pred_label, pred_item) -- never the gold
    label (Phase 2 Decision 3).

    `strategy` selects which CandidateAnalysis is preferred when a token has
    multiple plausible stem/suffix splits -- see select_best_analysis for
    the three available strategies. Only which analysis is picked (and
    therefore, for the TR bucket, which stem the non-Turkish-stem-evidence
    check runs against) changes with `strategy`; the candidacy RULE itself
    (which pred_label buckets are eligible, what the TR bucket additionally
    requires) is unchanged across strategies.

    Returns (is_candidate: bool, reason: str|None, analysis: CandidateAnalysis|None).

    Phase 5F: pass return_candidates=True to additionally get, as a 4th
    tuple element, the full list of CandidateAnalysis objects enumerated for
    this token (empty for pred_label in NON_CANDIDATE_LABELS, where no
    enumeration is attempted at all). This lets a caller (Batch G bookkeeping
    in build_structured_feature_dict) reuse the single enumeration this
    function already performs internally instead of enumerating a second
    time -- it does not change what is computed, only what is returned.
    Default False preserves the exact 3-tuple return of every existing caller.

    Buckets (Decision 4 of the preflight report / Phase 2 brief):
      - predicted UID                                        -> always a candidate
      - predicted NE with a plausible Turkish suffix analysis -> candidate
      - predicted TR with a plausible Turkish suffix analysis
        AND evidence of a potentially non-Turkish stem         -> candidate
        (Phase 4D: OR, failing that, a duplicated-consonant-recovered
        lexicon match -- see recover_english_stem_via_duplicated_consonant;
        opt-in via allow_stem_orthographic_recovery, off by default)

      - predicted EN, OTHER, MIXED, LANG3                      -> never a candidate

    IMPORTANT (Phase 1 fix): `analysis` is now returned whenever
    best_analysis_for_token found one, REGARDLESS of whether the token
    ended up a candidate. is_candidate/reason are unaffected -- the
    candidate-selection rule itself has not changed -- but a non-candidate
    NE/TR row can still carry a real CandidateAnalysis for feature
    extraction purposes. Previously the analysis was silently discarded
    (replaced with None) whenever candidacy was rejected, which zeroed out
    every morphological feature for those rows even when a valid suffix
    split existed.
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
        # Phase 1 fix: previously returned (False, None, None) here, discarding
        # `analysis` even when it was None anyway (a no-op in the NE branch --
        # analysis IS None on this path). Kept explicit for symmetry with the
        # TR branch below, where the fix is NOT a no-op.
        return _result(False, None, analysis, candidates)

    if pred_label == "TR":
        candidates = enumerate_candidate_analyses(pred_item, annotator, min_stem_len, verbal_level, allow_informal_orthography)
        analysis = select_best_analysis(candidates, strategy)
        if analysis is not None and is_non_turkish_stem_evidence(annotator, analysis.stem, cfg):
            return _result(True, CANDIDATE_REASON_TR_SUSPECT_STEM, analysis, candidates)

        # Phase 4D: narrowly-scoped typo-tolerant fallback. Applied ONLY
        # when ALL of: (1) a fully-consumed suffix analysis exists (implicit
        # -- `analysis is not None`); (2) it was only reachable via the
        # Phase 4C-2 informal-orthography fallback; (3) ordinary lexicon +
        # fastText evidence both just failed (we're past that check above);
        # (4)/(5) enforced inside recover_english_stem_via_duplicated_consonant
        # (stem length >= 5, and a duplicated-consonant lexicon match exists).
        # `analysis.stem` itself is never modified -- the recovered form is
        # stored separately on a REPLACED analysis object (not present in
        # `candidates`, which reflects the plain enumeration).
        if (allow_stem_orthographic_recovery and analysis is not None
                and analysis.informal_suffix_normalization):
            recovered = recover_english_stem_via_duplicated_consonant(analysis.stem, annotator)
            if recovered is not None:
                recovered_analysis = _dataclass_replace(
                    analysis, stem_orthographic_recovery=True, recovered_english_stem=recovered)
                return _result(True, CANDIDATE_REASON_TR_SUSPECT_STEM, recovered_analysis, candidates)

        # Phase 1 bug fix: a plausible suffix analysis can exist here even
        # though the non-Turkish-stem-evidence check failed (candidacy is
        # correctly rejected either way) -- previously this returned
        # (False, None, None), silently discarding a real analysis and
        # zeroing every morphological feature for the row downstream. The
        # candidacy decision (False) is unchanged; only feature propagation
        # is fixed by returning `analysis` here instead of None.
        return _result(False, None, analysis, candidates)

    # Any other/unexpected predicted label (should not occur in practice,
    # since the pipeline only ever emits the 7 schema labels): not a candidate.
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
    """Build the structured (non-char-n-gram) feature dict for one token
    row, per Phase 2 Decision 5B. All features are derived from the
    predicted (machine) item/label -- consistent with how a real production
    reranker would see the token at inference time, and with the candidate
    definition in classify_candidate, which is also pred-only.

    String-valued entries (pred_label, whole-token/stem fastText language)
    are intentionally left as raw strings: sklearn.feature_extraction.DictVectorizer
    one-hot encodes string-valued dict entries automatically and treats
    numeric/bool entries as-is, which is the "encoded appropriately" +
    scipy-sparse-combination pipeline required by Decision 6 of the brief.

    Phase 5A, Batch C (language-confidence interaction: ft_prob_delta,
    ft_lang_agreement, stem_evidence_strength) is included by default
    (include_batch_c=True, matching Phase 5A's own behaviour) but can be
    excluded for isolated ablation via include_batch_c=False, without
    deleting the implementation.

    Phase 5B, Batch A (parser-derived structure: analysis_source,
    candidate_reason, is_candidate, split_position_ratio) is EXCLUDED by
    default (include_batch_a=False) so every existing call site/test is
    unaffected; pass include_batch_a=True to add it. `is_candidate` and
    `candidate_reason` must be supplied by the caller (they are decided by
    classify_candidate, not recomputed here -- this function never infers
    or redefines them).

    Phase 5D, Batch B (morphological complexity: morph_tag_count, has_case,
    has_plural, has_possessive, has_derivational_suffix,
    has_verbal_morphology, morph_complexity) is EXCLUDED by default
    (include_batch_b=False). Every value is read directly off the already-
    computed `analysis.ud_feats`/`analysis.deriv`/`analysis.amb` frozensets
    -- no new parser call, no change to candidate generation or to what
    Annotator._parse_tr_suffixes_full / the experimental verbal parser
    produce. See CandidateAnalysis and _parse_tr_suffixes_full (cs_pipeline.py)
    for the tag vocabulary this reads: Case=*, Number=Plur, Poss=Yes,
    Person[psor]=*, Number[psor]=* in ud_feats; Deriv=*/DerivPOS=* (nominal)
    or Verbal=* (experimental verbal fallback, Phase 4A-4E) in deriv.

    Phase 5E, Batch G (candidate-analysis ambiguity / selection-process
    metadata: analysis_candidate_count, selection_is_unique,
    has_nominal_verbal_competition) is EXCLUDED by default
    (include_batch_g=False). Unlike every other batch, this one needs the
    FULL list of candidate analyses considered for this token (before
    select_best_analysis picked one) -- the caller must pass it via
    `candidate_analyses` (Phase 5F: reused from classify_candidate's own
    internal enumeration via return_candidates=True, not enumerated a
    second time; see tools/build_reranker_dataset.py). This function does
    not call enumerate_candidate_analyses or select_best_analysis itself
    for generation purposes -- it only reads
    `_STRATEGY_KEYS[candidate_strategy]` (the existing, unmodified selection
    key function) to check for ties among the SUPPLIED list, so neither
    candidate generation nor selection behaviour is changed.
    `candidate_analyses=None` or `[]` (no analysis attempted, or the token
    was never candidate-eligible) yields analysis_candidate_count=0,
    has_nominal_verbal_competition=False, and selection_is_unique=True
    (vacuously -- nothing exists to tie with).

    Phase 5F: a fourth candidate feature, distinct_stem_count, was
    implemented in Phase 5E and then REMOVED here after the integrity
    cleanup showed it is structurally identical to analysis_candidate_count
    under the current enumerate_candidate_analyses algorithm -- every split
    position yields a stem of a different length, so no two candidates for
    the same token can ever share a stem, making the two counts always
    equal (verified: 3097/3097 rows). Not reintroduced as a duplicate
    encoding.

    Phase 5G, Batch D (English-stem quality: stem_english_confidence,
    stem_turkish_confidence, stem_lexicon_contrast) is EXCLUDED by default
    (include_batch_d=False). Every value is a deterministic function of
    fields already computed above (stem_ft_lang, stem_ft_prob,
    stem_in_english_freq, stem_in_turkish_all) -- no new fastText or lexicon
    call. Per the Phase 5G inventory: stem_length_ratio (stem_length /
    token_length) was considered and OMITTED -- CandidateAnalysis.split_position
    equals len(stem) by construction (see its docstring/every enumeration
    call site), so stem_length_ratio would be mathematically identical to
    Batch A's existing split_position_ratio in every row; implementing it
    would be a pure duplicate. stem_is_short was also omitted -- it would
    require inventing an arbitrary length threshold not derived from any
    existing pipeline constant, and stem_length is already a raw baseline
    feature.
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
            # Phase 4C-2: marks whether this analysis was only reachable via
            # the informal-orthography fallback (s->ş, i->ı) during verbal-
            # suffix matching. False for every analysis unless that fallback
            # was both explicitly enabled and actually needed.
            "informal_suffix_normalization": analysis.informal_suffix_normalization,
            # Phase 4D: marks whether TR-bucket candidacy was granted via the
            # duplicated-consonant lexicon-recovery fallback rather than
            # ordinary lexicon/fastText evidence. `stem` itself is untouched;
            # the recovered lexicon form lives on the analysis object only
            # (analysis.recovered_english_stem), not as a separate raw-string
            # feature here.
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

    # Phase 5A, Batch C (language-confidence interaction). Computed from
    # values already present in `feats` above -- no new fastText or lexicon
    # calls, no changes to any existing key. Gated behind include_batch_c so
    # it can be excluded for an isolated Batch-A-only ablation (Phase 5B)
    # without deleting the implementation.
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

    # Phase 5B, Batch A (parser-derived structure). `is_candidate` and
    # `candidate_reason` are supplied by the caller -- this function never
    # recomputes or infers candidacy. Gated behind include_batch_a so it is
    # excluded from every existing call site/test unless explicitly requested.
    if include_batch_a:
        feats["analysis_source"] = analysis.source if analysis is not None else "none"
        feats["candidate_reason"] = candidate_reason if candidate_reason is not None else "none"
        feats["is_candidate"] = bool(is_candidate)
        token_length = feats["token_length"]
        if analysis is not None and token_length > 0:
            feats["split_position_ratio"] = analysis.split_position / token_length
        else:
            feats["split_position_ratio"] = 0.0

    # Phase 5D, Batch B (morphological complexity). Every value below is
    # read directly off analysis.ud_feats/deriv/amb (already produced by
    # Annotator._parse_tr_suffixes_full or the experimental verbal fallback
    # -- see CandidateAnalysis) -- no new parsing, no new lexicon/fastText
    # calls, no change to candidate generation. Gated behind include_batch_b
    # so every existing call site/test is unaffected by default.
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

        # Thresholds chosen on tag COUNT alone (not on any specific tag
        # identity): 0-1 tags = simple (e.g. a single case ending, or no
        # suffix at all), 2-3 = moderate (e.g. plural+case, or a full
        # possessive triple, or a derivational pair), 4+ = complex (stacked
        # morphology, e.g. possessive + case + plural).
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

    # Phase 5E, Batch G (candidate-analysis ambiguity / selection-process
    # metadata). Computed purely from the SUPPLIED `candidate_analyses` list
    # (the full enumeration result, not just the one selected analysis) --
    # no new parsing, no change to enumerate_candidate_analyses or
    # select_best_analysis. `best_second_score_gap` was considered and
    # deliberately omitted (see Phase 5E inventory): the active selection
    # key is a lexicographic tuple
    # (segment_count, stem_length, feature_count, -split_position), and a
    # tuple-vs-tuple lexicographic ordering has no well-defined scalar
    # "gap" without inventing an arbitrary weighting across incompatible
    # units -- so it is not exposed here.
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

    # Phase 5G, Batch D (English-stem quality). Every value below is a
    # deterministic function of fields already set above -- no new fastText
    # or lexicon call. stem_lexicon_contrast uses stem_in_turkish_all (not
    # stem_in_turkish_top) as the Turkish side, since english_freq_words has
    # only one granularity -- turkish_all is the comparable, equally-broad
    # counterpart.
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
# Residual verbal MIXED detector -- wired into production as of the phase
# authorized after Phase 4's offline validation (see CHANGELOG). Runs from
# reranker_integration.apply_reranker(), AFTER the frozen Phase 5F reranker
# pass, on whatever tokens are still labeled UID/TR post-rerank. Separate
# from, and does not alter, the frozen reranker's own candidate/feature/
# inference code above -- this section only ADDS new, independent
# parsing/evidence helpers, reusing the existing verbal-suffix machinery
# (_parse_experimental_verbal_suffix at the existing, unmodified PHASE_4E
# level -- DEFAULT_VERBAL_MORPHOLOGY_LEVEL itself is never touched) rather
# than duplicating it.
#
# Confirmed offline (not assumed): PHASE_4E already fully parses suffixes
# like "lamışım"/"ladı" -- simply not the frozen-model default level. The
# ONE genuine gap is 1st/3rd-person-plural verbal agreement (e.g. "ladık"),
# which no existing level models at all -- VERBAL_FIRST_PERSON_PLURAL below
# is the one new closed-class table this stage adds.
# ---------------------------------------------------------------------------

# Bare and past-fused 1st-person-plural agreement, structured exactly like
# VERBAL_FIRST_PERSON_SINGULAR above (bare forms stand alone; the
# past+1pl portmanteau is fused, since Turkish "-dık/-dik/-duk/-dük/-tık/
# -tik/-tuk/-tük" is a genuine irregular fusion, not two separately
# strippable morphemes). Bare single-character "-k" is deliberately
# EXCLUDED -- too promiscuous alone (many ordinary Turkish words end in
# "k": büyük, küçük, artık, gerek...); confirmed unnecessary for every
# target analysis this stage needs to recover.
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
    """Peels, right-to-left: (1) the NEW 1st-person-plural agreement stage
    above (bare or fused-with-past) -- agreement is the outermost/rightmost
    Turkish verbal morpheme, so tried first; (2) delegates whatever remains
    to _parse_experimental_verbal_suffix at the existing, unmodified
    PHASE_4E level (verbalizer/infinitive/passive-inchoative/tense-aspect-
    mood/2nd-person/1st-singular -- all reused unmodified; the module
    default VERBAL_MORPHOLOGY_PHASE_4C1 is never referenced here). Returns
    (segments, tags, fully_consumed, used_informal), the same shape as
    _parse_experimental_verbal_suffix so callers can treat it uniformly.
    """
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
    """Right-to-left split enumeration (mirrors enumerate_candidate_
    analyses's own loop shape), verbal-only via parse_residual_verbal_suffix
    -- nominal MIXED candidates are already covered by classify_candidate
    above and are out of this stage's scope. Apostrophe-bearing tokens are
    excluded (condition 10) -- they are handled by the existing production
    apostrophe-MIXED path, not this stage. Returns a list of dicts: {stem,
    suffix, split_position, segments, tags, used_informal}.
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
    """Condition 10: excludes probable proper names, acronyms, codes, URLs,
    mentions, hashtags, and other non-lexical noise. Reuses
    cs_pipeline.is_other_token read-only (local import -- this module stays
    import-free of cs_pipeline at module load time, matching its existing
    isolation contract; cs_pipeline.py itself never imports this module,
    so no circular import is introduced) plus a capitalization heuristic
    consistent with the one validated for the offline UID resolver's
    proper-name guard.
    """
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
    """Condition 6/production policy: STRICT English-lexicon evidence only
    -- a direct annotator.english_freq_words hit. Deliberately narrower
    than is_non_turkish_stem_evidence above (which also accepts a
    high-confidence fastText EN prediction): offline ablation found the
    fastText fallback produced byte-identical predictions and metrics to
    the strict-lexicon-only rule on both evaluated corpora, so it is not
    wired into production (this function) -- it remains available,
    unchanged, via is_non_turkish_stem_evidence for offline experimentation
    only.
    """
    return stem.lower() in annotator.english_freq_words


def _residual_verbal_has_lexicon_confirmed_competing_nominal_stem(token_l: str, annotator) -> bool:
    """Condition 8: whether a NOMINAL suffix split (annotator._parse_tr_
    suffixes_full, reused read-only, the same production parser
    _analysis_from_suffix above already calls) leaves a Turkish-lexicon-
    confirmed stem -- a genuinely plausible competing analysis, as opposed
    to trivially matching a short suffix on an otherwise-nonsense remainder
    (found necessary during offline testing: the nominal parser trivially
    matches a bare 2-char suffix like "ım" almost anywhere, e.g. stripping
    it from "uploadlamışım" leaves the nonsense stem "uploadlamış", which is
    not real competing evidence -- requiring the competing stem to itself
    be Turkish-lexicon-confirmed mirrors condition 7's own
    positive-evidence-only standard).
    """
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
    """Applies every residual-verbal promotion condition (see CLAUDE.md-
    adjacent production brief / CHANGELOG for the numbered list). Returns
    (should_promote: bool, chosen_candidate_or_None, reason: str) -- reason
    is always populated, even on rejection.

    `strict_lexicon_only` defaults to True (production policy: strict
    English-lexicon evidence only, per the offline ablation finding that
    broad fastText-fallback evidence produced identical results on both
    evaluated corpora while adding unmeasured false-positive risk on unseen
    tokens). reranker_integration.py's production call site never overrides
    this default. Passing False remains available for offline experiments
    only (mirrors is_non_turkish_stem_evidence's broader evidence rule) --
    not wired into any production call path.
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


# ===========================================================================
# formerly reranker_integration.py -- production integration of the frozen
# Phase 5F MIXED-token reranker.
# ===========================================================================

"""Production integration for the frozen Phase 5F MIXED-token reranker
(Baseline + Batch A + Batch C + pruned Batch G, threshold 0.85).

As of Phase 6.3, this module is wired into cs_annotator_app.py's
run_pipeline(): the reranker now runs automatically, on every annotation
request, as a post-processing stage. The production sequence is

    Annotator.annotate() -> apply_reranker() -> _ensure_matrix_embed_consistency() -> _populate_table()

Annotator.annotate() (cs_pipeline.py) remains the primary rule-based
annotation engine and is unmodified -- apply_reranker() only ever promotes
an already-produced UID/NE/TR label to MIXED when the frozen model's
probability clears the validated threshold; it never invents a label
Annotator.annotate() didn't already consider, and it cannot demote or
otherwise alter TR/EN/MIXED/OTHER/LANG3 labels. A final label a user sees
may therefore come from either the rule-based annotator alone, or from this
reranking stage.

As of the residual-verbal-detector integration (offline-validated, then
authorized for production -- see CHANGELOG), apply_reranker() runs a
SECOND, independent post-processing stage internally, strictly after the
frozen-reranker pass above and before Matrix/Embedded Language is
recomputed: mixed_reranker.evaluate_residual_verbal_promotion (strict
English-lexicon evidence only -- the broad fastText-fallback variant is
NOT used in production) may promote a token still labeled UID or TR after
the frozen-reranker pass to MIXED, recovering verbal code-switching forms
(e.g. "designladık") the frozen model's own feature set does not target.
This stage never touches NE, and never touches a token the reranker pass
already promoted to MIXED. It does not use, load, or alter the frozen
Phase 5F model, resources/models/*, or its 0.85 threshold in any way --
it is a separate, purely rule/lexicon-based layer. cs_annotator_app.py is
unchanged: this stage is folded into apply_reranker() itself, not a new
production entry point.

As of the UID->TR resolver integration (offline-validated across the real
corpus and two synthetic benchmarks with zero harmful changes and zero
regression on any other label -- see CHANGELOG/README "UID->TR Resolver"
section), apply_reranker() runs a THIRD, independent post-processing stage
internally, strictly after the residual verbal detector above and before
Matrix/Embedded Language is recomputed: uid_resolver.decide() (see
uid_resolver.py -- a separate, independently-testable, purely
rule/lexicon/fastText-based module, not duplicated here) may promote a
token STILL labeled UID after both prior stages to TR, when its
conservative, explainable, multi-signal evidence threshold is met. This
stage is gated behind UID_TR_RESOLVER_ENABLED (a named, explicit production
constant -- flip to False to restore the exact pre-integration output
byte-for-byte; see test_uid_to_tr_resolver_disabled_flag_restores_previous_output).
It never touches NE/EN/MIXED/OTHER/LANG3, never touches a token already
promoted to MIXED by either prior stage, never performs UID->EN, and never
retrains/rethresholds/modifies the frozen Phase 5F model, resources/models/*,
or any lexicon -- see uid_resolver.py's own module docstring for the full
hard-exclusion-gate and evidence-model design. cs_annotator_app.py is
unchanged: this stage, like the residual verbal detector before it, is
folded into apply_reranker() itself, not a new production entry point.

Model artifacts live in resources/models/ (model.joblib, vectorizer.joblib,
metadata.json), copied byte-for-byte from the frozen Phase 5F experiment
(artifacts/mixed_reranker/phase5f_batch_acg_pruned/) -- never regenerated or
edited here. See CHANGELOG.md / README.md "MIXED-Token Reranker" section
for the experiment history and rejected-batch rationale.

Loading is always lazy: load_reranker_bundle() only touches the filesystem
or imports joblib/scikit-learn when a caller explicitly calls it -- nothing
happens at import time or at application startup. cs_annotator_app.py calls
it once, on the first annotation request, and caches the result (bundle or
None) for the rest of the application session. Every failure mode (missing
dependency, missing file, invalid metadata, corrupted model) is caught and
results in load_reranker_bundle() returning None -- it never raises and
never shows a GUI dialog (it has no GUI dependency at all). When the bundle
is None, apply_reranker() returns its input completely unchanged, so
annotation output falls back to exactly the original rule-based behavior
with no crash and no interruption.

apply_reranker() reuses existing, already-validated production/experimental
code read-only rather than duplicating it: annotation_model.is_meta_row_token
for line classification (the same helper cs_annotator_app.py already uses),
mixed_reranker.classify_candidate/build_structured_feature_dict for feature
generation (byte-identical to how the frozen model was trained), and
Annotator._decide_matrix_embed for Matrix/Embedded Language recomputation
(the same private method Annotator.annotate() itself calls) -- exactly the
"reuse Annotator's own methods read-only" pattern mixed_reranker.py already
established for _parse_tr_suffixes_full/_ft_predict/_split_mixed_apostrophe.
Neither the parser, Annotator.annotate() internals, nor the saved-project
(.trenproj) and TXT/CSV export formats are touched by any of this -- the
text format in and out of apply_reranker() is identical to
Annotator.annotate()'s own output.
"""


# Named production feature flag for the UID->TR resolver stage (see module
# docstring). True = integrated (current production default, authorized by
# the offline evaluation in CHANGELOG/README). Set to False to disable the
# stage entirely and restore apply_reranker()'s exact pre-integration
# output -- covered by
# test_uid_to_tr_resolver_disabled_flag_restores_previous_output in
# tests/test_reranker_integration.py. Does not affect the frozen reranker
# or residual verbal detector stages, which run unconditionally either way.
UID_TR_RESOLVER_ENABLED = True

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "models")
MODEL_FILENAME = "model.joblib"
VECTORIZER_FILENAME = "vectorizer.joblib"
METADATA_FILENAME = "metadata.json"

# Frozen Phase 5F contract -- see CLAUDE.md-adjacent Phase 6 brief. Any
# metadata.json that doesn't match this fingerprint is rejected, not
# silently accepted.
EXPECTED_THRESHOLD = 0.85
EXPECTED_THRESHOLD_POLICY = "precision_0.75"
EXPECTED_MODEL_TYPE = "LogisticRegression"

# Batch feature-key fingerprints. There is no explicit "phase" string field
# anywhere in the metadata.json produced by tools/train_mixed_reranker.py --
# "is this Phase 5F" is defined here structurally, by which batch feature
# groups are actually present in feature_configuration.structured_features,
# not by a self-reported label that could drift from what the model was
# actually trained on.
_BATCH_A_KEYS = frozenset({"analysis_source", "candidate_reason", "is_candidate", "split_position_ratio"})
_BATCH_C_KEYS = frozenset({"stem_evidence_strength", "ft_prob_delta", "ft_lang_agreement"})
_BATCH_G_PRUNED_KEYS = frozenset({"analysis_candidate_count", "selection_is_unique", "has_nominal_verbal_competition"})
_BATCH_G_REMOVED_KEY = "distinct_stem_count"  # must be ABSENT (pruned in Phase 5F)
_BATCH_B_KEYS = frozenset({"morph_tag_count", "has_case", "has_plural", "has_possessive",
                           "has_derivational_suffix", "has_verbal_morphology", "morph_complexity"})
_BATCH_D_KEYS = frozenset({"stem_english_confidence", "stem_turkish_confidence", "stem_lexicon_contrast"})


class ReRankerBundle(NamedTuple):
    """Everything needed to run the frozen Phase 5F reranker at inference
    time. Purely a data container -- this module does not perform inference."""
    model: object
    tfidf: object
    dictvec: object
    threshold: float
    metadata: dict


def _warn(message: str) -> None:
    """Best-effort stderr note for debugging a rejected/failed load. Never
    allowed to raise -- printing itself is wrapped defensively."""
    try:
        print(f"[reranker_integration] {message}", file=sys.stderr)
    except Exception:
        pass


def validate_metadata(metadata: dict) -> Tuple[bool, List[str]]:
    """Validate that `metadata` describes the frozen Phase 5F configuration:
    Batch A enabled, Batch C enabled, Batch G enabled (pruned form, i.e.
    distinct_stem_count absent), Batch B disabled, Batch D disabled, and the
    precision>=0.75-selected threshold equal to 0.85.

    Returns (is_valid, reasons) -- `reasons` is empty iff is_valid is True,
    otherwise it lists every failed check (not just the first) so a caller
    or log can see the full picture.
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
    """Read the frozen precision>=0.75-selected threshold from `metadata`.
    Returns None if the expected key is absent or not a number -- callers
    must not fall back to a hardcoded default; a missing threshold means
    the bundle cannot be used.
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
    """Read and parse a JSON file, returning None on any failure (missing
    file, unreadable, malformed JSON) rather than raising."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _warn(f"could not read/parse {path!r}: {e}")
        return None


def _load_joblib_file(joblib_module, path: str):
    """Load one joblib file, returning None (and warning) on any failure
    instead of letting the exception propagate."""
    try:
        return joblib_module.load(path)
    except Exception as e:
        _warn(f"could not load {path!r}: {e}")
        return None


def load_reranker_bundle(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[ReRankerBundle]:
    """Lazily load and validate the frozen Phase 5F reranker bundle from
    `model_dir` (default: resources/models/ next to this file).

    Never raises. Returns None if:
      - joblib/scikit-learn are not importable,
      - model.joblib, vectorizer.joblib, or metadata.json are missing or
        unreadable,
      - metadata.json fails validate_metadata() (wrong batches enabled,
        wrong threshold, wrong model type, etc.),
      - the loaded vectorizer bundle doesn't contain the expected
        'tfidf'/'dictvec' keys,
      - joblib.load() itself raises for either file (e.g. a corrupted or
        incompatible pickle).

    Called once per application session, from cs_annotator_app.py's
    run_pipeline(), on the first annotation request; the result is cached
    there for the rest of the session (see App._reranker_bundle /
    App._reranker_load_attempted).
    """
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

    return ReRankerBundle(model=model, tfidf=tfidf, dictvec=dictvec, threshold=threshold, metadata=metadata)


# ---------------------------------------------------------------------------
# Reranking logic. Operates on Annotator.annotate()'s raw text output
# (blank-line-separated blocks; SentenceID/MatrixLang/EmbedLang meta rows;
# "Item\tLabel" token rows) and returns the exact same text format. Called
# from cs_annotator_app.py's run_pipeline() on every annotation request.
# ---------------------------------------------------------------------------

# Only these predicted labels are ever candidate-eligible (mirrors
# mixed_reranker.NON_CANDIDATE_LABELS/classify_candidate exactly) -- checked
# here too so non-candidate token rows never even reach classify_candidate.
_CANDIDATE_ELIGIBLE_LABELS = frozenset({"UID", "NE", "TR"})

# The labels Annotator.annotate() itself feeds into labels_in_sent for
# _decide_matrix_embed (see its per-token loop: OTHER and UID are never
# appended, TR/EN/MIXED/NE are). Reproduced here only to recompute the same
# vote after a label override -- not a redefinition of the rule itself.
_LABELS_COUNTED_FOR_MATRIX_EMBED = frozenset({"TR", "EN", "MIXED", "NE"})


def _parse_block_lines(block_text: str) -> List[Dict]:
    """Parse one '\\n\\n'-delimited block of annotate()'s raw text output
    into an ordered list of line records, using
    annotation_model.is_meta_row_token (the same helper cs_annotator_app.py
    already uses) to distinguish meta rows from token rows -- not a new
    parsing convention.
    """
    records: List[Dict] = []
    for ln in block_text.split("\n"):
        if ln == "":
            records.append({"kind": "blank"})
            continue
        parts = ln.split("\t")
        if len(parts) != 2:
            # Not the 2-field "Item\tLabel"/"Name\tValue" shape
            # annotate() always emits -- pass through verbatim rather than
            # guess at its meaning.
            records.append({"kind": "raw", "raw": ln})
            continue
        first, second = parts
        if annotation_model.is_meta_row_token(first):
            records.append({"kind": "meta", "name": first, "value": second})
        else:
            records.append({"kind": "token", "item": first, "label": second})
    return records


def _rerank_token_label(item: str, label: str, annotator, cfg, bundle: ReRankerBundle) -> str:
    """Return the (possibly overridden) label for one token, using the
    frozen Phase 5F feature generation and threshold. Never raises -- any
    unexpected failure leaves the original label untouched, exactly like a
    rejected candidate.
    """
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


# Residual verbal detector: eligible ONLY for UID/TR (never NE) -- see the
# production rule's condition list (CHANGELOG). Checked against a token's
# CURRENT label, i.e. after _rerank_token_label has already run for that
# token in apply_reranker()'s first pass, so this stage only ever sees the
# frozen reranker's own output, never the original rule-based label it may
# have overridden.
_RESIDUAL_VERBAL_ELIGIBLE_LABELS = frozenset({"UID", "TR"})


def _apply_residual_verbal_promotion(item: str, label: str, annotator, cfg: dict) -> str:
    """Return the (possibly overridden) label for one token via the strict,
    production-policy residual verbal detector (mixed_reranker.
    evaluate_residual_verbal_promotion, strict_lexicon_only left at its
    default True -- the broad fastText-fallback variant is never requested
    here). Never raises -- any unexpected failure leaves the label
    untouched, mirroring _rerank_token_label's own fail-safe contract, so a
    single token's residual-parsing error cannot interrupt annotation of
    the rest of the block.
    """
    if label not in _RESIDUAL_VERBAL_ELIGIBLE_LABELS:
        return label
    try:
        promote, _candidate, _reason = evaluate_residual_verbal_promotion(item, annotator, cfg)
        return "MIXED" if promote else label
    except Exception as e:
        _warn(f"residual verbal detection failed for {item!r} (label={label!r}), keeping original: {e}")
        return label


# UID->TR resolver: eligible ONLY for UID (never TR/NE/EN/MIXED/OTHER/LANG3)
# -- see uid_resolver.check_eligibility for the full hard-exclusion-gate
# list. Checked against a token's CURRENT label, i.e. after BOTH the frozen
# reranker pass and the residual verbal detector pass have already run for
# that token, so this stage only ever sees a token neither prior stage
# promoted -- it can never pre-empt a MIXED promotion, only ever act on
# whatever is left labeled UID once both have finished.
_UID_TR_RESOLVER_ELIGIBLE_LABELS = frozenset({"UID"})


def _apply_uid_to_tr_resolver_stage(item: str, label: str, annotator, cfg: dict,
                                     matrix_lang: Optional[str]) -> str:
    """Return the (possibly overridden) label for one token via the
    experimental-but-now-integrated UID->TR resolver (uid_resolver.decide(),
    UID->TR only -- see uid_resolver.py's module docstring for the complete
    hard-exclusion-gate and evidence-model design). Returns `label`
    unchanged, with no effect at all, whenever UID_TR_RESOLVER_ENABLED is
    False. Never raises -- any unexpected failure leaves the label
    untouched, mirroring _rerank_token_label's and
    _apply_residual_verbal_promotion's own fail-safe contract, so a single
    token's resolver failure cannot interrupt annotation of the rest of the
    block.
    """
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
    """Post-process Annotator.annotate()'s raw text output with the frozen
    Phase 5F reranker.

    Input and output are the EXACT same tab-separated text format
    annotate() itself produces -- no schema change, no new fields, no new
    meta rows. The rewrites are: (1) a token row's Label, when the frozen
    reranker promotes a UID/NE/TR candidate to MIXED; (1b) a token row's
    Label, when the residual verbal detector (strict evidence only, see
    _apply_residual_verbal_promotion) subsequently promotes a still-UID/TR
    candidate to MIXED -- always AFTER (1), never touching NE, and never
    reconsidering a token (1) already promoted; (1c) a token row's Label,
    when the UID->TR resolver (see _apply_uid_to_tr_resolver_stage,
    gated behind UID_TR_RESOLVER_ENABLED) subsequently promotes a
    still-UID candidate to TR -- always AFTER (1) and (1b), never touching
    TR/EN/MIXED/NE/OTHER/LANG3, and never reconsidering a token either
    prior stage already promoted; (2) an existing MatrixLang/EmbedLang
    row's value, recomputed via the unmodified Annotator._decide_matrix_embed,
    and only for blocks where a label actually changed by any of the three
    stages. Blocks with no label change are returned byte-for-byte
    identical to the input.

    If `bundle` is None (model unavailable), returns `annotated_text`
    completely unchanged -- this is the fallback path when
    load_reranker_bundle() failed for any reason, so annotation output is
    identical to the original rule-based-only behavior, INCLUDING the
    residual verbal stage and the UID->TR resolver stage (neither runs in
    this fallback path either, since both are defined as stages that
    follow the frozen reranker, not a replacement for it). Never raises.
    Performs no parser or candidate-generation changes -- called from
    cs_annotator_app.py's run_pipeline() on every annotation request,
    right after Annotator.annotate() and before
    _ensure_matrix_embed_consistency().
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


# ===========================================================================
# formerly uid_resolver.py -- UID->TR resolver (production-integrated via
# apply_reranker() above; apply_uid_to_tr_resolver() below remains a
# standalone/offline-evaluation entry point).
# ===========================================================================

"""EXPERIMENTAL, offline-only UID->TR resolver.

*** NOT wired into production. Not imported by cs_annotator_app.py or
*** reranker_integration.py. Do not import this module from either of
*** those files without explicit maintainer authorization -- see
*** CLAUDE.md section 2/3 and the offline evaluation report this module
*** was built for.

Motivation / required conceptual placement
--------------------------------------------------------------------------
An earlier Turkish-stem fallback (folded directly into
Annotator._choose_label) was implemented and evaluated, then reverted: it
ran too early, ahead of MIXED detection, and pre-empted genuine MIXED/UID
tokens by converting them to TR before the reranker or residual-verbal
stage ever saw them (see CHANGELOG.md). This module is a from-scratch
replacement designed specifically to avoid that failure mode: it is meant
to be invoked, in an offline evaluation harness ONLY, strictly after every
existing production label-assignment stage has already run --

    Annotator.annotate()                              (cs_pipeline.py)
    -> NE Policies C/D                                 (cs_pipeline.py, inside annotate())
    -> frozen Phase 5F MIXED reranker                   (reranker_integration.apply_reranker)
    -> residual verbal MIXED detector                   (reranker_integration.apply_reranker,
                                                           mixed_reranker.evaluate_residual_verbal_promotion)
    -> [THIS MODULE] UID -> TR resolver, offline only
    -> Matrix/Embedded Language recomputation            (only if this stage promoted anything)

apply_uid_to_tr_resolver() below inspects ONLY token rows whose CURRENT
label (i.e. after every stage above has already run) is exactly "UID". It
never looks at, and never overwrites, TR/EN/MIXED/NE/OTHER/LANG3, and never
touches SentenceID. It only ever promotes UID -> TR -- there is no UID->EN
path in this module (English is already largely covered by the existing
lexicon/reranker machinery, and the EN direction carries more misclassification
risk per the brief this module was built from).

Scope and safety design
--------------------------------------------------------------------------
A UID token is only eligible to be considered at all if it survives a set
of hard exclusion gates (check_eligibility) -- URLs/mentions/hashtags/
numbers/punctuation/emoji/alphanumeric codes (reusing cs_pipeline.
is_other_token unmodified), email-shaped tokens, apostrophe-bearing tokens,
all-caps acronyms, alphanumeric identifiers, tokens that look like a
proper name (capitalized -- the same heuristic mixed_reranker's residual
verbal stage already uses for its own proper-name guard), a direct English
lexicon hit on the whole token, and -- most importantly -- any token for
which mixed_reranker.enumerate_candidate_analyses finds an
English-root-plus-Turkish-suffix analysis (i.e. anything the frozen
reranker/residual stage would treat as a MIXED candidate). That last gate
is what keeps this module from ever contesting a genuine MIXED token.

An eligible token is then scored using extract_evidence()/score_evidence():
an additive, explainable point system over independent signals (trusted
Turkish lexicon evidence for the whole token or a recovered nominal stem,
a valid Turkish suffix-chain analysis via the existing, unmodified
Annotator._has_valid_turkish_nominal_analysis, strong Turkish fastText
support, and Turkish-specific orthographic evidence), plus a capped,
non-decisive auxiliary bonus for sentence-level MatrixLang=TR agreement.
No single signal, and no pair of the weaker signals alone, can reach the
promotion threshold -- see PROMOTION_THRESHOLD/MIN_PRIMARY_SIGNALS and the
comments beside them for the exact arithmetic and its rationale. Anything
that does not clear the threshold is left exactly as UID.

Architecture
--------------------------------------------------------------------------
Every function here is pure (no I/O, no global state) except
apply_uid_to_tr_resolver(), which operates on the same blank-line-delimited
"Item\\tLabel" text format Annotator.annotate()/apply_reranker() already
use, and returns that same format back -- no schema change. This module
reuses, read-only, existing tested logic rather than duplicating it:
cs_pipeline.is_other_token, mixed_reranker.enumerate_candidate_analyses /
is_non_turkish_stem_evidence / fasttext_predict_raw, and
reranker_integration._parse_block_lines (the same block-parsing convention
apply_reranker() already uses, so this module's block handling cannot
silently drift from it). Annotator._has_valid_turkish_nominal_analysis and
Annotator._decide_matrix_embed are called read-only, exactly like
reranker_integration.py already does. Nothing here retrains, rethresholds,
or modifies the frozen Phase 5F reranker or its resources/models/ files.

Callable standalone (e.g. from a throwaway offline evaluation script)
without opening the GUI. cs_pipeline is only ever imported lazily, inside
the one function that needs is_other_token (mirroring mixed_reranker.py's
own local-import convention for the same helper), so importing this module
does not force cs_pipeline's module-level `import stanza` unless/until a
caller actually needs it.
"""


# ---------------------------------------------------------------------------
# Hard exclusion gates
# ---------------------------------------------------------------------------

MIN_TOKEN_LEN = 2

# A conservative, ASCII-safe email shape: local@domain.tld, no embedded
# whitespace or '@'. Distinct from cs_pipeline.MENTION_RE ("@\w+"), which
# matches a bare "@handle" but not a full address.
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
    """Heuristic guard for "known/protected named entity": this stage never
    sees NER output directly (it only sees an already-final UID label), so
    it mirrors the same capitalization heuristic mixed_reranker's residual
    verbal stage already uses for its own proper-name guard
    (_residual_verbal_looks_like_proper_name_or_noise) rather than
    inventing a different rule."""
    return token[:1].isupper()


def _has_strong_direct_english_match(token_l: str, annotator) -> bool:
    return token_l in annotator.english_freq_words


def _has_english_root_turkish_suffix_analysis(token: str, annotator, cfg) -> bool:
    """True if ANY enumerated stem/suffix split of `token` (nominal or
    verbal, at the broadest available verbal level so this safety gate is
    maximally cautious) has a stem carrying non-Turkish/English evidence --
    i.e. exactly the shape mixed_reranker's own candidate generation would
    treat as a MIXED candidate. A token that matches this must never be
    touched by this resolver; it is left for the existing MIXED-detection
    stages, never for this stage to pre-empt them.
    """
    candidates = enumerate_candidate_analyses(
        token, annotator, verbal_level=VERBAL_MORPHOLOGY_PHASE_4E)
    return any(is_non_turkish_stem_evidence(annotator, c.stem, cfg) for c in candidates)


def check_eligibility(token: str, label: str, annotator, cfg) -> Tuple[bool, str]:
    """Whether `token` (currently labeled `label`) may even be considered
    by this resolver. Returns (is_eligible, reason) -- reason is always
    populated, including on success ("eligible"), so callers/logging never
    have to special-case the True branch.
    """
    if label != "UID":
        return False, "label_not_uid"
    if not token or len(token) < MIN_TOKEN_LEN:
        return False, "too_short"

    from cs_pipeline import is_other_token  # local import, see module docstring
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


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------

# Primary signals: independent, each individually insufficient. Weights are
# deliberately set so that no single signal, and no bare pair of the
# weakest signals, crosses PROMOTION_THRESHOLD on its own -- see the
# arithmetic note beside PROMOTION_THRESHOLD below.
PRIMARY_SIGNAL_WEIGHTS: Dict[str, int] = {
    "trusted_lexicon": 3,
    "suffix_chain": 3,
    "fasttext_strong": 3,
    "orthographic": 2,
}
PRIMARY_SIGNALS = frozenset(PRIMARY_SIGNAL_WEIGHTS)

# Auxiliary signal: sentence-level MatrixLang agreement. Explicitly weak --
# never counted toward MIN_PRIMARY_SIGNALS, and its weight alone can never
# bridge more than one missing point of PROMOTION_THRESHOLD.
AUXILIARY_SIGNAL_WEIGHTS: Dict[str, int] = {
    "matrix_language": 1,
}

MIN_PRIMARY_SIGNALS = 2

# Two primary signals alone = 3+3 = 6 (or 3+2 = 5), always short of this
# threshold -- promotion requires EITHER three primary signals (>= 8) OR
# two primary signals plus the auxiliary MatrixLang bonus (== 7, only when
# the two chosen signals are each worth >= 3). This is what "at least two
# independent positive signals" (the evaluation-protocol floor) plus "a
# deliberately conservative threshold" (more than that floor in practice)
# means concretely in this module.
PROMOTION_THRESHOLD = 7

TURKISH_ORTHOGRAPHIC_CHARS = set("çÇğĞıİöÖşŞüÜ")


@dataclass(frozen=True)
class EvidenceItem:
    signal: str
    weight: int
    description: str


def _recovered_turkish_stem(token: str, annotator) -> Optional[str]:
    """The stem of the first NOMINAL candidate analysis (mixed_reranker.
    enumerate_candidate_analyses, source == "nominal" only -- this signal
    is about a closed-class Turkish suffix chain, not the experimental
    verbal tables) whose stem is itself confirmed in the trusted Turkish
    lexicon. Returns None if no such split exists. Does not modify the
    token; returns the original-case stem substring for explanation
    purposes only.
    """
    for candidate in enumerate_candidate_analyses(token, annotator):
        if candidate.source != "nominal":
            continue
        stem_l = candidate.stem.lower()
        if stem_l in annotator.turkish_freq_all or stem_l in annotator.turkish_freq_top:
            return candidate.stem
    return None


def extract_evidence(token: str, annotator, cfg, matrix_lang: Optional[str] = None) -> List[EvidenceItem]:
    """Build the full, explainable evidence list for `token`. Does not
    check eligibility -- callers must call check_eligibility() first (see
    decide(), which does exactly that).
    """
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


# ---------------------------------------------------------------------------
# Decision + explanation
# ---------------------------------------------------------------------------

class ResolverDecision(NamedTuple):
    token: str
    current_label: str
    proposed_label: str
    promote: bool
    score: int
    evidence: Tuple[EvidenceItem, ...]
    reason: str


def decide(token: str, label: str, annotator, cfg, matrix_lang: Optional[str] = None) -> ResolverDecision:
    """The single entry point combining eligibility, evidence, scoring, and
    the promotion decision for one token. Deterministic: same inputs always
    produce the same ResolverDecision.
    """
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
    """Structured, human-readable explanation in the fixed shape:
    Token / Current / Proposed / Score / Evidence / Decision.
    """
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


# ---------------------------------------------------------------------------
# Applying the resolver to already-annotated blocks (offline use only)
# ---------------------------------------------------------------------------

# This stage only ever inspects token rows CURRENTLY labeled UID -- never
# NE/OTHER/LANG3/TR/EN/MIXED. Checked here, before decide() is even called,
# so those rows never appear in the returned `decisions` list either.
_RESOLVER_ELIGIBLE_LABELS = frozenset({"UID"})


def apply_uid_to_tr_resolver(annotated_text: str, annotator, cfg: dict) -> Tuple[str, List[ResolverDecision]]:
    """Offline-only post-processing stage: promotes eligible UID token rows
    to TR (see decide()) in text already produced by
    Annotator.annotate() -> apply_reranker() (frozen reranker + residual
    verbal stage). Input/output format is identical to that pipeline's own
    text output -- no schema change. Blocks with no promotion are returned
    byte-for-byte identical to the input; MatrixLang/EmbedLang are
    recomputed (via the unmodified Annotator._decide_matrix_embed) only for
    blocks where a promotion actually occurred.

    Never raises: a single token's evaluation failure is caught and treated
    as "no promotion" for that token, mirroring apply_reranker()'s own
    fail-safe contract, so one bad token cannot interrupt the rest of the
    block.

    Returns (new_text, decisions) -- `decisions` lists every ResolverDecision
    actually made (i.e. every UID token inspected), in encounter order, for
    offline logging/evaluation. Production integration (reranker_integration.
    apply_reranker()) calls decide() directly, per token, inline in its own
    block loop -- it never calls this function; this function exists purely
    as a standalone/offline-evaluation convenience, which is also why the
    reranker_integration import below is local rather than module-level
    (see the note beside this module's imports at the top of the file).
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
