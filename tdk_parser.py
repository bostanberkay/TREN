# tdk_parser.py
"""Hierarchical Turkish morphological "parser" for the TDK Checker tool:
given an arbitrary token, proposes a root/lemma, an ordered list of
suffix segments, a part-of-speech-aware category, and a human-readable
explanation of every proposed boundary -- for display and (by the user)
manual correction in the TDK Checker window.

Why not flat character-by-character suffix stripping
--------------------------------------------------------------------------
The previous version of this module picked among candidate stem/suffix
splits using `mixed_reranker.select_best_analysis`'s
"most suffix segments" strategy -- correct for that module's own purpose
(maximizing MIXED-candidate evidence), but wrong here: it would pick
"fil" + "m" + "in" over "film" + "in" for "filmin" (more segments, but
the wrong root), and independently, calling `mixed_reranker.
enumerate_candidate_analyses` at its DEFAULT verbal level (Phase 4C-1)
never even considers past/evidential/progressive/future tense at all
(those are Phase 4B/4E additions), so "sürdü" was never recognized as
"sür" + "dü" (an atomic past-tense suffix) and fell through to a nominal
single-character accusative peel ("sürd" + "ü") instead.

This module fixes both problems with its own, purpose-built ranking and,
for verb tense/aspect/mood specifically, its own STRUCTURED suffix table
(`VERB_TAM_ENTRIES`/`VERB_PERSON_ENTRIES` below) -- atomic, full
vowel-harmony-variant surface forms (e.g. "dü" is one past-tense suffix,
never decomposed into "d" + "ü"; "-Iyor" is represented with all four
harmony variants, not just the bare consonant-final form
`mixed_reranker`'s own experimental table happens to cover). Nominal
morphology (plural/possessive/case) is NOT reimplemented -- it is already
correct and already hierarchical in `cs_pipeline`'s existing tables
(peeled case -> possessive -> plural -> derivational from the right,
which read left-to-right IS "root -> derivational -> plural -> possessive
-> case"); this module reuses that read-only via
`mixed_reranker.enumerate_candidate_analyses` and only replaces the
SELECTION policy among the candidates it enumerates.

Selection policy (see `_score_candidate`)
--------------------------------------------------------------------------
There is deliberately NO hard "if the whole token is in the lexicon, never
split it" rule. An earlier version of this module had exactly that as a
first-priority short-circuit, and it was wrong: `resources/frequent_tr_
words.txt` is a CORPUS FREQUENCY list, not a lemma list, so it contains
huge numbers of independently-attested INFLECTED surface forms too (e.g.
"geldi", "sürdü", "filmin" all appear in it in their own right). A
membership-only check therefore blocked the split for exactly the
inflected forms this parser most needs to split.

Instead, "the whole token, unsplit" is generated as one more candidate
and competes on equal footing, in the SAME additively-scored pool, against
every split candidate (from `mixed_reranker.enumerate_candidate_analyses`,
nominal + its own experimental verbal fallback, PLUS this module's own
structured verb TAM/person table):
     - stem attested in the top-1000-frequency Turkish lexicon: strong
       bonus (a real, common root -- not merely an attested SURFACE FORM;
       frequency word lists contain inflected forms too, e.g. "filmi",
       so raw lexicon membership alone is not enough to prefer "film"
       over "filmi" -- frequency RANK is what actually disambiguates it).
     - stem attested anywhere in the full Turkish lexicon: smaller bonus.
     - stem attested in the English lexicon (and not Turkish): a
       comparable bonus, for genuine English-root-plus-Turkish-suffix
       tokens (e.g. "cloudumuza").
     - this module's own structured verb match: a structural bonus
       (well-formed, atomic tense/aspect/mood +/- person suffix is
       inherently stronger evidence than an arbitrary leftover
       character), stacked with mixed_reranker's own "verbal"-source
       candidates at a smaller bonus, and with a comparable (smaller)
       bonus for a nominal split whose EVERY segment is a recognized
       Turkish case/possessive/plural/derivational suffix -- this is
       what lets a genuine split ("ev" + "ler") outscore an unsplit
       reading of a merely-frequent inflected surface form ("evler").
       The unsplit "whole token" candidate itself never receives this
       bonus, since it makes no structural claim at all.
     - longer stem: a modest bonus (the tie-break within an otherwise
       equally-well-evidenced group).
     - each single-character segment: a penalty (never an outright
       rejection -- some single-character Turkish suffixes are entirely
       legitimate, e.g. dative "-e" in "ev-e", and remain selectable when
       no better-evidenced alternative exists).
     - vowel-harmony agreement between the stem's last vowel and the
       first proposed segment's first vowel: a small bonus/penalty (soft,
       not a hard rejection, since documented loanword exceptions to
       vowel harmony exist in Turkish, e.g. "saat-ler").
   The highest-scoring candidate wins; ties break on earliest split
   position for determinism. If the winning candidate is the unsplit
   "whole token" one, the result category depends on whether it is itself
   lexicon-confirmed (`full_turkish_lexical_item`) or not (an unconfirmed,
   unanalyzable token falls back to `invalid_parser_proposal`).

This module never decides a label. It never touches `cs_pipeline.py` or
`mixed_reranker.py`. Nothing here is a statistically calibrated model --
every score is a deterministic, explainable point total, exactly like
every other evidence layer in this codebase.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cs_pipeline as cp
import mixed_reranker as mr

MIN_ROOT_LEN = 2

# mixed_reranker's own "verbal" fallback table is a recall-oriented,
# non-atomic candidate generator built for MIXED-evidence purposes (see
# module docstring) -- it will happily propose 2-character coincidental
# stems (e.g. "ka" + "le" + "m" for "kalem", a real base noun with no
# suffix at all). This module's own structured VERB_TAM_ENTRIES table
# already covers every required atomic tense/aspect/mood form (down to a
# 2-character root, e.g. "gel" is already 3, but the table itself allows
# MIN_ROOT_LEN); the looser, non-atomic "verbal" fallback is trusted only
# for stems of 3+ characters, to keep it from out-scoring a genuine
# unsplit base word on a spurious short coincidental match.
MR_VERBAL_MIN_ROOT_LEN = 3

# ---------------------------------------------------------------------------
# Vowel harmony (soft scoring signal only -- see module docstring)
# ---------------------------------------------------------------------------

FRONT_VOWELS = set("eiöü")
BACK_VOWELS = set("aıou")
ROUNDED_VOWELS = set("oöuü")
UNROUNDED_VOWELS = set("aeıi")
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS


def _last_vowel(s: str) -> Optional[str]:
    return next((c for c in reversed(s) if c in ALL_VOWELS), None)


def _first_vowel(s: str) -> Optional[str]:
    return next((c for c in s if c in ALL_VOWELS), None)


def vowel_harmony_consistent(stem: str, first_segment: str) -> Optional[bool]:
    """Two-way (front/back) vowel harmony agreement between `stem`'s last
    vowel and `first_segment`'s first vowel. Returns None (indeterminate,
    never penalized) when either side has no vowel to compare."""
    sv, fv = _last_vowel(stem), _first_vowel(first_segment)
    if sv is None or fv is None:
        return None
    return (sv in FRONT_VOWELS) == (fv in FRONT_VOWELS)


# ---------------------------------------------------------------------------
# Structured suffix entries -- tdk_parser's OWN table, additional to (never
# replacing) mixed_reranker.py's tables. Covers verb tense/aspect/mood with
# full vowel-harmony surface-form variants, atomically (never split further).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuffixEntry:
    underlying_form: str
    category: str
    part_of_speech: str
    surface_forms: Tuple[str, ...]
    requires_vowel_harmony: bool
    allows_consonant_alternation: bool
    valid_predecessor_categories: Tuple[str, ...]
    gloss: str


VERB_TAM_ENTRIES: Tuple[SuffixEntry, ...] = (
    SuffixEntry("-DI", "past_definite", "verb",
                ("dı", "di", "du", "dü", "tı", "ti", "tu", "tü"),
                True, True, ("root",), "past tense (-DI)"),
    SuffixEntry("-mIş", "evidential_past", "verb",
                ("mış", "miş", "muş", "müş"),
                True, False, ("root",), "evidential/reported past tense (-mIş)"),
    SuffixEntry("-Iyor", "progressive", "verb",
                ("ıyor", "iyor", "uyor", "üyor"),
                True, False, ("root",), "present progressive (-Iyor)"),
    SuffixEntry("-(y)AcAk", "future", "verb",
                ("acak", "ecek", "yacak", "yecek"),
                True, False, ("root",), "future tense (-(y)AcAk)"),
)

VERB_PERSON_ENTRIES: Tuple[SuffixEntry, ...] = (
    SuffixEntry("-m", "first_singular", "verb", ("m",), False, False,
                ("past_definite", "evidential_past"), "1st person singular agreement"),
    SuffixEntry("-n", "second_singular", "verb", ("n",), False, False,
                ("past_definite", "evidential_past"), "2nd person singular agreement"),
    SuffixEntry("-k", "first_plural", "verb", ("k",), False, False,
                ("past_definite", "evidential_past"), "1st person plural agreement"),
    SuffixEntry("-(s)InIz", "second_plural", "verb",
                ("sınız", "siniz", "sunuz", "sünüz"), True, False,
                ("past_definite", "evidential_past", "progressive", "future"), "2nd person plural agreement"),
    SuffixEntry("-lAr", "third_plural", "verb", ("lar", "ler"), True, False,
                ("past_definite", "evidential_past", "progressive", "future"), "3rd person plural agreement"),
)
# Third person singular is zero-marked in Turkish -- deliberately no entry:
# a bare TAM suffix with nothing following it already IS the complete 3sg
# form (e.g. "sür-dü" = "(s/he/it) surfed", no additional suffix exists to
# invent). See module docstring.


def _enumerate_structured_verb_candidates(token_l: str) -> List[Tuple[str, Tuple[str, ...], Tuple[SuffixEntry, ...]]]:
    """Every (stem, segments, suffix_entries) triple obtainable by matching
    an atomic VERB_TAM_ENTRIES form, optionally followed by an atomic
    VERB_PERSON_ENTRIES form whose `valid_predecessor_categories` allows
    it, with FULL consumption of the remainder. Never matches a bare
    single character as if it were a tense suffix -- every TAM surface
    form is 2+ characters by construction."""
    out = []
    n = len(token_l)
    for split in range(MIN_ROOT_LEN, n):
        stem, rest = token_l[:split], token_l[split:]
        for tam in VERB_TAM_ENTRIES:
            for form in tam.surface_forms:
                if rest == form:
                    out.append((stem, (form,), (tam,)))
                elif rest.startswith(form):
                    remainder = rest[len(form):]
                    for person in VERB_PERSON_ENTRIES:
                        if tam.category not in person.valid_predecessor_categories:
                            continue
                        if remainder in person.surface_forms:
                            out.append((stem, (form, remainder), (tam, person)))
    return out


# ---------------------------------------------------------------------------
# Nominal segment classification (read-only reuse of cs_pipeline's existing
# suffix tables, purely for explanatory labeling -- no new nominal suffix
# invented, no change to which splits are considered valid).
# ---------------------------------------------------------------------------

def _classify_nominal_segment(segment: str) -> Tuple[str, str]:
    seg_l = segment.lower()
    if seg_l in cp.BUFFER_N_ACC:
        return cp.BUFFER_N_ACC[seg_l], "buffer-n accusative allomorph"
    if seg_l in cp.BUFFER_N_DAT:
        return cp.BUFFER_N_DAT[seg_l], "buffer-n dative allomorph"
    if seg_l in cp.CASE_ENDINGS:
        return cp.CASE_ENDINGS[seg_l], "Turkish case suffix"
    if seg_l in cp.POSS_LONG:
        return "+".join(cp.POSS_LONG[seg_l]), "Turkish possessive suffix (plural possessor)"
    if seg_l in cp.POSS_SHORT:
        return "+".join(cp.POSS_SHORT[seg_l]), "Turkish possessive suffix"
    if seg_l in cp.PLUR:
        return cp.PLUR[seg_l], "Turkish plural suffix"
    if seg_l in cp.DERIV_SUFFIXES:
        return "+".join(cp.DERIV_SUFFIXES[seg_l]), "Turkish derivational suffix"
    if seg_l in ("ı", "i", "u", "ü"):
        return "Amb=P3sg_or_Acc", "ambiguous 3rd-person-possessive/accusative suffix"
    return "unrecognized", "unrecognized segment"


# ---------------------------------------------------------------------------
# Unified candidate model + scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentExplanation:
    segment: str
    category: str
    valid: bool
    rule: str

    def to_dict(self) -> dict:
        return {"segment": self.segment, "category": self.category, "valid": self.valid, "rule": self.rule}


# When the winning split's score beats the unsplit "whole token" reading
# by less than this margin, AND the unsplit reading is itself genuinely
# lexicon-attested, the two readings are too close to call with confidence
# from corpus-frequency evidence alone (e.g. "kalem" [a base noun, "pen"]
# vs "kale"+"m" [fortress + 1sg possessive] -- both plausible, TDK lookup
# needed to resolve). See parse_token(). Chosen well below the smallest
# margin observed for a genuinely correct required split (kitaplarda's
# ~140) and comfortably above the "kalem" case's ~30.
AMBIGUITY_MARGIN = 50.0

CATEGORY_FULL_LEXICAL = "full_turkish_lexical_item"
CATEGORY_ENGLISH_ROOT = "english_root_turkish_suffix"
CATEGORY_AMBIGUOUS = "ambiguous_candidate"
CATEGORY_INVALID = "invalid_parser_proposal"
CATEGORY_MANUAL = "manual_correction"

_POS_VERB = "verb"
_POS_NOUN = "noun"
_POS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Candidate:
    stem: str
    segments: Tuple[str, ...]
    source: str  # "verb_structured" | "verbal" | "nominal" | "full_token"
    stem_in_top: bool
    stem_in_all: bool
    stem_in_english: bool
    harmony: Optional[bool]
    explanations: Tuple[SegmentExplanation, ...]
    all_segments_valid: bool = False


def _score_candidate(c: _Candidate) -> Tuple[float, int]:
    score = 0.0
    if c.stem_in_top:
        score += 1000.0
    elif c.stem_in_all:
        score += 500.0
    if c.stem_in_english and not c.stem_in_all and not c.stem_in_top:
        score += 400.0
    if c.source == "verb_structured":
        score += 300.0
    elif c.source == "verbal":
        score += 200.0
    elif c.source == "nominal":
        # Per-segment (not flat) credit: at an equally-attested lexicon
        # tier, a stem like "kitaplar" (itself an attested surface form,
        # but really "kitap" + plural, not a root) must not out-score the
        # fuller, genuinely-minimal decomposition "kitap" + "lar" + "da"
        # just because it happens to be a few characters longer -- see
        # module docstring.
        score += sum(90.0 for e in c.explanations if e.valid)
    score += len(c.stem) * 10.0
    score -= sum(1 for s in c.segments if len(s) == 1) * 50.0
    if c.harmony is True:
        score += 10.0
    elif c.harmony is False:
        score -= 5.0
    # tie-break: earliest split position (longer stem already rewarded
    # above; this only matters for genuine ties) -- returned separately so
    # callers can sort deterministically without relying on Python's
    # stable-sort behavior across equal-score floats from unrelated inputs.
    return score, len(c.stem)


def _lexicon_flags(stem: str, annotator) -> Tuple[bool, bool, bool]:
    stem_l = stem.lower()
    top = stem_l in getattr(annotator, "turkish_freq_top", ())
    allw = top or stem_l in getattr(annotator, "turkish_freq_all", ())
    eng = stem_l in getattr(annotator, "english_freq_words", ())
    return top, allw, eng


def _build_candidates(token: str, annotator) -> List[_Candidate]:
    token_l = token.lower()
    candidates: List[_Candidate] = []

    for stem, segments, entries in _enumerate_structured_verb_candidates(token_l):
        if len(stem) < MIN_ROOT_LEN:
            continue
        top, allw, eng = _lexicon_flags(stem, annotator)
        harmony = vowel_harmony_consistent(stem, segments[0]) if segments else None
        explanations = tuple(
            SegmentExplanation(seg, entry.category, True,
                                f"structured Turkish verb suffix ({entry.underlying_form}): {entry.gloss}")
            for seg, entry in zip(segments, entries)
        )
        candidates.append(_Candidate(stem, segments, "verb_structured", top, allw, eng, harmony,
                                      explanations, all_segments_valid=True))

    try:
        mr_candidates = mr.enumerate_candidate_analyses(token, annotator, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    except Exception:
        mr_candidates = []
    for cand in mr_candidates:
        stem = cand.stem
        min_len = MR_VERBAL_MIN_ROOT_LEN if cand.source == "verbal" else MIN_ROOT_LEN
        if len(stem) < min_len:
            continue
        top, allw, eng = _lexicon_flags(stem, annotator)
        segments = tuple(cand.segments)
        harmony = vowel_harmony_consistent(stem, segments[0]) if segments else None
        if cand.source == "verbal":
            explanations = tuple(
                SegmentExplanation(seg, "Verbal", True,
                                    "Turkish verbal suffix (experimental morphology table)")
                for seg in segments
            )
            all_valid = True
        else:
            explanations = _explanations_for_nominal(segments)
            all_valid = bool(segments) and all(e.valid for e in explanations)
        candidates.append(_Candidate(stem, segments, cand.source, top, allw, eng, harmony,
                                      explanations, all_segments_valid=all_valid))

    # "Whole token, unsplit" always competes as one more candidate in the
    # SAME scored pool -- see module docstring for why this replaced a
    # hard priority-0 lexicon-membership short-circuit.
    top, allw, eng = _lexicon_flags(token, annotator)
    candidates.append(_Candidate(token, (), "full_token", top, allw, eng, None, ()))

    return candidates


def _explanations_for_nominal(segments: Tuple[str, ...]) -> Tuple[SegmentExplanation, ...]:
    out = []
    for seg in segments:
        category, rule = _classify_nominal_segment(seg)
        out.append(SegmentExplanation(seg, category, category != "unrecognized", rule))
    return tuple(out)


def _select_best(candidates: List[_Candidate]) -> Optional[_Candidate]:
    if not candidates:
        return None
    scored = [(_score_candidate(c), c) for c in candidates]
    scored.sort(key=lambda pair: (pair[0][0], pair[0][1]), reverse=True)
    return scored[0][1]


def _categorize(candidate: Optional[_Candidate]) -> Tuple[str, str]:
    """Returns (category, part_of_speech) for the winning candidate."""
    if candidate is None:
        return CATEGORY_INVALID, _POS_UNKNOWN
    if candidate.source in ("verb_structured", "verbal"):
        pos = _POS_VERB
    elif candidate.source == "nominal":
        pos = _POS_NOUN
    else:
        pos = _POS_UNKNOWN
    if candidate.stem_in_top or candidate.stem_in_all:
        return CATEGORY_FULL_LEXICAL, pos
    if candidate.stem_in_english:
        return CATEGORY_ENGLISH_ROOT, pos
    if candidate.segments:
        return CATEGORY_AMBIGUOUS, pos
    return CATEGORY_INVALID, pos


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseResult:
    token: str
    root: str
    segments: Tuple[str, ...]
    success: bool
    source: str  # "full_lexical_item" | "verb_structured" | "verbal" | "nominal" | "manual" | "whole_token_fallback"
    category: str  # CATEGORY_* constant
    part_of_speech: str  # "verb" | "noun" | "unknown"
    reason: str
    segment_explanations: Tuple[SegmentExplanation, ...] = ()
    harmony_consistent: Optional[bool] = None
    stem_in_turkish_lexicon: bool = False
    stem_in_english_lexicon: bool = False

    @property
    def suffix(self) -> str:
        return "".join(self.segments)

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "root": self.root,
            "segments": list(self.segments),
            "success": self.success,
            "source": self.source,
            "category": self.category,
            "part_of_speech": self.part_of_speech,
            "reason": self.reason,
            "segment_explanations": [e.to_dict() for e in self.segment_explanations],
            "harmony_consistent": self.harmony_consistent,
            "stem_in_turkish_lexicon": self.stem_in_turkish_lexicon,
            "stem_in_english_lexicon": self.stem_in_english_lexicon,
        }


def _whole_token_fallback(token: str, reason: str) -> ParseResult:
    return ParseResult(token=token, root=token, segments=(), success=False,
                        source="whole_token_fallback", category=CATEGORY_INVALID,
                        part_of_speech=_POS_UNKNOWN, reason=reason)


def parse_token(token: str, annotator) -> ParseResult:
    """Deterministic, hierarchical Turkish morphological analysis of
    `token` -- see module docstring for the full selection policy. Never
    raises: any failure in the underlying analysis machinery degrades to
    a whole-token fallback. Never assigns a language label; that remains
    entirely manual (see cs_annotator_app.py's TDK Checker integration).
    """
    if not token or not str(token).strip():
        return _whole_token_fallback(token or "", "empty token")

    token_l = token.lower()

    try:
        candidates = _build_candidates(token, annotator)
    except Exception as e:
        return _whole_token_fallback(token, f"parser error: {e}")

    # _build_candidates always appends a "whole token, unsplit" candidate,
    # so this pool is never empty and `best` is never None here.
    best = _select_best(candidates)

    reconstructed = f"{best.stem}{''.join(best.segments)}"
    if reconstructed.lower() != token_l:
        return _whole_token_fallback(token, "malformed segmentation: stem+suffix does not reconstruct token")

    category, pos = _categorize(best)
    lexicon_confirmed = best.stem_in_top or best.stem_in_all or best.stem_in_english

    if not best.segments and not lexicon_confirmed:
        # The winning candidate is the unsplit whole token AND it is not
        # itself confirmed by any lexicon -- an unanalyzable token, not a
        # confident result. Report it the same way a truly empty candidate
        # pool always has (whole_token_fallback / success False).
        return _whole_token_fallback(token, "no valid Turkish suffix-chain split found")

    source = "full_lexical_item" if (best.source == "full_token" and not best.segments) else best.source
    n_segments = len(best.segments)
    ambiguous_margin = False
    if best.segments and category in (CATEGORY_FULL_LEXICAL, CATEGORY_ENGLISH_ROOT):
        full_tok = next((c for c in candidates if c.source == "full_token"), None)
        if full_tok is not None and (full_tok.stem_in_top or full_tok.stem_in_all or full_tok.stem_in_english):
            gap = _score_candidate(best)[0] - _score_candidate(full_tok)[0]
            if gap < AMBIGUITY_MARGIN:
                ambiguous_margin = True
                category = CATEGORY_AMBIGUOUS

    if ambiguous_margin:
        reason = (
            f"{n_segments} suffix segment(s) found via "
            f"{'structured verb morphology' if best.source == 'verb_structured' else best.source + ' analysis'}"
            f", but the unsplit token \"{token}\" is itself also a plausible lexical item -- "
            "ambiguous without a dictionary lookup"
        )
    elif n_segments:
        reason = (
            f"{n_segments} suffix segment(s) found via "
            f"{'structured verb morphology' if best.source == 'verb_structured' else best.source + ' analysis'}"
        )
    else:
        reason = "the complete token is itself a known Turkish lexical item; no suffix split proposed"
    return ParseResult(
        token=token, root=best.stem, segments=best.segments, success=True, source=source,
        category=category, part_of_speech=pos, reason=reason,
        segment_explanations=best.explanations, harmony_consistent=best.harmony,
        stem_in_turkish_lexicon=(best.stem_in_top or best.stem_in_all),
        stem_in_english_lexicon=best.stem_in_english,
    )


# ---------------------------------------------------------------------------
# Manual correction (user-edited root/segments) -- always category
# CATEGORY_MANUAL / source "manual"; never re-validated against the
# automatic ranking policy above (the user's explicit correction always
# takes precedence for DISPLAY purposes; `success` only reflects whether
# it reconstructs the token, never whether the automatic parser "agrees").
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[+\-\s]+")


def reparse_with_manual_root(token: str, root: str) -> ParseResult:
    if not token:
        return ParseResult(token=token or "", root=root or "", segments=(), success=False,
                            source="manual", category=CATEGORY_INVALID, part_of_speech=_POS_UNKNOWN,
                            reason="empty token")
    if not root or not token.lower().startswith(root.lower()):
        return ParseResult(token=token, root=token, segments=(), success=False,
                            source="manual", category=CATEGORY_INVALID, part_of_speech=_POS_UNKNOWN,
                            reason="root is not a prefix of the token")
    remainder = token[len(root):]
    segments = (remainder,) if remainder else ()
    return ParseResult(token=token, root=root, segments=segments, success=True,
                        source="manual", category=CATEGORY_MANUAL, part_of_speech=_POS_UNKNOWN,
                        reason="manually corrected")


def segments_from_text(token: str, root: str, segments_text: str) -> ParseResult:
    root = (root or "").strip()
    parts = [p.strip() for p in _SPLIT_RE.split(segments_text or "") if p.strip()]
    segments = tuple(parts)
    reconstructed = f"{root}{''.join(segments)}"
    ok = bool(root) and reconstructed.lower() == (token or "").lower()
    return ParseResult(
        token=token or "", root=root, segments=segments, success=ok,
        source="manual", category=(CATEGORY_MANUAL if ok else CATEGORY_INVALID), part_of_speech=_POS_UNKNOWN,
        reason="manually corrected" if ok else "root + segments do not reconstruct the token",
    )


# ---------------------------------------------------------------------------
# Lightweight lexicon loading for the parser only -- NOT the full
# cs_pipeline.Annotator() (which also loads fastText and requires Stanza
# availability). Reading two plain word-list files is local, fast
# (well under a second), and fully offline -- this is all parse_token
# ever needs (turkish_freq_top/_all, english_freq_words).
# ---------------------------------------------------------------------------

def load_lexicon_annotator(freq_tr: str = "frequent_tr_words.txt",
                            freq_en: str = "frequent_en_words.txt"):
    from cs_pipeline import Annotator
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set()
    obj.turkish_freq_all = set()
    obj.english_freq_words = set()
    obj.ner = None
    try:
        with open(freq_tr, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0].lower()
                obj.turkish_freq_all.add(w)
                if i < 1000:
                    obj.turkish_freq_top.add(w)
    except OSError:
        pass
    try:
        with open(freq_en, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    obj.english_freq_words.add(parts[0].lower())
    except OSError:
        pass
    return obj
