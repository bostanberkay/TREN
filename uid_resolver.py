# uid_resolver.py
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

import re
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import mixed_reranker as mr

# NOTE: reranker_integration is NOT imported at module level here, even
# though apply_uid_to_tr_resolver() below reuses its block-parsing
# convention read-only. reranker_integration.py now imports THIS module
# (production integration, see its own docstring) -- a top-level import
# back in this direction would be circular. apply_uid_to_tr_resolver()
# below imports it lazily, inside the function, instead; this is a
# standalone/offline-evaluation entry point only (production calls
# decide() directly, per token, and never calls this function at all), so
# the lazy import has no production-path cost.

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
    candidates = mr.enumerate_candidate_analyses(
        token, annotator, verbal_level=mr.VERBAL_MORPHOLOGY_PHASE_4E)
    return any(mr.is_non_turkish_stem_evidence(annotator, c.stem, cfg) for c in candidates)


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
    for candidate in mr.enumerate_candidate_analyses(token, annotator):
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
    lang, prob = mr.fasttext_predict_raw(annotator, token_l)
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
    import reranker_integration as ri  # local: see note at top of file

    labels_counted_for_matrix_embed = ri._LABELS_COUNTED_FOR_MATRIX_EMBED

    blocks = annotated_text.split("\n\n")
    new_blocks = []
    all_decisions: List[ResolverDecision] = []

    for block in blocks:
        records = ri._parse_block_lines(block)
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
