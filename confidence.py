# confidence.py
"""Production-safe, deterministic confidence and uncertainty layer.

Purpose
--------------------------------------------------------------------------
For every already-labeled token (i.e. AFTER the full, unmodified production
pipeline -- Annotator.annotate() -> reranking.apply_reranker()
[frozen Phase 5F reranker + residual verbal detector + UID->TR resolver] ->
Matrix/Embedded Language consistency -- has already run and decided a final
label), this module computes a per-token confidence record: how much
converging/conflicting evidence supports the label the pipeline already
chose. It NEVER changes a label. It is a purely observational, read-only
second pass over data the production pipeline already produced.

Hard safety guarantees (see CLAUDE.md section 2/3 and the task brief this
module was built from):
  - Never renames/redefines the 7-label schema.
  - Never promotes, demotes, or otherwise mutates any token's label.
  - Never retrains, rethresholds, or modifies the frozen Phase 5F reranker,
    resources/models/*, or any lexicon.
  - Never adds a new NER model -- the only NER call this module ever makes
    (see `compute_entity_types`) reuses the SAME cached Stanza pipeline
    object (`annotator.ner`) the production pipeline already loaded.
  - Never learns from user corrections or persisted "review" history --
    every score is a pure, deterministic function of the token, its labels
    (rule-based and final), and the annotator/cfg/bundle passed in. Running
    this module twice on identical input always produces identical output.

NOT a statistical/calibrated confidence
--------------------------------------------------------------------------
`confidence_score` is a deterministic, additive point score built from
explainable rule-based evidence (lexicon membership, fastText probability,
Turkish suffix-chain analysis, the frozen reranker's own probability/
margin, the residual verbal detector's verdict, the UID->TR resolver's own
evidence score, Matrix/Embedded Language consistency, and hard token-shape
exclusions). It is NOT the output of a model trained/validated against a
held-out gold-labeled confidence dataset, and must never be presented or
reported as a calibrated probability of correctness. Every serialized
record below carries `CALIBRATION_NOTE` verbatim for exactly this reason.
Calibration against a suitable gold corpus is future work (see
docs/roadmap.md-adjacent evaluation report this module was built for).

Evidence sources consulted (read-only reuse of existing, already-tested
production/experimental code -- nothing here duplicates their logic):
  - rule-based label (recovered by diffing the pre-reranker
    Annotator.annotate() output against the final post-pipeline output;
    supplied by the caller, see `attach_confidence_to_blocks`)
  - Turkish/English lexicon membership (`annotator.turkish_freq_top/_all`,
    `annotator.english_freq_words`)
  - fastText language/probability (`reranking.fasttext_predict_raw`,
    itself a thin read-only adapter over `Annotator._ft_predict`)
  - Turkish suffix-chain / morphology evidence
    (`Annotator._has_valid_turkish_nominal_analysis`, read-only)
  - Stanza NE evidence and entity type (best-effort only, gated behind
    `compute_entity_types`; see `_detect_entity_type`)
  - the frozen reranker's own probability/margin, recomputed read-only
    exactly the way `reranking._rerank_token_label` does
    (`reranking.classify_candidate`/`build_structured_feature_dict` +
    `bundle.model.predict_proba`) -- never touches the frozen model itself
  - the residual verbal detector's verdict
    (`reranking.evaluate_residual_verbal_promotion`, read-only)
  - the UID->TR resolver's own explainable evidence/score
    (`reranking.decide`, read-only)
  - MatrixLang/EmbedLang sentence-level consistency
  - token-shape hard exclusions (URLs, mentions, hashtags, numbers, codes,
    emoji, apostrophes -- `cs_pipeline.is_other_token`, imported lazily
    exactly like reranking.py's UID->TR resolver section already does, so importing this module
    does not force `cs_pipeline`'s module-level `import stanza`)

Persistence
--------------------------------------------------------------------------
`attach_confidence_to_blocks` stores one JSON-serializable dict at
`row["confidence"]` per non-meta token row, plus `row["reviewed"]` (a plain
bool, defaulted to False, set True only by the review tool's Apply action).
Both are ordinary extra keys on the same row dicts `.trenproj` already
round-trips verbatim (see `annotation_model.datasets_to_payload`) -- no
schema version bump, no change to TXT/CSV/CoNLL/JSONL export, and a legacy
project loaded without a "confidence" key on its rows is handled
gracefully everywhere in this module (treated as "not yet computed", never
a crash) -- see `get_confidence`/`is_reviewed` below.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import scipy.sparse as sp

import annotation_model
import reranking as mr
import reranking as ur

CALIBRATION_NOTE = (
    "Deterministic, rule-based confidence estimate. NOT statistically "
    "calibrated against a held-out gold-labeled confidence corpus."
)

# ---------------------------------------------------------------------------
# Configurable thresholds (requirement: "Use configurable thresholds").
# HIGH: score >= thresholds["HIGH"]; MEDIUM: thresholds["MEDIUM"] <= score <
# thresholds["HIGH"]; LOW: score < thresholds["MEDIUM"].
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"HIGH": 0.85, "MEDIUM": 0.60}

BAND_HIGH = "HIGH"
BAND_MEDIUM = "MEDIUM"
BAND_LOW = "LOW"

ALL_LABELS = ("TR", "EN", "MIXED", "UID", "NE", "OTHER", "LANG3")

# Reranker-candidate-eligible predicted labels, re-derived from
# reranking's own frozen constants (SCHEMA_LABELS minus
# NON_CANDIDATE_LABELS) rather than duplicating the set by hand -- stays in
# sync automatically if either constant ever changes.
_RERANKER_ELIGIBLE_LABELS = frozenset(mr.SCHEMA_LABELS) - mr.NON_CANDIDATE_LABELS

# Mirrors reranking._RESIDUAL_VERBAL_ELIGIBLE_LABELS /
# _UID_TR_RESOLVER_ELIGIBLE_LABELS (not imported directly, to avoid any
# import-time coupling to that module's private names -- these are the
# same two frozensets, restated here as this module's own read-only
# eligibility gates for evidence *reconstruction*, not label mutation).
_RESIDUAL_VERBAL_ELIGIBLE_LABELS = frozenset({"UID", "TR"})
_UID_TR_RESOLVER_ELIGIBLE_LABELS = frozenset({"UID"})


def band_for_score(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if score >= t["HIGH"]:
        return BAND_HIGH
    if score >= t["MEDIUM"]:
        return BAND_MEDIUM
    return BAND_LOW


@dataclass(frozen=True)
class ConfidenceRecord:
    token: str
    rule_based_label: Optional[str]
    final_label: str
    confidence_score: float
    confidence_band: str
    uncertainty_reasons: Tuple[str, ...]
    evidence_summary: Tuple[str, ...]
    review_recommended: bool
    promoted_by: Optional[str]

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "rule_based_label": self.rule_based_label,
            "final_label": self.final_label,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_band": self.confidence_band,
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "evidence_summary": list(self.evidence_summary),
            "review_recommended": self.review_recommended,
            "promoted_by": self.promoted_by,
            "calibration_note": CALIBRATION_NOTE,
        }


@dataclass(frozen=True)
class _CommonEvidence:
    tr_top: bool
    tr_all: bool
    en_lex: bool
    ft_lang: str
    ft_prob: float
    has_suffix_analysis: bool
    orthographic: bool
    has_apostrophe: bool
    hard_excluded: bool


def _gather_common_evidence(token: str, annotator, cfg) -> _CommonEvidence:
    from cs_pipeline import is_other_token  # local import, see module docstring
    token_l = token.lower()
    ft_lang, ft_prob = mr.fasttext_predict_raw(annotator, token)
    return _CommonEvidence(
        tr_top=token_l in annotator.turkish_freq_top,
        tr_all=token_l in annotator.turkish_freq_all,
        en_lex=token_l in annotator.english_freq_words,
        ft_lang=ft_lang,
        ft_prob=ft_prob,
        has_suffix_analysis=annotator._has_valid_turkish_nominal_analysis(token_l),
        orthographic=any(ch in ur.TURKISH_ORTHOGRAPHIC_CHARS for ch in token),
        has_apostrophe=("'" in token or "’" in token),
        hard_excluded=is_other_token(token),
    )


def _reranker_probability(token: str, rule_label: str, annotator, cfg, bundle) -> Optional[float]:
    """Re-derive the frozen Phase 5F reranker's own probability for `token`
    at its rule-based label, exactly the way
    reranking._rerank_token_label does internally -- read-only,
    never touches the frozen model/threshold. Returns None if `bundle` is
    unavailable, `rule_label` was never candidate-eligible, or any step
    fails (mirrors _rerank_token_label's own fail-safe contract: a failure
    here must never interrupt confidence computation for the rest of the
    token/block)."""
    if bundle is None or rule_label not in _RERANKER_ELIGIBLE_LABELS:
        return None
    try:
        is_candidate, reason, analysis, candidates = mr.classify_candidate(
            rule_label, token, annotator, cfg, return_candidates=True)
        if not is_candidate:
            return None
        feats = mr.build_structured_feature_dict(
            token, rule_label, analysis, annotator, cfg,
            is_candidate=is_candidate, candidate_reason=reason,
            include_batch_a=True, include_batch_c=True, include_batch_g=True,
            candidate_analyses=candidates, candidate_strategy=mr.DEFAULT_CANDIDATE_STRATEGY)
        X = sp.hstack([bundle.tfidf.transform([token]), bundle.dictvec.transform([feats])]).tocsr()
        return float(bundle.model.predict_proba(X)[0, 1])
    except Exception:
        return None


def _residual_verbal_evidence(token: str, rule_label: str, annotator, cfg) -> Optional[Tuple[bool, str]]:
    """Re-derive reranking.evaluate_residual_verbal_promotion's verdict
    for `token`, read-only, only when `rule_label` was ever eligible for
    that stage in production (mirrors
    reranking._RESIDUAL_VERBAL_ELIGIBLE_LABELS). Returns
    (promote, reason) or None if not eligible / evaluation failed."""
    if rule_label not in _RESIDUAL_VERBAL_ELIGIBLE_LABELS:
        return None
    try:
        promote, _candidate, reason = mr.evaluate_residual_verbal_promotion(token, annotator, cfg)
        return promote, reason
    except Exception:
        return None


def _uid_resolver_evidence(token: str, rule_label: str, annotator, cfg, matrix_lang):
    """Re-derive reranking.decide()'s full explainable decision for
    `token`, read-only, only when `rule_label` was ever eligible in
    production (mirrors
    reranking._UID_TR_RESOLVER_ELIGIBLE_LABELS). Returns a
    ur.ResolverDecision or None if not eligible / evaluation failed."""
    if rule_label not in _UID_TR_RESOLVER_ELIGIBLE_LABELS:
        return None
    try:
        return ur.decide(token, "UID", annotator, cfg, matrix_lang=matrix_lang)
    except Exception:
        return None


def _detect_entity_type(annotator, sentence_text: str, token: str) -> Optional[str]:
    """Best-effort Stanza entity-type lookup for `token` within
    `sentence_text`. Reuses the SAME cached Stanza pipeline object the
    production pipeline already loaded (`annotator.ner`) -- never loads or
    constructs a new NER model. This re-runs NER inference for the
    sentence (Stanza itself keeps no per-call state affecting future
    calls), so it is gated behind `compute_entity_types=False` by default
    in the interactive app (see attach_confidence_to_blocks) and only
    enabled where the extra inference cost is acceptable (e.g. an offline
    evaluation run). Returns None on any failure, when NER was never
    enabled for this annotator, or when no matching entity is found --
    never raises."""
    try:
        if getattr(annotator, "ner", None) is None:
            return None
        doc = annotator.ner(sentence_text)
        for ent in getattr(doc, "ents", None) or []:
            pieces = set(p for p in re.findall(r"\w+['’]?\w*|\w+", ent.text) if p)
            if token in pieces:
                return ent.type
        return None
    except Exception:
        return None


def _promoted_by(rule_label: str, final_label: str, reranker_prob: Optional[float],
                  bundle, residual, uid_decision) -> Optional[str]:
    """Deterministically identify which production stage (if any) changed
    `token`'s label from `rule_label` to `final_label`, by re-checking each
    stage's own real condition in the same order production applies them
    (frozen reranker -> residual verbal detector -> UID->TR resolver). This
    never guesses: every branch below reuses evidence already recomputed
    read-only by this module. Returns None when the label was never
    changed (rule_label == final_label)."""
    if rule_label == final_label:
        return None
    if final_label == "MIXED":
        if reranker_prob is not None and bundle is not None and reranker_prob >= bundle.threshold:
            return "frozen_reranker"
        if residual is not None and residual[0]:
            return "residual_verbal_detector"
        return "unknown_stage"
    if rule_label == "UID" and final_label == "TR":
        if uid_decision is not None and uid_decision.promote:
            return "uid_to_tr_resolver"
        return "unknown_stage"
    return "unknown_stage"


# ---------------------------------------------------------------------------
# Per-label scoring. Each function returns (score, reasons, evidence_summary)
# -- `score` unclamped (clamped to [0, 1] by the caller), `reasons` a list of
# short machine-stable strings explaining any uncertainty, `evidence_summary`
# a list of short human-readable evidence descriptions.
# ---------------------------------------------------------------------------

def _score_other(ev: _CommonEvidence, **_) -> Tuple[float, List[str], List[str]]:
    if ev.hard_excluded:
        return 0.98, [], [
            "token shape matches a deterministic OTHER exclusion pattern "
            "(URL/mention/hashtag/number/code/emoji/punctuation)"
        ]
    return 0.45, ["other_label_does_not_match_any_automatic_exclusion_pattern"], [
        "OTHER label present but token shape does not match any automatic exclusion pattern "
        "(likely a manual/non-standard assignment)"
    ]


def _score_lang3(ev: _CommonEvidence, **_) -> Tuple[float, List[str], List[str]]:
    return 0.30, ["lang3_is_manual_only_no_automatic_pipeline_evidence_exists"], [
        "LANG3 is never assigned by the automatic pipeline (manual-only, see CLAUDE.md); "
        "this label was necessarily set by a human reviewer"
    ]


def _score_ne(ev: _CommonEvidence, ne_entity_type: Optional[str] = None, **_) -> Tuple[float, List[str], List[str]]:
    reasons: List[str] = []
    summary: List[str] = []
    if ne_entity_type:
        summary.append(f"Stanza entity subtype = {ne_entity_type}")
        if ne_entity_type in ("PERSON", "LOCATION"):
            score = 0.90
        elif ne_entity_type == "ORGANIZATION":
            score = 0.85
        else:
            score = 0.70
            reasons.append("entity_subtype_outside_person_location_organization")
    else:
        score = 0.65
        reasons.append("entity_type_not_available_best_effort_only")
        summary.append("entity subtype not recomputed for this run (best-effort evidence only)")
    if ev.en_lex and ev.has_suffix_analysis:
        score -= 0.10
        reasons.append("policy_boundary_case_english_lexicon_and_turkish_suffix_both_present")
        summary.append("token also has an English-lexicon match AND a Turkish suffix analysis "
                        "(NE Policy C/D boundary case)")
    return score, reasons, summary


def _score_tr(ev: _CommonEvidence, cfg=None, rule_label=None, final_label=None,
              promoted_by=None, uid_decision=None, **_) -> Tuple[float, List[str], List[str]]:
    ft_min = (cfg or {}).get("FT_TR_MIN", 0.80)
    reasons: List[str] = []
    summary: List[str] = []
    score = 0.30
    if ev.tr_top:
        score = max(score, 0.90)
        summary.append("token in top-1000 Turkish frequency lexicon")
    elif ev.tr_all:
        score = max(score, 0.80)
        summary.append("token in full Turkish frequency lexicon")
    if ev.ft_lang == "TR" and ev.ft_prob >= ft_min:
        score = max(score, 0.75)
        summary.append(f"fastText predicts TR (p={ev.ft_prob:.2f} >= {ft_min:.2f})")
    if ev.has_suffix_analysis:
        score += 0.05
        summary.append("valid Turkish suffix-chain analysis")
    if ev.orthographic:
        score += 0.03
        summary.append("Turkish-specific orthographic character present")
    if ev.en_lex:
        score -= 0.25
        reasons.append("token_also_present_in_english_lexicon")
        summary.append("also found in English frequency lexicon (conflicting signal)")
    if promoted_by == "uid_to_tr_resolver":
        reasons.append("label_recovered_by_uid_to_tr_resolver_not_direct_rule_evidence")
        summary.append("promoted from UID by the UID->TR resolver, not a direct rule-based match")
        score = min(score, 0.80)
        if uid_decision is not None:
            summary.append(
                f"uid_to_tr_resolver score={uid_decision.score} "
                f"(promotion_threshold={ur.PROMOTION_THRESHOLD})")
    if not ev.tr_top and not ev.tr_all and not ev.has_suffix_analysis and not (
            ev.ft_lang == "TR" and ev.ft_prob >= ft_min):
        reasons.append("no_strong_positive_turkish_evidence")
    return score, reasons, summary


def _score_en(ev: _CommonEvidence, cfg=None, **_) -> Tuple[float, List[str], List[str]]:
    ft_min = (cfg or {}).get("FT_EN_MIN", 0.80)
    reasons: List[str] = []
    summary: List[str] = []
    score = 0.30
    if ev.en_lex:
        score = max(score, 0.85)
        summary.append("token in English frequency lexicon")
    if ev.ft_lang == "EN" and ev.ft_prob >= ft_min:
        score = max(score, 0.75)
        summary.append(f"fastText predicts EN (p={ev.ft_prob:.2f} >= {ft_min:.2f})")
    if ev.tr_top or ev.tr_all:
        score -= 0.30
        reasons.append("token_also_present_in_turkish_lexicon")
        summary.append("also found in Turkish frequency lexicon (conflicting signal)")
    if ev.has_suffix_analysis:
        score -= 0.10
        reasons.append("turkish_suffix_analysis_also_present")
        summary.append("a valid Turkish suffix-chain analysis also exists for this token")
    if not ev.en_lex and not (ev.ft_lang == "EN" and ev.ft_prob >= ft_min):
        reasons.append("no_strong_positive_english_evidence")
    return score, reasons, summary


def _score_mixed(ev: _CommonEvidence, rule_label=None, reranker_prob=None, bundle=None,
                  residual=None, **_) -> Tuple[float, List[str], List[str]]:
    reasons: List[str] = []
    summary: List[str] = []
    if rule_label == "MIXED":
        score = 0.88
        summary.append("rule-based MIXED detection (apostrophe-split or closed-class suffix match) -- "
                        "a strict, deterministic production rule, not a probabilistic model")
        return score, reasons, summary
    if reranker_prob is not None and bundle is not None and reranker_prob >= bundle.threshold:
        score = reranker_prob
        margin = reranker_prob - bundle.threshold
        summary.append(f"frozen reranker probability={reranker_prob:.3f} "
                        f"(threshold={bundle.threshold:.2f}, margin=+{margin:.3f})")
        if margin < 0.05:
            reasons.append("reranker_score_close_to_threshold")
        return score, reasons, summary
    if residual is not None and residual[0]:
        score = 0.72
        summary.append(f"residual verbal MIXED detector promoted this token (reason={residual[1]})")
        reasons.append("promoted_by_residual_verbal_detector_not_probabilistic_model")
        return score, reasons, summary
    score = 0.5
    reasons.append("mixed_label_without_reconstructable_promotion_evidence")
    summary.append("no reconstructable rule-based/reranker/residual-detector evidence found for this MIXED label")
    return score, reasons, summary


def _score_uid(ev: _CommonEvidence, reranker_prob=None, bundle=None, residual=None,
               uid_decision=None, **_) -> Tuple[float, List[str], List[str]]:
    score = 0.50
    reasons: List[str] = ["label_uid_by_definition_below_lid_confidence_threshold"]
    summary: List[str] = []
    if reranker_prob is not None and bundle is not None:
        gap = bundle.threshold - reranker_prob
        summary.append(f"frozen reranker probability={reranker_prob:.3f} (did not clear "
                        f"threshold={bundle.threshold:.2f})")
        if gap < 0.10:
            score -= 0.15
            reasons.append("near_miss_frozen_reranker_threshold")
        else:
            score += 0.05
    if uid_decision is not None:
        summary.append(f"uid_to_tr_resolver score={uid_decision.score}/{ur.PROMOTION_THRESHOLD} "
                        f"({uid_decision.reason})")
        if uid_decision.score >= ur.PROMOTION_THRESHOLD - 2:
            score -= 0.15
            reasons.append("near_miss_uid_to_tr_resolver_threshold")
        elif uid_decision.score == 0:
            score += 0.05
    if residual is not None and residual[0] is False and residual[1] not in (
            "no_verbal_candidate", "proper_name_or_noise"):
        score -= 0.05
        reasons.append("residual_verbal_candidate_considered_and_rejected")
        summary.append(f"residual verbal detector considered a candidate but rejected it "
                        f"(reason={residual[1]})")
    if (not ev.tr_top and not ev.tr_all and not ev.en_lex and not ev.has_suffix_analysis
            and ev.ft_lang not in ("TR", "EN")):
        score += 0.05
        summary.append("no lexicon/suffix/fastText evidence found for any label (genuinely unresolvable)")
    # UID is semantically "confidence below the LID threshold" -- it must
    # never be reported as a HIGH-confidence label no matter how much
    # corroborating "nothing else fits" evidence accumulates.
    score = min(score, 0.75)
    return score, reasons, summary


_SCORERS = {
    "OTHER": _score_other,
    "LANG3": _score_lang3,
    "NE": _score_ne,
    "TR": _score_tr,
    "EN": _score_en,
    "MIXED": _score_mixed,
    "UID": _score_uid,
}


def _matrix_embed_consistency_adjustment(final_label: str, matrix_lang: Optional[str],
                                          embed_lang: Optional[str]) -> Tuple[float, List[str], List[str]]:
    if final_label == "EN" and matrix_lang == "TR" and embed_lang not in (None, "", "EN"):
        return -0.05, ["embed_lang_inconsistent_with_en_token_in_tr_matrix_sentence"], [
            f"sentence MatrixLang=TR but EmbedLang={embed_lang!r} (expected EN for this EN token)"]
    if final_label == "TR" and matrix_lang == "EN" and embed_lang not in (None, "", "TR"):
        return -0.05, ["embed_lang_inconsistent_with_tr_token_in_en_matrix_sentence"], [
            f"sentence MatrixLang=EN but EmbedLang={embed_lang!r} (expected TR for this TR token)"]
    return 0.0, [], []


def compute_token_confidence(token: str, rule_label: Optional[str], final_label: str, annotator, cfg,
                              *, matrix_lang: Optional[str] = None, embed_lang: Optional[str] = None,
                              bundle=None, ne_entity_type: Optional[str] = None,
                              thresholds: Optional[Dict[str, float]] = None) -> ConfidenceRecord:
    """The single entry point combining evidence gathering, per-label
    scoring, and banding for one token. Deterministic: identical inputs
    always produce an identical ConfidenceRecord. Never raises -- any
    unexpected failure in an evidence sub-step is caught locally (see the
    `_*_evidence`/`_reranker_probability` helpers) and simply omits that
    piece of evidence rather than aborting the whole computation.

    `rule_label` is the label Annotator.annotate() itself produced for this
    token, BEFORE apply_reranker() ran (None if unknown/unavailable, e.g. a
    manually-inserted row with no corresponding pre-rerank row -- treated
    as "no promotion detected", never a crash).
    """
    rule_label_eff = rule_label if rule_label is not None else final_label
    try:
        ev = _gather_common_evidence(token, annotator, cfg)
    except Exception:
        # A malformed/incomplete `annotator` (e.g. missing lexicon sets)
        # must never crash confidence computation -- fall back to a
        # neutral "no evidence available" reading instead.
        ev = _CommonEvidence(tr_top=False, tr_all=False, en_lex=False, ft_lang="", ft_prob=0.0,
                              has_suffix_analysis=False, orthographic=False,
                              has_apostrophe=("'" in token or "’" in token), hard_excluded=False)
    reranker_prob = _reranker_probability(token, rule_label_eff, annotator, cfg, bundle)
    residual = _residual_verbal_evidence(token, rule_label_eff, annotator, cfg)
    uid_decision = _uid_resolver_evidence(token, rule_label_eff, annotator, cfg, matrix_lang)
    promoted_by = _promoted_by(rule_label_eff, final_label, reranker_prob, bundle, residual, uid_decision)

    scorer = _SCORERS.get(final_label)
    if scorer is None:
        score, reasons, summary = 0.5, ["unrecognized_label_outside_seven_label_schema"], []
    else:
        score, reasons, summary = scorer(
            ev, cfg=cfg, rule_label=rule_label_eff, final_label=final_label,
            reranker_prob=reranker_prob, bundle=bundle, residual=residual,
            uid_decision=uid_decision, promoted_by=promoted_by, ne_entity_type=ne_entity_type,
        )
        reasons = list(reasons)
        summary = list(summary)

    if final_label in ("TR", "EN", "MIXED"):
        delta, mreasons, msummary = _matrix_embed_consistency_adjustment(final_label, matrix_lang, embed_lang)
        score += delta
        reasons.extend(mreasons)
        summary.extend(msummary)

    if ev.has_apostrophe and final_label not in ("MIXED", "OTHER"):
        reasons.append("apostrophe_bearing_token_with_non_mixed_label")
        summary.append("token contains an apostrophe but was not labeled MIXED")

    score = max(0.0, min(1.0, score))
    band = band_for_score(score, thresholds)
    review_recommended = band in (BAND_LOW, BAND_MEDIUM)

    return ConfidenceRecord(
        token=token,
        rule_based_label=rule_label,
        final_label=final_label,
        confidence_score=score,
        confidence_band=band,
        uncertainty_reasons=tuple(reasons),
        evidence_summary=tuple(summary),
        review_recommended=review_recommended,
        promoted_by=promoted_by,
    )


# ---------------------------------------------------------------------------
# Block/dataset-level integration
# ---------------------------------------------------------------------------

def _block_matrix_embed(rows) -> Tuple[Optional[str], Optional[str]]:
    matrix_lang = None
    embed_lang = None
    for r in rows:
        tok = str(r.get("token", "") or "").strip()
        if tok == "MatrixLang":
            matrix_lang = r.get("label")
        elif tok == "EmbedLang":
            embed_lang = r.get("label")
    return matrix_lang, embed_lang


def _non_meta_rows(rows):
    return [r for r in rows if not annotation_model.is_meta_row_token(r.get("token", ""))]


def compute_block_confidence(rows, rule_rows, annotator, cfg, bundle=None,
                              compute_entity_types: bool = False,
                              thresholds: Optional[Dict[str, float]] = None) -> None:
    """Mutate `rows` (one block/sentence's list of row dicts, in the same
    shape `annotation_model.parse_annotated_text_to_blocks` produces) in
    place: sets `row["confidence"]` (a JSON-serializable dict, see
    ConfidenceRecord.to_dict) and `row.setdefault("reviewed", False)` on
    every non-meta token row.

    `rule_rows` must be the SAME sentence's rows parsed from the
    pre-reranker Annotator.annotate() output (same token order) -- used
    only to recover each token's pre-reranker rule label by position among
    non-meta rows. A length mismatch (should not happen in production,
    since neither apply_reranker() nor Matrix/Embed consistency ever
    add/remove token rows) degrades gracefully to `rule_label=None` for the
    unmatched tail rather than raising or misaligning.
    """
    matrix_lang, embed_lang = _block_matrix_embed(rows)
    rule_non_meta = _non_meta_rows(rule_rows) if rule_rows else []

    sentence_text = None
    if compute_entity_types:
        sentence_text = " ".join(str(r.get("token", "") or "") for r in _non_meta_rows(rows))

    j = 0
    for r in rows:
        tok = r.get("token", "")
        if annotation_model.is_meta_row_token(tok):
            continue
        rule_label = rule_non_meta[j].get("label") if j < len(rule_non_meta) else None
        j += 1

        final_label = r.get("label")
        ne_type = None
        if compute_entity_types and final_label == "NE" and sentence_text:
            ne_type = _detect_entity_type(annotator, sentence_text, tok)

        try:
            record = compute_token_confidence(
                tok, rule_label, final_label, annotator, cfg,
                matrix_lang=matrix_lang, embed_lang=embed_lang, bundle=bundle,
                ne_entity_type=ne_type, thresholds=thresholds,
            )
            r["confidence"] = record.to_dict()
        except Exception as e:
            # Never let one token's confidence computation interrupt the
            # rest of the block/dataset -- mirrors apply_reranker()'s own
            # fail-safe contract. The label itself (r["label"]) is never
            # touched here regardless of this failure.
            r["confidence"] = ConfidenceRecord(
                token=tok, rule_based_label=rule_label, final_label=final_label,
                confidence_score=0.0, confidence_band=BAND_LOW,
                uncertainty_reasons=(f"confidence_computation_failed: {e}",),
                evidence_summary=(), review_recommended=True, promoted_by=None,
            ).to_dict()
        r.setdefault("reviewed", False)


def attach_confidence_to_blocks(blocks, rule_blocks, annotator, cfg, bundle=None,
                                 compute_entity_types: bool = False,
                                 thresholds: Optional[Dict[str, float]] = None) -> None:
    """Mutate `blocks` (a full dataset's list of blocks) in place, calling
    `compute_block_confidence` for every block. `rule_blocks` must be the
    pre-reranker parse of the SAME annotation run (same block count, same
    per-block token order) -- see `_run_annotation_pipeline_with_confidence`
    in cs_annotator_app.py for how the two texts are produced together.
    A block-count mismatch degrades gracefully (extra blocks in `blocks`
    get `rule_label=None` for every token) rather than raising."""
    for bidx, rows in enumerate(blocks):
        rule_rows = rule_blocks[bidx] if rule_blocks and bidx < len(rule_blocks) else []
        compute_block_confidence(rows, rule_rows, annotator, cfg, bundle=bundle,
                                  compute_entity_types=compute_entity_types, thresholds=thresholds)


# ---------------------------------------------------------------------------
# Read-only accessors used by the review tool / tests. Both are defensive
# against legacy rows/projects that predate this module (no "confidence"
# key at all) -- never raise, never fabricate a fake high-confidence result.
# ---------------------------------------------------------------------------

def get_confidence(row: dict) -> Optional[dict]:
    val = row.get("confidence")
    return val if isinstance(val, dict) else None


def is_reviewed(row: dict) -> bool:
    return bool(row.get("reviewed", False))


def mark_reviewed(row: dict) -> None:
    row["reviewed"] = True


def note_manual_edit(row: dict) -> None:
    """Call whenever a row's label is manually changed outside the
    automatic pipeline (e.g. the review tool's Apply action). The row's
    existing "confidence" record (if any) was computed for the PREVIOUS
    label and would be actively misleading if left in place (its
    evidence_summary/uncertainty_reasons describe a label this row no
    longer carries) -- this replaces it with a minimal, honest marker
    instead of leaving stale text around or silently deleting the key.
    Never recomputes real evidence inline (that would mean running the
    frozen reranker/residual-verbal/UID-resolver inference on every
    manual edit); a fresh, evidence-backed confidence record for the new
    label is only ever produced by re-running the annotation pipeline.
    Also marks the row reviewed=True. Deterministic, no I/O."""
    row["confidence"] = {
        "token": row.get("token", ""),
        "rule_based_label": None,
        "final_label": row.get("label", ""),
        "confidence_score": 1.0,
        "confidence_band": BAND_HIGH,
        "uncertainty_reasons": ["manually_edited_after_automatic_annotation"],
        "evidence_summary": ["label/gloss set directly by a human reviewer, not scored by automatic evidence"],
        "review_recommended": False,
        "promoted_by": "manual_edit",
        "calibration_note": CALIBRATION_NOTE,
    }
    row["reviewed"] = True


def band_of(row: dict) -> Optional[str]:
    conf = get_confidence(row)
    return conf.get("confidence_band") if conf else None


def is_review_required(row: dict) -> bool:
    """Whether `row` should appear in an "all uncertain tokens" view,
    across any label (TR/EN/MIXED/UID/NE/OTHER/LANG3) -- reuses the
    confidence record's own `review_recommended` flag verbatim (the same
    flag `compute_token_confidence` sets from the record's band; see
    `band_for_score`/`DEFAULT_THRESHOLDS`), never a label-based rule. A row
    with no confidence record at all (legacy project, or never
    re-annotated since this layer was added) is not "required" -- there is
    no evidence either way, so it is excluded rather than assumed
    uncertain."""
    conf = get_confidence(row)
    return bool(conf) and bool(conf.get("review_recommended"))
