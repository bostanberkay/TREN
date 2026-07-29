# reranker_integration.py
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

import json
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

import scipy.sparse as sp

import annotation_model
import mixed_reranker as mr

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
        is_candidate, reason, analysis, candidates = mr.classify_candidate(
            label, item, annotator, cfg, return_candidates=True)
        if not is_candidate:
            return label

        feats = mr.build_structured_feature_dict(
            item, label, analysis, annotator, cfg,
            is_candidate=is_candidate, candidate_reason=reason,
            include_batch_a=True, include_batch_c=True, include_batch_g=True,
            candidate_analyses=candidates, candidate_strategy=mr.DEFAULT_CANDIDATE_STRATEGY)

        X = sp.hstack([bundle.tfidf.transform([item]), bundle.dictvec.transform([feats])]).tocsr()
        prob = bundle.model.predict_proba(X)[0, 1]
        if prob >= bundle.threshold:
            return "MIXED"
        return label
    except Exception as e:
        _warn(f"reranking failed for {item!r} (label={label!r}), keeping original: {e}")
        return label


def apply_reranker(annotated_text: str, annotator, cfg: dict, bundle: Optional[ReRankerBundle]) -> str:
    """Post-process Annotator.annotate()'s raw text output with the frozen
    Phase 5F reranker.

    Input and output are the EXACT same tab-separated text format
    annotate() itself produces -- no schema change, no new fields, no new
    meta rows. The only rewrites are: (1) a token row's Label, when the
    reranker promotes a UID/NE/TR candidate to MIXED; (2) an existing
    MatrixLang/EmbedLang row's value, recomputed via the unmodified
    Annotator._decide_matrix_embed, and only for blocks where a label
    actually changed. Blocks with no label change are returned
    byte-for-byte identical to the input.

    If `bundle` is None (model unavailable), returns `annotated_text`
    completely unchanged -- this is the fallback path when
    load_reranker_bundle() failed for any reason, so annotation output is
    identical to the original rule-based-only behavior. Never raises.
    Performs no parser or candidate-generation changes -- called from
    cs_annotator_app.py's run_pipeline() on every annotation request, right
    after Annotator.annotate() and before _ensure_matrix_embed_consistency().
    """
    if bundle is None:
        return annotated_text

    blocks = annotated_text.split("\n\n")
    new_blocks = []
    for block in blocks:
        records = _parse_block_lines(block)

        changed = False
        for rec in records:
            if rec["kind"] != "token":
                continue
            new_label = _rerank_token_label(rec["item"], rec["label"], annotator, cfg, bundle)
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
