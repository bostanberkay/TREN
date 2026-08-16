#!/usr/bin/env python
# tools/ne_cascade_policy.py
"""Phase 3A, isolated experiment: investigate and benchmark conservative
cascade policies for candidate tokens whose ORIGINAL predicted label is NE,
to prevent unsupported NE->MIXED flips without materially reducing recovery
of genuine gold-MIXED tokens originally predicted NE.

Read-only with respect to Phase 2: loads the already-trained model,
vectorizers, dataset, and split manifest from an existing
tools/build_reranker_dataset.py + tools/train_mixed_reranker.py run (default:
artifacts/mixed_reranker/strategy_comparison/highest_suffix_segments/) and
never writes into that directory. Does not retrain anything -- the
classifier architecture and its weights are exactly Phase 2's.

Four policies (Decision A-D):
  A. unrestricted    -- current behaviour: any candidate flips to MIXED if
                        model P(MIXED) >= the general threshold, regardless
                        of original predicted label.
  B. block_ne        -- candidates originally predicted NE NEVER flip to
                        MIXED, regardless of model probability.
  C. ne_threshold    -- candidates originally predicted NE flip to MIXED
                        only if P(MIXED) >= a SEPARATE, higher threshold
                        (tuned independently of the general threshold).
  D. ne_stem_evidence -- candidates originally predicted NE flip to MIXED
                        only if BOTH P(MIXED) >= the general threshold AND
                        independent (non-model) evidence the stem is
                        non-Turkish: stem_in_english_freq OR (stem fastText
                        language == EN AND stem fastText prob >= FT_EN_MIN).
                        This is the exact same test already used by
                        reranking.is_non_turkish_stem_evidence for the
                        TR candidate bucket -- reused here for consistency,
                        not reinvented.

Non-NE candidates are ALWAYS governed by the general threshold under every
policy -- only the NE bucket's rule changes. This mirrors
tools/train_mixed_reranker.py's simulate_cascade() but adds the NE-specific
gate; kept as a separate function here (rather than editing that function)
to avoid touching Phase 1/2 code paths while this is still exploratory.

Selection criterion (DEV ONLY, stated explicitly, applied mechanically):
  1. Minimize harmful NE->MIXED changes on DEV (gold label != MIXED, but
     ends up simulated as MIXED).
  2. Among policies tied on (1), maximize genuine NE->MIXED corrections
     retained on DEV (gold label == MIXED, ends up simulated as MIXED).
  3. Among policies still tied, maximize overall DEV MIXED F1.
The winning policy (with, for policy C, its DEV-tuned ne_threshold) is then
evaluated EXACTLY ONCE on the test split.
"""
import argparse
import csv
import json
import os
import sys

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cs_pipeline import DEFAULTS

POLICY_UNRESTRICTED = "unrestricted"
POLICY_BLOCK_NE = "block_ne"
POLICY_NE_THRESHOLD = "ne_threshold"
POLICY_NE_STEM_EVIDENCE = "ne_stem_evidence"
ALL_POLICIES = (POLICY_UNRESTRICTED, POLICY_BLOCK_NE, POLICY_NE_THRESHOLD, POLICY_NE_STEM_EVIDENCE)


def stem_evidence(structured_features: dict) -> bool:
    """Same definition as reranking.is_non_turkish_stem_evidence,
    applied to the ALREADY-COMPUTED stem features stored on the row (no
    Annotator/fastText call needed here -- build_reranker_dataset.py
    already ran it once per row)."""
    if structured_features.get("stem_length", 0) == 0:
        return False  # no candidate analysis at all -> no stem to have evidence about
    if structured_features.get("stem_in_english_freq"):
        return True
    return structured_features.get("stem_ft_lang") == "EN" and \
        structured_features.get("stem_ft_prob", 0.0) >= DEFAULTS["FT_EN_MIN"]


def simulate_with_ne_policy(rows, probs, threshold, policy, ne_threshold=None):
    """Apply one policy to a set of (row, prob) pairs. Returns a list of
    dicts, one per row, with simulated_label/changed/beneficial/harmful/neutral
    -- same semantics as tools/train_mixed_reranker.py's simulate_cascade."""
    if policy == POLICY_NE_THRESHOLD and ne_threshold is None:
        raise ValueError("ne_threshold policy requires an explicit ne_threshold")

    results = []
    for row, prob in zip(rows, probs):
        is_ne = row["pred_label"] == "NE"
        eligible = row["is_candidate"]
        if eligible and prob >= threshold:
            would_flip = True
        else:
            would_flip = False

        if is_ne and would_flip:
            if policy == POLICY_UNRESTRICTED:
                pass  # no extra restriction
            elif policy == POLICY_BLOCK_NE:
                would_flip = False
            elif policy == POLICY_NE_THRESHOLD:
                would_flip = prob >= ne_threshold
            elif policy == POLICY_NE_STEM_EVIDENCE:
                would_flip = stem_evidence(row["structured_features"])
            else:
                raise ValueError(f"unknown policy: {policy!r}")

        simulated_label = "MIXED" if would_flip else row["pred_label"]
        changed = simulated_label != row["pred_label"]
        beneficial = changed and row["gold_label"] == "MIXED"
        harmful = changed and row["gold_label"] != "MIXED" and row["pred_label"] == row["gold_label"]
        neutral = changed and not beneficial and not harmful
        results.append({**row, "model_probability": float(prob), "simulated_label": simulated_label,
                         "changed": changed, "beneficial": beneficial, "harmful": harmful, "neutral": neutral})
    return results


def multiclass_report(gold_labels, pred_labels):
    labels_present = sorted(set(gold_labels) | set(pred_labels))
    p, r, f1, support = precision_recall_fscore_support(gold_labels, pred_labels, labels=labels_present, zero_division=0)
    per_label = {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
                 for i, lab in enumerate(labels_present)}
    acc = float(accuracy_score(gold_labels, pred_labels))
    macro_f1 = float(f1_score(gold_labels, pred_labels, labels=labels_present, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gold_labels, pred_labels, labels=labels_present, average="weighted", zero_division=0))
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_label": per_label}


def ne_stats(sim_rows):
    """NE-specific counts used by the selection criterion and the final report."""
    ne_rows = [r for r in sim_rows if r["pred_label"] == "NE"]
    harmful_ne = [r for r in ne_rows if r["changed"] and r["gold_label"] != "MIXED"]
    genuine_ne = [r for r in ne_rows if r["gold_label"] == "MIXED"]
    genuine_ne_retained = [r for r in genuine_ne if r["simulated_label"] == "MIXED"]
    return {
        "n_ne_candidates": len(ne_rows),
        "n_harmful_ne_changes": len(harmful_ne),
        "harmful_ne_tokens": [r["text"] for r in harmful_ne],
        "n_genuine_ne_mixed": len(genuine_ne),
        "n_genuine_ne_retained": len(genuine_ne_retained),
        "genuine_ne_retained_tokens": [r["text"] for r in genuine_ne_retained],
        "genuine_ne_missed_tokens": [r["text"] for r in genuine_ne if r["simulated_label"] != "MIXED"],
    }


def overall_mixed_f1(sim_rows):
    y_true = [1 if r["gold_label"] == "MIXED" else 0 for r in sim_rows]
    y_pred = [1 if r["simulated_label"] == "MIXED" else 0 for r in sim_rows]
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return float(f1[1])


def selection_key(sim_rows):
    """(fewer harmful is better, more genuine-retained is better, higher F1 is better)
    -- negate the "more/higher is better" terms so a plain min() finds the winner,
    matching the 3-tier criterion documented in the module docstring."""
    ns = ne_stats(sim_rows)
    return (ns["n_harmful_ne_changes"], -ns["n_genuine_ne_retained"], -overall_mixed_f1(sim_rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default="artifacts/mixed_reranker/strategy_comparison/highest_suffix_segments",
                     help="Existing Phase 2 artifacts directory (read-only) containing dataset.json, "
                          "split_manifest.json, model.joblib, vectorizer.joblib, metadata.json")
    ap.add_argument("--out-dir", required=True, help="Output directory for this Phase 3A experiment's own artifacts")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] Loading Phase 2 artifacts (read-only) from {args.artifacts_dir} ...")
    rows = json.load(open(os.path.join(args.artifacts_dir, "dataset.json"), encoding="utf-8"))
    metadata = json.load(open(os.path.join(args.artifacts_dir, "metadata.json"), encoding="utf-8"))
    model = joblib.load(os.path.join(args.artifacts_dir, "model.joblib"))
    vec = joblib.load(os.path.join(args.artifacts_dir, "vectorizer.joblib"))
    tfidf, dictvec = vec["tfidf"], vec["dictvec"]

    base_threshold = metadata["selected_thresholds"]["max_f1"]
    print(f"    base (general) decision threshold, frozen from Phase 2: {base_threshold}")

    dev_rows = [r for r in rows if r["split"] == "dev"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"    dev={len(dev_rows)} rows, test={len(test_rows)} rows")

    def probs_for(rows_):
        text = [r["text"] for r in rows_]
        struct = [r["structured_features"] for r in rows_]
        X = sp.hstack([tfidf.transform(text), dictvec.transform(struct)]).tocsr()
        return model.predict_proba(X)[:, 1]

    dev_probs = probs_for(dev_rows)
    test_probs = probs_for(test_rows)

    print("[2/4] Tuning on DEV ONLY...")
    ne_threshold_grid = [round(x, 2) for x in np.arange(base_threshold, 1.00, 0.01)]
    best_c_threshold, best_c_key, best_c_sim = None, None, None
    for t in ne_threshold_grid:
        sim = simulate_with_ne_policy(dev_rows, dev_probs, base_threshold, POLICY_NE_THRESHOLD, ne_threshold=t)
        key = selection_key(sim)
        if best_c_key is None or key < best_c_key:
            best_c_key, best_c_threshold, best_c_sim = key, t, sim
    print(f"    Policy C dev-tuned ne_threshold: {best_c_threshold}")

    dev_results = {}
    for policy in (POLICY_UNRESTRICTED, POLICY_BLOCK_NE, POLICY_NE_STEM_EVIDENCE):
        dev_results[policy] = simulate_with_ne_policy(dev_rows, dev_probs, base_threshold, policy)
    dev_results[POLICY_NE_THRESHOLD] = best_c_sim

    print("\n    --- DEV comparison across policies ---")
    policy_labels = {POLICY_UNRESTRICTED: "A unrestricted", POLICY_BLOCK_NE: "B block_ne",
                      POLICY_NE_THRESHOLD: f"C ne_threshold={best_c_threshold}", POLICY_NE_STEM_EVIDENCE: "D ne_stem_evidence"}
    for policy in ALL_POLICIES:
        sim = dev_results[policy]
        ns = ne_stats(sim)
        f1 = overall_mixed_f1(sim)
        print(f"    {policy_labels[policy]:28} harmful_NE={ns['n_harmful_ne_changes']} "
              f"genuine_NE_retained={ns['n_genuine_ne_retained']}/{ns['n_genuine_ne_mixed']} "
              f"overall_MIXED_F1={f1:.4f}  selection_key={selection_key(sim)}")

    selected_policy = min(ALL_POLICIES, key=lambda p: selection_key(dev_results[p]))
    print(f"\n    SELECTED POLICY (dev-only criterion): {policy_labels[selected_policy]}")

    print(f"\n[3/4] Evaluating selected policy on TEST (first and only time)...")
    test_sim = simulate_with_ne_policy(test_rows, test_probs, base_threshold, selected_policy,
                                        ne_threshold=best_c_threshold if selected_policy == POLICY_NE_THRESHOLD else None)
    gold = [r["gold_label"] for r in test_sim]
    original = [r["pred_label"] for r in test_sim]
    simulated = [r["simulated_label"] for r in test_sim]
    orig_report = multiclass_report(gold, original)
    sim_report = multiclass_report(gold, simulated)
    ns_test = ne_stats(test_sim)
    changed_tokens = [(r["text"], r["pred_label"], r["gold_label"], r["simulated_label"], round(r["model_probability"], 4))
                       for r in test_sim if r["changed"]]

    print("\n[4/4] Writing artifacts...")
    with open(os.path.join(args.out_dir, "dev_policy_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "ne_threshold_if_applicable", "harmful_ne_changes", "harmful_ne_tokens",
                     "genuine_ne_retained", "genuine_ne_total", "overall_mixed_f1", "selection_key"])
        for policy in ALL_POLICIES:
            sim = dev_results[policy]
            ns = ne_stats(sim)
            w.writerow([policy, best_c_threshold if policy == POLICY_NE_THRESHOLD else "",
                        ns["n_harmful_ne_changes"], "; ".join(ns["harmful_ne_tokens"]),
                        ns["n_genuine_ne_retained"], ns["n_genuine_ne_mixed"],
                        f"{overall_mixed_f1(sim):.4f}", str(selection_key(sim))])

    with open(os.path.join(args.out_dir, "test_changed_tokens.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "original_pred_label", "gold_label", "simulated_label", "model_probability"])
        for row in changed_tokens:
            w.writerow(row)

    report = {
        "base_threshold": base_threshold,
        "ne_threshold_grid_tested": ne_threshold_grid,
        "policy_c_selected_ne_threshold": best_c_threshold,
        "selection_criterion": "1) minimize DEV harmful NE->MIXED changes; 2) among ties, maximize DEV genuine "
                                "NE->MIXED corrections retained; 3) among remaining ties, maximize overall DEV MIXED F1. "
                                "Computed and applied using DEV split only; TEST touched exactly once, after selection.",
        "dev_comparison": {policy: {**ne_stats(dev_results[policy]), "overall_mixed_f1": overall_mixed_f1(dev_results[policy])}
                           for policy in ALL_POLICIES},
        "selected_policy": selected_policy,
        "test_results": {
            "original_multiclass": orig_report,
            "simulated_multiclass": sim_report,
            "ne_stats": ns_test,
            "all_changed_tokens": changed_tokens,
        },
    }
    with open(os.path.join(args.out_dir, "ne_policy_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    lines = []
    def p(s=""):
        lines.append(s)
    p("=" * 78)
    p("PHASE 3A -- NE CASCADE POLICY EXPERIMENT")
    p("=" * 78)
    p(f"Base (general) decision threshold (frozen from Phase 2, highest_suffix_segments): {base_threshold}")
    p(f"\n--- DEV-ONLY POLICY COMPARISON (selection criterion, see docstring) ---")
    for policy in ALL_POLICIES:
        sim = dev_results[policy]
        ns = ne_stats(sim)
        p(f"  {policy_labels[policy]:28} harmful_NE={ns['n_harmful_ne_changes']}  "
          f"genuine_NE_retained={ns['n_genuine_ne_retained']}/{ns['n_genuine_ne_mixed']}  "
          f"overall_MIXED_F1={overall_mixed_f1(sim):.4f}")
    p(f"\nSELECTED POLICY: {policy_labels[selected_policy]}")
    p(f"\n--- TEST RESULTS (selected policy, evaluated once) ---")
    p(f"Overall Label accuracy:  original={orig_report['accuracy']:.4f}  ->  simulated={sim_report['accuracy']:.4f}")
    p(f"Macro F1:                original={orig_report['macro_f1']:.4f}  ->  simulated={sim_report['macro_f1']:.4f}")
    p(f"Weighted F1:             original={orig_report['weighted_f1']:.4f}  ->  simulated={sim_report['weighted_f1']:.4f}")
    om = orig_report["per_label"].get("MIXED", {"precision": 0, "recall": 0, "f1": 0})
    sm = sim_report["per_label"].get("MIXED", {"precision": 0, "recall": 0, "f1": 0})
    p(f"MIXED precision:         original={om['precision']:.4f}  ->  simulated={sm['precision']:.4f}")
    p(f"MIXED recall:            original={om['recall']:.4f}  ->  simulated={sm['recall']:.4f}")
    p(f"MIXED F1:                original={om['f1']:.4f}  ->  simulated={sm['f1']:.4f}")
    p(f"Genuine NE->MIXED corrections retained: {ns_test['n_genuine_ne_retained']}/{ns_test['n_genuine_ne_mixed']} "
      f"-> {ns_test['genuine_ne_retained_tokens']}")
    if ns_test["genuine_ne_missed_tokens"]:
        p(f"  (missed: {ns_test['genuine_ne_missed_tokens']})")
    p(f"Harmful NE->MIXED changes: {ns_test['n_harmful_ne_changes']} -> {ns_test['harmful_ne_tokens']}")
    p(f"All changed tokens (text, orig_pred, gold, simulated, model_prob):")
    for row in changed_tokens:
        p(f"  {row}")

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(os.path.join(args.out_dir, "experiment_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\nWrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
