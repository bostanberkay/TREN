#!/usr/bin/env python
# tools/train_mixed_reranker.py
"""Phase 2, isolated experiment: train and evaluate a baseline MIXED
reranker (character TF-IDF + structured features -> LogisticRegression)
from the dataset built by tools/build_reranker_dataset.py.

This script does not import or modify cs_pipeline.py's Annotator behaviour,
does not touch cs_annotator_app.py, and writes only into --out-dir (a
gitignored experimental artifact directory). The corpus CSVs themselves are
never opened here -- this script only reads dataset.json / split_manifest.json
produced by the build step.

Strict ordering, enforced by this script's structure (not just documented):
  1. Fit TF-IDF + structured vectorizers on TRAIN rows only.
  2. Fit baseline-3 (char-TFIDF-only) and the full model on TRAIN only.
  3. Threshold search (max-F1, and best-threshold-with-precision>=0.75)
     using DEV predictions only.
  4. Freeze the selected threshold(s). Only then are TEST predictions
     computed, and only once -- TEST is never used for any decision above.
"""
import argparse
import csv
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone

import joblib
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                              f1_score, precision_recall_fscore_support,
                              roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mixed_reranker as mr

SCHEMA_LABELS = list(mr.SCHEMA_LABELS)
RANDOM_STATE = 42
MAX_ITER = 5000


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def load_dataset(dataset_path, manifest_path):
    with open(dataset_path, encoding='utf-8') as f:
        rows = json.load(f)
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    return rows, manifest


def split_rows(rows):
    by_split = defaultdict(list)
    for r in rows:
        by_split[r['split']].append(r)
    return by_split['train'], by_split['dev'], by_split['test']


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(train_rows, dev_rows, test_rows):
    """Fit TfidfVectorizer + DictVectorizer on TRAIN ONLY; transform dev/test.
    lowercase=True for the char n-gram vectorizer is an explicit, documented
    choice (case-folded character shapes generalize better across the small
    corpus) -- distinct from, and not in tension with, the alignment step's
    separate decision to apply NO lowercasing when matching gold/pred Item
    text for row alignment (a different purpose: identity vs. generalization).
    """
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), lowercase=True)
    dictvec = DictVectorizer(sparse=True)

    train_text = [r['text'] for r in train_rows]
    train_struct = [r['structured_features'] for r in train_rows]

    X_tfidf_train = tfidf.fit_transform(train_text)
    X_struct_train = dictvec.fit_transform(train_struct)

    def transform(rows):
        text = [r['text'] for r in rows]
        struct = [r['structured_features'] for r in rows]
        return tfidf.transform(text), dictvec.transform(struct)

    X_tfidf_dev, X_struct_dev = transform(dev_rows)
    X_tfidf_test, X_struct_test = transform(test_rows)

    combined_train = sp.hstack([X_tfidf_train, X_struct_train]).tocsr()
    combined_dev = sp.hstack([X_tfidf_dev, X_struct_dev]).tocsr()
    combined_test = sp.hstack([X_tfidf_test, X_struct_test]).tocsr()

    return {
        'tfidf': tfidf, 'dictvec': dictvec,
        'tfidf_only': {'train': X_tfidf_train, 'dev': X_tfidf_dev, 'test': X_tfidf_test},
        'combined': {'train': combined_train, 'dev': combined_dev, 'test': combined_test},
    }


def y_of(rows):
    return np.array([r['target'] for r in rows], dtype=int)


# ---------------------------------------------------------------------------
# Binary metrics helpers
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        'precision_mixed': float(p[1]), 'recall_mixed': float(r[1]), 'f1_mixed': float(f1[1]),
        'support_mixed': int(support[1]), 'support_not_mixed': int(support[0]),
        'confusion_matrix_labels': ['NOT_MIXED', 'MIXED'],
        'confusion_matrix': cm.tolist(),
    }


def prob_distribution(probs):
    if len(probs) == 0:
        return {}
    arr = np.asarray(probs)
    return {
        'n': int(arr.size), 'min': float(arr.min()), 'max': float(arr.max()),
        'mean': float(arr.mean()), 'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)), 'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)), 'p90': float(np.percentile(arr, 90)),
    }


def threshold_table(y_true, probs, grid=None):
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.01, 1.00, 0.01)]
    rows = []
    for t in grid:
        y_pred = (probs >= t).astype(int)
        p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            'threshold': t, 'precision_mixed': float(p[1]), 'recall_mixed': float(r[1]),
            'f1_mixed': float(f1[1]), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
    return rows


def select_thresholds(dev_table):
    """Max-F1 threshold, and best-recall threshold subject to precision >= 0.75
    (ties broken by F1). Both computed on DEV ONLY."""
    best_f1_row = max(dev_table, key=lambda r: (r['f1_mixed'], -r['threshold']))
    eligible = [r for r in dev_table if r['precision_mixed'] >= 0.75]
    if eligible:
        best_precision_row = max(eligible, key=lambda r: (r['recall_mixed'], r['f1_mixed']))
        precision_constraint_met = True
    else:
        best_precision_row = max(dev_table, key=lambda r: (r['precision_mixed'], r['f1_mixed']))
        precision_constraint_met = False
    return best_f1_row, best_precision_row, precision_constraint_met


# ---------------------------------------------------------------------------
# Multiclass (cascade) metrics helpers
# ---------------------------------------------------------------------------

def multiclass_report(gold_labels, pred_labels):
    labels_present = sorted(set(gold_labels) | set(pred_labels))
    p, r, f1, support = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=labels_present, zero_division=0)
    per_label = {lab: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f1[i]), 'support': int(support[i])}
                 for i, lab in enumerate(labels_present)}
    acc = float(accuracy_score(gold_labels, pred_labels))
    macro_f1 = float(f1_score(gold_labels, pred_labels, labels=labels_present, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(gold_labels, pred_labels, labels=labels_present, average='weighted', zero_division=0))
    cm = confusion_matrix(gold_labels, pred_labels, labels=labels_present)
    return {
        'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'per_label': per_label, 'labels': labels_present, 'confusion_matrix': cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Cascade simulation
# ---------------------------------------------------------------------------

def simulate_cascade(test_rows, full_model, tfidf, dictvec, threshold):
    """Non-candidate rows keep pred_label unchanged. Candidate rows become
    MIXED only if the full model's P(MIXED) >= threshold on THIS row;
    otherwise they retain pred_label (KEEP_ORIGINAL). Model probability is
    computed for every test row (for analysis/test_predictions.csv
    transparency) but only used to flip a label when is_candidate is True --
    this is enforced by the `if row['is_candidate']` gate below, not by
    omitting the computation.
    """
    text = [r['text'] for r in test_rows]
    struct = [r['structured_features'] for r in test_rows]
    X = sp.hstack([tfidf.transform(text), dictvec.transform(struct)]).tocsr()
    probs = full_model.predict_proba(X)[:, 1]

    results = []
    for row, prob in zip(test_rows, probs):
        if row['is_candidate'] and prob >= threshold:
            simulated_label = 'MIXED'
        else:
            simulated_label = row['pred_label']
        changed = simulated_label != row['pred_label']
        beneficial = changed and row['gold_label'] == 'MIXED'
        harmful = changed and row['gold_label'] != 'MIXED' and row['pred_label'] == row['gold_label']
        neutral = changed and not beneficial and not harmful
        results.append({
            **row, 'model_probability': float(prob), 'simulated_label': simulated_label,
            'changed': changed, 'beneficial': beneficial, 'harmful': harmful, 'neutral': neutral,
        })
    return results


def cascade_summary(cascade_rows):
    gold = [r['gold_label'] for r in cascade_rows]
    original = [r['pred_label'] for r in cascade_rows]
    simulated = [r['simulated_label'] for r in cascade_rows]

    original_report = multiclass_report(gold, original)
    simulated_report = multiclass_report(gold, simulated)

    changes = [r for r in cascade_rows if r['changed']]
    changes_by_original_label = defaultdict(int)
    for r in changes:
        changes_by_original_label[r['pred_label']] += 1

    beneficial = [r for r in changes if r['beneficial']]
    harmful = [r for r in changes if r['harmful']]
    neutral = [r for r in changes if r['neutral']]

    fp_by_gold_label = defaultdict(int)
    for r in changes:
        if r['gold_label'] != 'MIXED':
            fp_by_gold_label[r['gold_label']] += 1

    return {
        'original_multiclass': original_report,
        'simulated_multiclass': simulated_report,
        'n_changes': len(changes),
        'changes_by_original_label': dict(changes_by_original_label),
        'n_uid_to_mixed': changes_by_original_label.get('UID', 0),
        'n_ne_to_mixed': changes_by_original_label.get('NE', 0),
        'n_tr_to_mixed': changes_by_original_label.get('TR', 0),
        'n_beneficial': len(beneficial), 'n_harmful': len(harmful), 'n_neutral': len(neutral),
        'false_positive_changes_by_gold_label': dict(fp_by_gold_label),
    }


# ---------------------------------------------------------------------------
# Candidate-subset secondary analysis
# ---------------------------------------------------------------------------

def candidate_subset_metrics(rows, probs, threshold):
    cand_idx = [i for i, r in enumerate(rows) if r['is_candidate']]
    if not cand_idx:
        return {'n_candidates': 0}
    y_true = np.array([rows[i]['target'] for i in cand_idx])
    y_pred = (probs[cand_idx] >= threshold).astype(int)
    m = binary_metrics(y_true, y_pred)
    m['n_candidates'] = len(cand_idx)
    by_label = defaultdict(lambda: {'n': 0, 'n_mixed_gold': 0})
    for i in cand_idx:
        lab = rows[i]['pred_label']
        by_label[lab]['n'] += 1
        if rows[i]['target'] == 1:
            by_label[lab]['n_mixed_gold'] += 1
    m['by_original_pred_label'] = {k: v for k, v in by_label.items()}
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='Path to dataset.json from build_reranker_dataset.py')
    ap.add_argument('--split-manifest', required=True, help='Path to split_manifest.json')
    ap.add_argument('--out-dir', required=True, help='Output directory for model/metadata/report artifacts')
    ap.add_argument('--threshold-policy', choices=['precision_0.75', 'max_f1'], default='precision_0.75',
                     help='Which dev-selected threshold is treated as the PRIMARY/headline one (Phase 4D). '
                          'Both thresholds are still computed and fully reported regardless -- this only '
                          'controls which one is recorded as metadata["selected_thresholds"]["active"] and '
                          'presented first in the report. Default changed to precision_0.75 in Phase 4D.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/7] Loading dataset...")
    rows, manifest = load_dataset(args.dataset, args.split_manifest)
    train_rows, dev_rows, test_rows = split_rows(rows)
    print(f"    train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")

    y_train, y_dev, y_test = y_of(train_rows), y_of(dev_rows), y_of(test_rows)

    print("[2/7] Baseline 1 (existing rule-based output, ALL scoreable rows)...")
    b1_gold_all = [r['gold_label'] for r in rows]
    b1_pred_all = ['MIXED' if r['pred_label'] == 'MIXED' else 'NOT_MIXED' for r in rows]
    b1_target_all = [1 if r['target'] == 1 else 0 for r in rows]
    b1_pred_binary_all = [1 if r['pred_label'] == 'MIXED' else 0 for r in rows]
    baseline1 = binary_metrics(np.array(b1_target_all), np.array(b1_pred_binary_all))
    baseline1_by_split = {}
    for name, split_rows_ in (('train', train_rows), ('dev', dev_rows), ('test', test_rows)):
        yt = np.array([r['target'] for r in split_rows_])
        yp = np.array([1 if r['pred_label'] == 'MIXED' else 0 for r in split_rows_])
        baseline1_by_split[name] = binary_metrics(yt, yp)
    print(f"    ALL rows: precision={baseline1['precision_mixed']:.4f} recall={baseline1['recall_mixed']:.4f} "
          f"f1={baseline1['f1_mixed']:.4f} support_mixed={baseline1['support_mixed']}")

    print("[3/7] Baseline 2 (candidate rows, always predict KEEP_ORIGINAL)...")
    cand_all = [r for r in rows if r['is_candidate']]
    if cand_all:
        yt = np.array([r['target'] for r in cand_all])
        yp = np.zeros(len(cand_all), dtype=int)  # always predict NOT_MIXED (KEEP_ORIGINAL)
        baseline2 = binary_metrics(yt, yp)
        baseline2['n_candidates'] = len(cand_all)
    else:
        baseline2 = {'n_candidates': 0}
    baseline2_by_split = {}
    for name, split_rows_ in (('train', train_rows), ('dev', dev_rows), ('test', test_rows)):
        c = [r for r in split_rows_ if r['is_candidate']]
        if c:
            yt = np.array([r['target'] for r in c])
            yp = np.zeros(len(c), dtype=int)
            m = binary_metrics(yt, yp)
            m['n_candidates'] = len(c)
        else:
            m = {'n_candidates': 0}
        baseline2_by_split[name] = m
    print(f"    ALL candidates: n={baseline2.get('n_candidates', 0)} "
          f"recall(missed MIXED)={baseline2.get('recall_mixed', float('nan'))}")

    print("[4/7] Fitting vectorizers + models on TRAIN only...")
    feats = build_features(train_rows, dev_rows, test_rows)

    baseline3 = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=MAX_ITER)
    baseline3.fit(feats['tfidf_only']['train'], y_train)
    b3_probs_dev = baseline3.predict_proba(feats['tfidf_only']['dev'])[:, 1]
    b3_table_dev = threshold_table(y_dev, b3_probs_dev)
    b3_best_f1, _, _ = select_thresholds(b3_table_dev)
    b3_pred_dev = (b3_probs_dev >= b3_best_f1['threshold']).astype(int)
    baseline3_dev_metrics = binary_metrics(y_dev, b3_pred_dev)
    baseline3_dev_metrics['selected_threshold'] = b3_best_f1['threshold']
    print(f"    Baseline 3 (char-TFIDF only) dev @F1-threshold {b3_best_f1['threshold']:.2f}: "
          f"P={baseline3_dev_metrics['precision_mixed']:.4f} R={baseline3_dev_metrics['recall_mixed']:.4f} "
          f"F1={baseline3_dev_metrics['f1_mixed']:.4f}")

    full_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=MAX_ITER)
    full_model.fit(feats['combined']['train'], y_train)
    probs_dev = full_model.predict_proba(feats['combined']['dev'])[:, 1]
    print("    Full model (char-TFIDF + structured) fit complete.")

    print(f"[5/7] Threshold selection on DEV ONLY (active policy: {args.threshold_policy})...")
    dev_table = threshold_table(y_dev, probs_dev)
    best_f1_row, best_precision_row, precision_constraint_met = select_thresholds(dev_table)
    print(f"    max-F1 threshold={best_f1_row['threshold']:.2f} "
          f"(P={best_f1_row['precision_mixed']:.4f} R={best_f1_row['recall_mixed']:.4f} F1={best_f1_row['f1_mixed']:.4f})")
    if precision_constraint_met:
        print(f"    best P>=0.75 threshold={best_precision_row['threshold']:.2f} "
              f"(P={best_precision_row['precision_mixed']:.4f} R={best_precision_row['recall_mixed']:.4f})")
    else:
        print(f"    WARNING: no dev threshold reached precision>=0.75; "
              f"reporting highest-precision threshold instead "
              f"(threshold={best_precision_row['threshold']:.2f}, P={best_precision_row['precision_mixed']:.4f})")

    dev_pred_f1 = (probs_dev >= best_f1_row['threshold']).astype(int)
    dev_metrics_f1 = binary_metrics(y_dev, dev_pred_f1)
    dev_ap = float(average_precision_score(y_dev, probs_dev)) if len(set(y_dev)) > 1 else None
    dev_roc = float(roc_auc_score(y_dev, probs_dev)) if len(set(y_dev)) > 1 else None
    dev_prob_dist = prob_distribution(probs_dev)
    dev_cand_metrics = candidate_subset_metrics(dev_rows, probs_dev, best_f1_row['threshold'])

    print("[6/7] TEST evaluation (threshold now frozen; TEST touched for the first time)...")
    probs_test = full_model.predict_proba(feats['combined']['test'])[:, 1]
    test_pred_f1 = (probs_test >= best_f1_row['threshold']).astype(int)
    test_metrics_f1 = binary_metrics(y_test, test_pred_f1)
    test_pred_precision = (probs_test >= best_precision_row['threshold']).astype(int)
    test_metrics_precision = binary_metrics(y_test, test_pred_precision)
    test_ap = float(average_precision_score(y_test, probs_test)) if len(set(y_test)) > 1 else None
    test_roc = float(roc_auc_score(y_test, probs_test)) if len(set(y_test)) > 1 else None
    test_prob_dist = prob_distribution(probs_test)
    test_cand_metrics = candidate_subset_metrics(test_rows, probs_test, best_f1_row['threshold'])
    print(f"    TEST @F1-threshold {best_f1_row['threshold']:.2f}: "
          f"P={test_metrics_f1['precision_mixed']:.4f} R={test_metrics_f1['recall_mixed']:.4f} "
          f"F1={test_metrics_f1['f1_mixed']:.4f}")

    print("[7/7] Cascade simulation on TEST split...")
    cascade_rows_f1 = simulate_cascade(test_rows, full_model, feats['tfidf'], feats['dictvec'], best_f1_row['threshold'])
    cascade_f1 = cascade_summary(cascade_rows_f1)
    cascade_rows_precision = simulate_cascade(test_rows, full_model, feats['tfidf'], feats['dictvec'], best_precision_row['threshold'])
    cascade_precision = cascade_summary(cascade_rows_precision)
    print(f"    [max-F1 threshold] original MIXED F1={cascade_f1['original_multiclass']['per_label'].get('MIXED', {}).get('f1', 0):.4f} "
          f"-> simulated MIXED F1={cascade_f1['simulated_multiclass']['per_label'].get('MIXED', {}).get('f1', 0):.4f}  "
          f"(changes={cascade_f1['n_changes']}, beneficial={cascade_f1['n_beneficial']}, harmful={cascade_f1['n_harmful']})")

    # -----------------------------------------------------------------
    # Write artifacts
    # -----------------------------------------------------------------
    model_path = os.path.join(args.out_dir, 'model.joblib')
    joblib.dump(full_model, model_path)
    vectorizer_path = os.path.join(args.out_dir, 'vectorizer.joblib')
    joblib.dump({'tfidf': feats['tfidf'], 'dictvec': feats['dictvec']}, vectorizer_path)

    dev_thresholds_path = os.path.join(args.out_dir, 'development_thresholds.csv')
    with open(dev_thresholds_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'precision_mixed', 'recall_mixed', 'f1_mixed', 'tp', 'fp', 'fn', 'tn'])
        for r in dev_table:
            w.writerow([r['threshold'], r['precision_mixed'], r['recall_mixed'], r['f1_mixed'],
                        r['tp'], r['fp'], r['fn'], r['tn']])

    test_predictions_path = os.path.join(args.out_dir, 'test_predictions.csv')
    with open(test_predictions_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['block_index', 'position_in_block', 'gold_token_id', 'text', 'gold_label', 'pred_label',
                     'is_candidate', 'candidate_reason', 'model_probability', 'simulated_label_f1_threshold',
                     'changed', 'beneficial', 'harmful', 'neutral'])
        for r in cascade_rows_f1:
            w.writerow([r['block_index'], r['position_in_block'], r['gold_token_id'], r['text'],
                        r['gold_label'], r['pred_label'], r['is_candidate'], r['candidate_reason'],
                        f"{r['model_probability']:.6f}", r['simulated_label'], r['changed'], r['beneficial'],
                        r['harmful'], r['neutral']])

    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'sklearn_version': sklearn.__version__,
        'random_seed': RANDOM_STATE,
        'input_file_hashes': manifest.get('input_file_hashes'),
        'split_block_indices': manifest.get('splits'),
        'feature_configuration': {
            'char_tfidf': {'analyzer': 'char', 'ngram_range': [2, 5], 'lowercase': True},
            'structured_features': sorted(train_rows[0]['structured_features'].keys()) if train_rows else [],
            'candidate_min_stem_len': mr.MIN_STEM_LEN,
            'candidate_tie_break_order': ['longest_stem', 'most_suffix_segments', 'most_morph_features', 'earliest_split_position'],
            'model': {'type': 'LogisticRegression', 'class_weight': 'balanced', 'random_state': RANDOM_STATE, 'max_iter': MAX_ITER},
        },
        'selected_thresholds': {
            'max_f1': best_f1_row['threshold'],
            'best_precision_ge_0.75': best_precision_row['threshold'],
            'precision_0.75_constraint_met_on_dev': precision_constraint_met,
            'policy': args.threshold_policy,
            'active': (best_precision_row['threshold'] if args.threshold_policy == 'precision_0.75'
                       else best_f1_row['threshold']),
        },
        'dev_metrics': {
            'at_max_f1_threshold': dev_metrics_f1, 'average_precision': dev_ap, 'roc_auc': dev_roc,
            'probability_distribution': dev_prob_dist, 'candidate_subset': dev_cand_metrics,
        },
        'test_metrics': {
            'at_max_f1_threshold': test_metrics_f1, 'at_precision_0.75_threshold': test_metrics_precision,
            'average_precision': test_ap, 'roc_auc': test_roc,
            'probability_distribution': test_prob_dist, 'candidate_subset': test_cand_metrics,
        },
        'baseline1_rule_based_all_rows': baseline1,
        'baseline1_by_split': baseline1_by_split,
        'baseline2_keep_original_candidates': baseline2,
        'baseline2_by_split': baseline2_by_split,
        'baseline3_char_tfidf_only_dev': baseline3_dev_metrics,
        'cascade_simulation_max_f1_threshold': cascade_f1,
        'cascade_simulation_precision_0.75_threshold': cascade_precision,
        'cascade_simulation_active_policy': (cascade_precision if args.threshold_policy == 'precision_0.75'
                                              else cascade_f1),
        'known_omissions': [
            'Stanza entity subtype (PERSON/ORG/etc.) omitted -- not available without rerunning the '
            'production NER pipeline; only boolean NE membership (from the predicted Label) is used.',
        ],
        'small_stratum_notes': manifest.get('small_stratum_notes', []),
        'n_cross_split_duplicate_token_forms': manifest.get('n_cross_split_duplicate_forms'),
    }
    metadata_path = os.path.join(args.out_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=1)

    report_path = os.path.join(args.out_dir, 'experiment_report.txt')
    write_report(report_path, manifest, baseline1, baseline1_by_split, baseline2, baseline2_by_split,
                 baseline3_dev_metrics, dev_metrics_f1, dev_ap, dev_roc, dev_prob_dist, dev_cand_metrics,
                 test_metrics_f1, test_metrics_precision, test_ap, test_roc, test_prob_dist, test_cand_metrics,
                 best_f1_row, best_precision_row, precision_constraint_met, cascade_f1, cascade_precision, metadata)

    print(f"\nWrote:\n  {model_path}\n  {vectorizer_path}\n  {metadata_path}\n  {dev_thresholds_path}\n"
          f"  {test_predictions_path}\n  {report_path}")


def write_report(path, manifest, baseline1, baseline1_by_split, baseline2, baseline2_by_split,
                  baseline3_dev, dev_metrics_f1, dev_ap, dev_roc, dev_prob_dist, dev_cand,
                  test_metrics_f1, test_metrics_precision, test_ap, test_roc, test_prob_dist, test_cand,
                  best_f1_row, best_precision_row, precision_constraint_met, cascade_f1, cascade_precision, metadata):
    lines = []
    def p(s=''):
        lines.append(s)

    p('=' * 78)
    p('MIXED RERANKER -- PHASE 2 EXPERIMENT REPORT (isolated, not integrated into production)')
    p('=' * 78)
    p(f"Generated: {metadata['created_at_utc']}")
    p(f"Python {metadata['python_version']}, scikit-learn {metadata['sklearn_version']}, seed={metadata['random_seed']}")

    p('\n--- 1. DATASET ---')
    cps = manifest['counts_per_split']
    p(f"Blocks kept: {manifest['block_counts']['kept']} / {manifest['block_counts']['total']}")
    p(f"Scoreable rows: {manifest['scoreable_row_count']}")
    for name in ('train', 'dev', 'test'):
        c = cps[name]
        p(f"  {name:5}: blocks={c['n_blocks']:4} tokens={c['n_tokens']:5} "
          f"MIXED={c['n_mixed']:4} not_MIXED={c['n_not_mixed']:5} candidates={c['n_candidates']:4} "
          f"(by pred label: {c['candidates_by_pred_label']})")
    p(f"Token forms duplicated across >=2 splits: {manifest['n_cross_split_duplicate_forms']} "
      f"(reported, not removed -- see split_manifest.json 'cross_split_duplicate_token_forms' for the full list)")
    if manifest.get('small_stratum_notes'):
        p("Stratification notes:")
        for n in manifest['small_stratum_notes']:
            p(f"  - {n}")

    p('\n--- 2. BASELINE 1: existing rule-based output (pred Label == MIXED), ALL scoreable rows ---')
    p(f"Precision={baseline1['precision_mixed']:.4f} Recall={baseline1['recall_mixed']:.4f} "
      f"F1={baseline1['f1_mixed']:.4f} support_MIXED={baseline1['support_mixed']} "
      f"support_NOT_MIXED={baseline1['support_not_mixed']}")
    p(f"Confusion matrix [NOT_MIXED,MIXED] x [NOT_MIXED,MIXED]: {baseline1['confusion_matrix']}")
    p("By split:")
    for name in ('train', 'dev', 'test'):
        m = baseline1_by_split[name]
        p(f"  {name:5}: P={m['precision_mixed']:.4f} R={m['recall_mixed']:.4f} F1={m['f1_mixed']:.4f} "
          f"support_MIXED={m['support_mixed']}")

    p('\n--- 3. BASELINE 2: on candidate rows, always predict KEEP_ORIGINAL ---')
    p(f"n_candidates={baseline2.get('n_candidates', 0)}")
    if baseline2.get('n_candidates'):
        p(f"Precision={baseline2['precision_mixed']:.4f} Recall={baseline2['recall_mixed']:.4f} "
          f"F1={baseline2['f1_mixed']:.4f} (recall is necessarily 0 by construction -- this baseline never "
          f"predicts MIXED; it exists to show what fraction of candidate rows are gold-MIXED)")
    p("By split:")
    for name in ('train', 'dev', 'test'):
        m = baseline2_by_split[name]
        p(f"  {name:5}: n_candidates={m.get('n_candidates', 0)} "
          f"support_MIXED={m.get('support_mixed', 'n/a')}")

    p('\n--- 4. BASELINE 3: character-TFIDF-only LogisticRegression (no structured features) ---')
    p(f"DEV @ its own max-F1 threshold ({baseline3_dev['selected_threshold']:.2f}): "
      f"P={baseline3_dev['precision_mixed']:.4f} R={baseline3_dev['recall_mixed']:.4f} F1={baseline3_dev['f1_mixed']:.4f}")

    p('\n--- 5. FULL MODEL (char-TFIDF + structured features) -- DEV SET ---')
    p(f"Average Precision (PR-AUC): {dev_ap}")
    p(f"ROC-AUC: {dev_roc}")
    p(f"Predicted probability distribution: {dev_prob_dist}")
    p(f"Max-F1 threshold: {best_f1_row['threshold']:.2f} -> "
      f"P={dev_metrics_f1['precision_mixed']:.4f} R={dev_metrics_f1['recall_mixed']:.4f} F1={dev_metrics_f1['f1_mixed']:.4f}")
    p(f"Confusion matrix [NOT_MIXED,MIXED] x [NOT_MIXED,MIXED]: {dev_metrics_f1['confusion_matrix']}")
    if precision_constraint_met:
        p(f"Best threshold with precision>=0.75: {best_precision_row['threshold']:.2f} -> "
          f"P={best_precision_row['precision_mixed']:.4f} R={best_precision_row['recall_mixed']:.4f} "
          f"F1={best_precision_row['f1_mixed']:.4f}")
    else:
        p(f"WARNING: no threshold on DEV reached precision>=0.75. Highest-precision threshold available: "
          f"{best_precision_row['threshold']:.2f} (P={best_precision_row['precision_mixed']:.4f}, "
          f"R={best_precision_row['recall_mixed']:.4f}) -- reported as a fallback, NOT a constraint-satisfying value.")
    p(f"Full threshold sweep: see development_thresholds.csv")
    p(f"Candidate-subset secondary analysis (DEV): {dev_cand}")

    p('\n--- 6. FULL MODEL -- TEST SET (evaluated once, after all DEV-only decisions were frozen) ---')
    p(f"Average Precision (PR-AUC): {test_ap}")
    p(f"ROC-AUC: {test_roc}")
    p(f"Predicted probability distribution: {test_prob_dist}")
    p(f"At max-F1 (dev-selected) threshold {best_f1_row['threshold']:.2f}: "
      f"P={test_metrics_f1['precision_mixed']:.4f} R={test_metrics_f1['recall_mixed']:.4f} F1={test_metrics_f1['f1_mixed']:.4f}")
    p(f"Confusion matrix [NOT_MIXED,MIXED] x [NOT_MIXED,MIXED]: {test_metrics_f1['confusion_matrix']}")
    p(f"At precision>=0.75 (dev-selected) threshold {best_precision_row['threshold']:.2f}: "
      f"P={test_metrics_precision['precision_mixed']:.4f} R={test_metrics_precision['recall_mixed']:.4f} "
      f"F1={test_metrics_precision['f1_mixed']:.4f}")
    p(f"Candidate-subset secondary analysis (TEST): {test_cand}")

    p('\n--- 7. CASCADE SIMULATION ON TEST SPLIT (non-candidates unchanged; candidates flip to MIXED only '
      'if model P(MIXED) >= threshold, else KEEP_ORIGINAL) ---')
    for label, cs in (('max-F1 threshold', cascade_f1), ('precision>=0.75 threshold', cascade_precision)):
        p(f"\n  [{label}]")
        orig, sim = cs['original_multiclass'], cs['simulated_multiclass']
        p(f"    Overall Label accuracy:  original={orig['accuracy']:.4f}  ->  simulated={sim['accuracy']:.4f}")
        p(f"    Macro F1:                original={orig['macro_f1']:.4f}  ->  simulated={sim['macro_f1']:.4f}")
        p(f"    Weighted F1:             original={orig['weighted_f1']:.4f}  ->  simulated={sim['weighted_f1']:.4f}")
        om = orig['per_label'].get('MIXED', {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0})
        sm = sim['per_label'].get('MIXED', {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0})
        p(f"    MIXED precision:         original={om['precision']:.4f}  ->  simulated={sm['precision']:.4f}")
        p(f"    MIXED recall:            original={om['recall']:.4f}  ->  simulated={sm['recall']:.4f}")
        p(f"    MIXED F1:                original={om['f1']:.4f}  ->  simulated={sm['f1']:.4f}")
        p(f"    Changes: {cs['n_changes']} total "
          f"(UID->MIXED={cs['n_uid_to_mixed']}, NE->MIXED={cs['n_ne_to_mixed']}, TR->MIXED={cs['n_tr_to_mixed']})")
        p(f"    Beneficial (fixed a real MIXED): {cs['n_beneficial']}")
        p(f"    Harmful (broke a previously-correct prediction): {cs['n_harmful']}")
        p(f"    Neutral (was already wrong, still wrong after): {cs['n_neutral']}")
        p(f"    False-positive changes by gold label: {cs['false_positive_changes_by_gold_label']}")

    p('\n--- 8. KNOWN OMISSIONS / LIMITATIONS ---')
    for note in metadata['known_omissions']:
        p(f"  - {note}")
    p("  - Candidate selection tie-break policy (longest-stem-first) differs from the production "
      "MIXED detector's own longest-suffix-first walk in Annotator._detect_mixed_no_apostrophe -- "
      "documented, not reconciled, per Phase 2 instructions.")
    p("  - This report does NOT constitute a claim of production improvement on its own -- section 7 "
      "(cascade simulation) is the only section that reflects the intended production behaviour end to end; "
      "sections 4-6 describe the binary classifier in isolation.")

    report = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
