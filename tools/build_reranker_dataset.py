#!/usr/bin/env python
# tools/build_reranker_dataset.py
"""Phase 2, isolated experiment: build a training dataset for the MIXED
reranker from a gold (manually corrected) CSV and a machine-output CSV.

Read-only with respect to the corpus: never writes to --gold, --pred,
--exclusions, or --segmentation-mismatches. All external corpus paths are
CLI arguments -- nothing personal or machine-specific is hardcoded here.

Alignment policy (see CLAUDE.md-adjacent Phase 2 preflight report for the
full rationale):

  Tier 1 -- GLOBAL structural pre-check (fatal, aborts the whole build if it
  fails): both files must have the same line count, the same blank-row
  (block-separator) positions, and the same meta-row-family positions
  (SentenceID / MatrixLang / EmbedLang). If this fails, the very premise of
  positional block-splitting is broken and no block-local patch is safe --
  the script stops and writes nothing.

  Tier 2 -- PER-BLOCK checks (local, non-fatal -- a failing block is
  excluded and logged, the rest of the build proceeds):
    - both blocks have the same number of real token rows
    - token Item text matches at every non-excluded position, after ONLY
      the explicitly approved cosmetic normalization:
        * strip leading/trailing whitespace
        * gold Item == "REDACTED" (exact case) matches any predicted Item
        * rows already listed in --segmentation-mismatches are excluded
          BEFORE the text-match check runs (no separate markup-stripping
          normalization is applied/re-derived here)
      No lowercasing, apostrophe, or Unicode normalization is applied.

Global Token IDs (the gold CSV's "Token"/idx column) and SentenceID values
are never used to align gold to pred -- alignment is by sentence-block
index (reconstructed from blank separator rows) and within-block token
position only. Token IDs are used in a narrower, safe way: to locate which
row a given --exclusions/--segmentation-mismatches report entry refers to
*within the gold file's own row order* (a stable, file-local identifier for
this fixed corpus snapshot), not to join gold to pred.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs_pipeline import Annotator, DEFAULTS
import mixed_reranker as mr

HEADER = ['Token', 'Item', 'Label', 'Gloss']
META_ITEMS = {'SentenceID', 'MatrixLang', 'EmbedLang'}
VALID_LABELS = set(mr.SCHEMA_LABELS)


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"FATAL: {path} is empty")
    header, body = rows[0], rows[1:]
    if header != HEADER:
        raise SystemExit(f"FATAL: unexpected header in {path}: {header} (expected {HEADER})")
    return body


def classify(row):
    token = row[0] if len(row) > 0 else ''
    item = row[1] if len(row) > 1 else ''
    label = row[2] if len(row) > 2 else ''
    gloss = row[3] if len(row) > 3 else ''
    if token == '' and item == '' and label == '' and gloss.strip() == '':
        return 'blank'
    if token == '' and item in META_ITEMS:
        return 'meta:' + item
    if token != '':
        return 'token'
    raise SystemExit(f"FATAL: unclassifiable row: {row!r}")


def global_structural_check(gold, pred, gold_path, pred_path):
    if len(gold) != len(pred):
        raise SystemExit(f"FATAL: line count mismatch: gold={len(gold)} ({gold_path}) "
                          f"pred={len(pred)} ({pred_path}). Cannot proceed with block alignment.")
    gold_kinds = [classify(r) for r in gold]
    pred_kinds = [classify(r) for r in pred]

    blank_gold = [i for i, k in enumerate(gold_kinds) if k == 'blank']
    blank_pred = [i for i, k in enumerate(pred_kinds) if k == 'blank']
    if blank_gold != blank_pred:
        raise SystemExit("FATAL: blank-row (block separator) positions differ between gold and pred. "
                          "Positional block alignment is not safe; aborting without writing any output.")

    for meta_name in ('meta:SentenceID', 'meta:MatrixLang', 'meta:EmbedLang'):
        gpos = [i for i, k in enumerate(gold_kinds) if k == meta_name]
        ppos = [i for i, k in enumerate(pred_kinds) if k == meta_name]
        if gpos != ppos:
            raise SystemExit(f"FATAL: {meta_name} row positions differ between gold and pred. "
                              "Meta-row structure is not valid for alignment; aborting.")

    tok_gold = [i for i, k in enumerate(gold_kinds) if k == 'token']
    tok_pred = [i for i, k in enumerate(pred_kinds) if k == 'token']
    if tok_gold != tok_pred:
        raise SystemExit("FATAL: token-row positions differ between gold and pred; aborting.")

    return gold_kinds, pred_kinds


def build_blocks(gold, pred, gold_kinds, pred_kinds):
    """Split into blocks using ONLY blank-row/meta-row structure (already
    verified identical between gold and pred by global_structural_check).
    Token IDs are read from gold here for later same-file lookups only."""
    blocks = []
    cur = None
    for i, gk in enumerate(gold_kinds):
        if gk == 'meta:SentenceID':
            cur = {
                'block_index': len(blocks),  # 0-based internally
                'gold_sentence_id': gold[i][2],
                'pred_sentence_id': pred[i][2],
                'gold_matrixlang': None, 'pred_matrixlang': None,
                'gold_embedlang': None, 'pred_embedlang': None,
                'token_rows': [],
            }
            blocks.append(cur)
        elif gk == 'meta:MatrixLang':
            cur['gold_matrixlang'] = gold[i][2]
            cur['pred_matrixlang'] = pred[i][2]
        elif gk == 'meta:EmbedLang':
            cur['gold_embedlang'] = gold[i][2]
            cur['pred_embedlang'] = pred[i][2]
        elif gk == 'token':
            g, p = gold[i], pred[i]
            cur['token_rows'].append({
                'position_in_block': len(cur['token_rows']),
                'gold_token_id': int(g[0]),
                'gold_item': g[1], 'gold_label': g[2], 'gold_gloss': g[3] if len(g) > 3 else '',
                'pred_item': p[1], 'pred_label': p[2], 'pred_gloss': p[3] if len(p) > 3 else '',
            })
    return blocks


def load_exclusions(path):
    """Returns (excluded_token_ids: set[int], rows: list[dict]) from the
    existing exclusions.csv. Rows with an empty token_id are informational
    only (do not exclude any token row) -- kept in `rows` for logging."""
    excluded_ids = set()
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
            tid = (row.get('token_id') or '').strip()
            if tid:
                excluded_ids.add(int(tid))
    return excluded_ids, rows


def load_segmentation_mismatches(path):
    """Returns (excluded_token_ids: set[int], events: list[dict],
    orphan_block_indices: set[int]).

    block_index in this report is 1-based (matches the existing
    eval_results/evaluate.py convention); converted to 0-based here to
    match this script's internal block_index.

    An event with an empty gold_parent_token_id means the existing report
    could not safely identify the parent side of the span -- per Decision
    2, that event's whole block must be excluded instead of just its rows.
    """
    excluded_ids = set()
    events = []
    orphan_block_indices = set()
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            events.append(row)
            block_idx_0based = int(row['block_index']) - 1
            parent_id = (row.get('gold_parent_token_id') or '').strip()
            echo_id = (row.get('gold_echo_token_id') or '').strip()
            if not parent_id or not echo_id:
                orphan_block_indices.add(block_idx_0based)
                continue
            excluded_ids.add(int(parent_id))
            excluded_ids.add(int(echo_id))
    return excluded_ids, events, orphan_block_indices


def check_and_collect(blocks, known_excluded_ids, orphan_block_indices):
    """Tier 2 per-block checks. Returns (scoreable_rows, excluded_blocks_log,
    excluded_rows_log)."""
    scoreable_rows = []
    excluded_blocks_log = []
    excluded_rows_log = []

    for b in blocks:
        gt = b['token_rows']
        block_idx = b['block_index']

        if block_idx in orphan_block_indices:
            excluded_blocks_log.append({
                'scope': 'block', 'block_index': block_idx, 'token_id': '', 'item': '',
                'reason': 'segmentation-mismatch report could not identify both sides of the affected '
                          'span (orphan wrapped row) for this block; excluding entire block per Decision 2 fallback',
            })
            continue

        # Per-block "equal token-row count" check (Decision 1): by
        # construction this always holds here, because build_blocks() only
        # ever produces one token-row list per block index, populated from
        # gold[i] and pred[i] at the SAME line i -- and Tier 1 already
        # asserted gold/pred token-row line positions are identical. There
        # is no way for gt to represent an unequal-count situation at this
        # point; the count guarantee lives in global_structural_check, not
        # here. Asserted anyway, defensively, rather than assumed silently.
        assert len(gt) == len(b['token_rows']), "internal invariant violated: token_rows built inconsistently"

        local_ok = True
        local_reason = None
        for tr in gt:
            gold_id = tr['gold_token_id']
            if gold_id in known_excluded_ids:
                excluded_rows_log.append({
                    'scope': 'row', 'block_index': block_idx, 'token_id': gold_id,
                    'item': tr['gold_item'],
                    'reason': 'listed in exclusions.csv or segmentation_mismatches.csv',
                })
                continue

            gold_item = tr['gold_item'].strip()
            pred_item = tr['pred_item'].strip()
            if gold_item == pred_item:
                pass
            elif gold_item == 'REDACTED':
                pass  # approved exception: REDACTED matches any predicted Item
            else:
                local_ok = False
                local_reason = (f"unexplained Item text mismatch at position {tr['position_in_block']} "
                                 f"(token_id={gold_id}): gold={tr['gold_item']!r} pred={tr['pred_item']!r}")
                break

        if not local_ok:
            excluded_blocks_log.append({
                'scope': 'block', 'block_index': block_idx, 'token_id': '', 'item': '', 'reason': local_reason,
            })
            continue

        for tr in gt:
            gold_id = tr['gold_token_id']
            if gold_id in known_excluded_ids:
                continue
            if tr['gold_label'] not in VALID_LABELS:
                # should be unreachable: corrupted-label rows are always in
                # exclusions.csv and therefore already skipped above, but
                # assert defensively rather than silently mis-scoring.
                raise SystemExit(f"FATAL: token_id={gold_id} has gold Label {tr['gold_label']!r} not in the "
                                  f"7-label schema and is NOT listed in --exclusions. Refusing to guess; "
                                  f"add it to exclusions.csv or investigate.")
            scoreable_rows.append({
                'block_index': block_idx,
                'position_in_block': tr['position_in_block'],
                'gold_token_id': gold_id,
                'gold_item': tr['gold_item'],
                'gold_label': tr['gold_label'],
                'pred_item': tr['pred_item'],
                'pred_label': tr['pred_label'],
                'is_redacted': tr['gold_item'].strip() == 'REDACTED',
                'target': 1 if tr['gold_label'] == 'MIXED' else 0,
            })

    return scoreable_rows, excluded_blocks_log, excluded_rows_log


def mixed_bucket(mixed_count):
    if mixed_count == 0:
        return 'none'
    if mixed_count == 1:
        return 'one'
    return 'multi'


def grouped_stratified_split(blocks, scoreable_rows, seed, train_frac, dev_frac, test_frac):
    """Deterministic, group-level (block) split, stratified by each block's
    count of gold-MIXED scoreable rows (bucketed: none / one / multi), so
    MIXED representation is preserved in every split. Never splits a block
    across two partitions.

    Allocation within each stratum: seeded shuffle of that stratum's block
    indices, then a proportional round-robin cut at train_frac/dev_frac
    boundaries. For strata too small to hit all three splits (e.g. a single
    block), blocks fall through to train first, then dev, then test, and
    this is recorded in the returned diagnostics so it's visible rather
    than silently invisible.
    """
    mixed_count_by_block = defaultdict(int)
    for r in scoreable_rows:
        if r['target'] == 1:
            mixed_count_by_block[r['block_index']] += 1

    kept_block_indices = sorted({b['block_index'] for b in blocks if b.get('_kept', True)})

    strata = defaultdict(list)
    for bidx in kept_block_indices:
        strata[mixed_bucket(mixed_count_by_block[bidx])].append(bidx)

    rng = random.Random(seed)
    train_ids, dev_ids, test_ids = [], [], []
    small_stratum_notes = []

    for bucket_name in sorted(strata.keys()):
        ids = sorted(strata[bucket_name])
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            # not enough blocks in this stratum to populate all three
            # splits; assign train-first so at least training data isn't
            # starved, and note the shortfall explicitly.
            for i, bidx in enumerate(ids):
                (train_ids if i == 0 else dev_ids if i == 1 else test_ids).append(bidx)
            small_stratum_notes.append(
                f"stratum '{bucket_name}' has only {n} block(s); could not populate all three splits "
                f"proportionally (train-first fallback used)")
            continue
        n_test = max(1, round(n * test_frac))
        n_dev = max(1, round(n * dev_frac))
        n_test = min(n_test, n - 2)
        n_dev = min(n_dev, n - n_test - 1)
        n_train = n - n_test - n_dev
        train_ids.extend(ids[:n_train])
        dev_ids.extend(ids[n_train:n_train + n_dev])
        test_ids.extend(ids[n_train + n_dev:])

    return {
        'train': sorted(train_ids),
        'dev': sorted(dev_ids),
        'test': sorted(test_ids),
        'strata': {k: sorted(v) for k, v in strata.items()},
        'small_stratum_notes': small_stratum_notes,
        'mixed_count_by_block': dict(mixed_count_by_block),
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', required=True, help='Path to manually corrected gold CSV')
    ap.add_argument('--pred', required=True, help='Path to machine-output CSV')
    ap.add_argument('--exclusions', required=True, help='Path to existing exclusions.csv')
    ap.add_argument('--segmentation-mismatches', required=True, help='Path to existing segmentation_mismatches.csv')
    ap.add_argument('--resources-dir', required=True,
                     help='Path to a directory containing frequent_tr_words.txt, frequent_en_words.txt, lid.176.ftz '
                          '(normally the repo\'s resources/ directory)')
    ap.add_argument('--out-dir', required=True, help='Output directory for dataset.json, split_manifest.json, '
                                                       'excluded_blocks.csv (created if missing)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-frac', type=float, default=0.70)
    ap.add_argument('--dev-frac', type=float, default=0.15)
    ap.add_argument('--test-frac', type=float, default=0.15)
    ap.add_argument('--candidate-strategy', choices=list(mr.CANDIDATE_STRATEGIES),
                     default=mr.DEFAULT_CANDIDATE_STRATEGY,
                     help='Tie-break policy for selecting among multiple plausible stem/suffix analyses '
                          'of the same token (Phase 2). Does not affect the block/token alignment or the '
                          'train/dev/test split, which are computed from gold labels only.')
    ap.add_argument('--verbal-morphology-level', choices=list(mr.VERBAL_MORPHOLOGY_LEVELS),
                     default=mr.DEFAULT_VERBAL_MORPHOLOGY_LEVEL,
                     help='Which experimental verbal-suffix categories are active in the candidate '
                          'generator (Phase 4A/4B/4C-1). Default is phase4c1 (promoted in Phase 4D -- it '
                          'matched or beat Phase 4A on the precision>=0.75 threshold policy). Phase 4A and '
                          '4B remain fully available and selectable via this flag.')
    ap.add_argument('--allow-informal-orthography', action='store_true',
                     help='Phase 4C-2: enable the experimental informal-orthography fallback (s->ş, i->ı) '
                          'for verbal-suffix matching only, used only when the standard spelling fails to '
                          'match. Off by default. Never affects the token/stem text itself or exported data.')
    ap.add_argument('--allow-stem-orthographic-recovery', action='store_true',
                     help='Phase 4D: enable the experimental duplicated-consonant English-lexicon recovery '
                          'fallback (e.g. triger->trigger) for TR-bucket candidacy, used only when the suffix '
                          'analysis was informally-normalized (Phase 4C-2) and ordinary lexicon/fastText '
                          'evidence both failed. Off by default. Never modifies the stem/token text itself.')
    ap.add_argument('--include-batch-a-features', action='store_true',
                     help='[ACTIVE BASELINE, Phase 5B] Include Batch A structured features (parser metadata: '
                          'analysis_source, candidate_reason, is_candidate, split_position_ratio). Part of the '
                          'current active experimental baseline (Phase 5F: Batch A + C + pruned G). Off by default.')
    ap.add_argument('--exclude-batch-c-features', action='store_true',
                     help='[ACTIVE BASELINE, Phase 5A] Batch C features (confidence interaction: ft_prob_delta, '
                          'ft_lang_agreement, stem_evidence_strength) are part of the current active experimental '
                          'baseline and are therefore included by default. Pass this flag to exclude them for an '
                          'isolated ablation of some other batch, without deleting the implementation.')
    ap.add_argument('--include-batch-b-features', action='store_true',
                     help='[EXPERIMENTAL ONLY -- REJECTED, Phase 5D] Include Batch B morphological-complexity '
                          'features (morph_tag_count, has_case, has_plural, has_possessive, '
                          'has_derivational_suffix, has_verbal_morphology, morph_complexity). Evaluated and '
                          'rejected: no active-policy cascade improvement over the Batch A+C baseline, plus a '
                          'coefficient sign-flip on the existing suffix_segment_count feature indicating '
                          'redundancy. NOT part of the active baseline -- kept available, off by default, only '
                          'for reproducing that experiment.')
    ap.add_argument('--include-batch-g-features', action='store_true',
                     help='[ACTIVE BASELINE, Phase 5E/5F] Include (pruned) Batch G candidate-analysis ambiguity / '
                          'selection-process features (analysis_candidate_count, selection_is_unique, '
                          'has_nominal_verbal_competition; distinct_stem_count was removed in Phase 5F as '
                          'structurally redundant with analysis_candidate_count). Part of the current active '
                          'experimental baseline (Phase 5F). Reuses the candidate list '
                          'classify_candidate(return_candidates=True) already enumerates internally -- no '
                          'second enumeration call. Off by default.')
    ap.add_argument('--include-batch-d-features', action='store_true',
                     help='[EXPERIMENTAL ONLY -- REJECTED, Phase 5G] Include Batch D English-stem-quality '
                          'features (stem_english_confidence, stem_turkish_confidence, stem_lexicon_contrast). '
                          'Computed entirely from existing lexicon/fastText fields already in the feature dict '
                          '-- no new lookup. Evaluated and rejected: introduced two new neutral misclassifications '
                          'at the active threshold policy with a net F1 regression versus the Phase 5F baseline. '
                          'NOT part of the active baseline -- kept available, off by default, only for '
                          'reproducing that experiment.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/6] Loading CSVs...")
    gold = load_csv(args.gold)
    pred = load_csv(args.pred)

    print(f"[2/6] Tier-1 global structural check...")
    gold_kinds, pred_kinds = global_structural_check(gold, pred, args.gold, args.pred)
    blocks = build_blocks(gold, pred, gold_kinds, pred_kinds)
    print(f"    {len(blocks)} sentence blocks, "
          f"{sum(len(b['token_rows']) for b in blocks)} total token rows (before exclusions)")

    print(f"[3/6] Loading known exclusions / segmentation mismatches...")
    excl_ids, excl_rows = load_exclusions(args.exclusions)
    seg_ids, seg_events, orphan_blocks = load_segmentation_mismatches(args.segmentation_mismatches)
    known_excluded_ids = excl_ids | seg_ids
    print(f"    exclusions.csv: {len(excl_ids)} token-row exclusion(s), {len(excl_rows)} total rows (incl. informational)")
    print(f"    segmentation_mismatches.csv: {len(seg_events)} events, {len(seg_ids)} token-row exclusion(s), "
          f"{len(orphan_blocks)} orphan block(s)")

    print(f"[4/6] Tier-2 per-block checks...")
    scoreable_rows, excluded_blocks_log, excluded_rows_log = check_and_collect(blocks, known_excluded_ids, orphan_blocks)
    excluded_block_indices = {e['block_index'] for e in excluded_blocks_log}
    for b in blocks:
        b['_kept'] = b['block_index'] not in excluded_block_indices
    kept_blocks = [b for b in blocks if b['_kept']]
    print(f"    kept blocks: {len(kept_blocks)} / {len(blocks)}")
    print(f"    scoreable token rows: {len(scoreable_rows)}")
    print(f"    row-level exclusions applied: {len(excluded_rows_log)}")

    print(f"[5/6] Candidate generation + feature extraction (strategy={args.candidate_strategy}, "
          f"verbal_level={args.verbal_morphology_level}, "
          f"allow_informal_orthography={args.allow_informal_orthography}, "
          f"allow_stem_orthographic_recovery={args.allow_stem_orthographic_recovery}, "
          f"loads Annotator/fastText once)...")
    annotator = Annotator(
        freq_tr=os.path.join(args.resources_dir, 'frequent_tr_words.txt'),
        freq_en=os.path.join(args.resources_dir, 'frequent_en_words.txt'),
        ft_path=os.path.join(args.resources_dir, 'lid.176.ftz'),
    )
    cfg = DEFAULTS.copy()

    candidate_counts_by_pred_label = defaultdict(int)
    for row in scoreable_rows:
        # Phase 5F: return_candidates=True makes classify_candidate hand back
        # the full enumerated candidate list it already computed internally
        # (empty for pred labels in mr.NON_CANDIDATE_LABELS, where it never
        # enumerates at all), so Batch G bookkeeping below does not need a
        # second enumerate_candidate_analyses call. is_candidate/reason/
        # analysis are unaffected -- see classify_candidate's docstring.
        result = mr.classify_candidate(row['pred_label'], row['pred_item'], annotator, cfg,
                                        strategy=args.candidate_strategy,
                                        verbal_level=args.verbal_morphology_level,
                                        allow_informal_orthography=args.allow_informal_orthography,
                                        allow_stem_orthographic_recovery=args.allow_stem_orthographic_recovery,
                                        return_candidates=args.include_batch_g_features)
        if args.include_batch_g_features:
            is_cand, reason, analysis, candidate_analyses = result
        else:
            is_cand, reason, analysis = result
            candidate_analyses = None
        row['is_candidate'] = is_cand
        row['candidate_reason'] = reason
        if is_cand:
            candidate_counts_by_pred_label[row['pred_label']] += 1

        row['structured_features'] = mr.build_structured_feature_dict(
            row['pred_item'], row['pred_label'], analysis, annotator, cfg,
            is_candidate=is_cand, candidate_reason=reason,
            include_batch_a=args.include_batch_a_features,
            include_batch_c=not args.exclude_batch_c_features,
            include_batch_b=args.include_batch_b_features,
            include_batch_g=args.include_batch_g_features,
            candidate_analyses=candidate_analyses,
            candidate_strategy=args.candidate_strategy,
            include_batch_d=args.include_batch_d_features)
        row['text'] = row['pred_item']
        # Phase 4D: carried at the row level (not as a model feature) purely
        # for inspection/reporting -- the recovered lexicon string itself,
        # when the duplicated-consonant fallback fired.
        row['extracted_stem'] = analysis.stem if analysis is not None else None
        row['recovered_english_stem'] = analysis.recovered_english_stem if analysis is not None else None
    n_candidates = sum(1 for r in scoreable_rows if r['is_candidate'])
    print(f"    candidates: {n_candidates} / {len(scoreable_rows)}  "
          f"(by pred label: {dict(candidate_counts_by_pred_label)})")

    print(f"[6/6] Grouped stratified split + writing artifacts...")
    split = grouped_stratified_split(blocks, scoreable_rows, args.seed, args.train_frac, args.dev_frac, args.test_frac)
    block_to_split = {}
    for name in ('train', 'dev', 'test'):
        for bidx in split[name]:
            block_to_split[bidx] = name
    for row in scoreable_rows:
        row['split'] = block_to_split.get(row['block_index'], 'UNASSIGNED')
    unassigned = [r for r in scoreable_rows if r['split'] == 'UNASSIGNED']
    if unassigned:
        raise SystemExit(f"FATAL: {len(unassigned)} scoreable rows belong to a block that was not assigned "
                          f"to any split -- this should be impossible given kept_blocks == union of split lists.")

    # duplicate token-form diagnostics across splits (exact pred Item text,
    # no normalization -- reported, not deduplicated, per Decision 7)
    forms_by_split = defaultdict(lambda: defaultdict(int))
    for row in scoreable_rows:
        forms_by_split[row['text']][row['split']] += 1
    cross_split_forms = {
        form: dict(counts) for form, counts in forms_by_split.items()
        if len([s for s, c in counts.items() if c > 0]) > 1
    }

    counts_per_split = {}
    for name in ('train', 'dev', 'test'):
        rows_in_split = [r for r in scoreable_rows if r['split'] == name]
        by_pred_label = defaultdict(int)
        for r in rows_in_split:
            if r['is_candidate']:
                by_pred_label[r['pred_label']] += 1
        counts_per_split[name] = {
            'n_blocks': len(split[name]),
            'n_tokens': len(rows_in_split),
            'n_mixed': sum(1 for r in rows_in_split if r['target'] == 1),
            'n_not_mixed': sum(1 for r in rows_in_split if r['target'] == 0),
            'n_candidates': sum(1 for r in rows_in_split if r['is_candidate']),
            'candidates_by_pred_label': dict(by_pred_label),
        }

    dataset_path = os.path.join(args.out_dir, 'dataset.json')
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(scoreable_rows, f, ensure_ascii=False, indent=1)

    manifest = {
        'seed': args.seed,
        'candidate_strategy': args.candidate_strategy,
        'verbal_morphology_level': args.verbal_morphology_level,
        'allow_informal_orthography': args.allow_informal_orthography,
        'allow_stem_orthographic_recovery': args.allow_stem_orthographic_recovery,
        'include_batch_a_features': args.include_batch_a_features,
        'include_batch_c_features': not args.exclude_batch_c_features,
        'include_batch_b_features': args.include_batch_b_features,
        'include_batch_g_features': args.include_batch_g_features,
        'include_batch_d_features': args.include_batch_d_features,
        'train_frac': args.train_frac, 'dev_frac': args.dev_frac, 'test_frac': args.test_frac,
        'stratification': 'block-level, bucketed by count of gold-MIXED scoreable rows in the block '
                           '(none / one / multi); see mixed_bucket() in this script',
        'splits': {k: split[k] for k in ('train', 'dev', 'test')},
        'strata': split['strata'],
        'small_stratum_notes': split['small_stratum_notes'],
        'counts_per_split': counts_per_split,
        'cross_split_duplicate_token_forms': cross_split_forms,
        'n_cross_split_duplicate_forms': len(cross_split_forms),
        'input_file_hashes': {
            'gold': sha256_of(args.gold),
            'pred': sha256_of(args.pred),
            'exclusions': sha256_of(args.exclusions),
            'segmentation_mismatches': sha256_of(args.segmentation_mismatches),
        },
        'block_counts': {'total': len(blocks), 'kept': len(kept_blocks), 'excluded': len(excluded_blocks_log)},
        'scoreable_row_count': len(scoreable_rows),
    }
    manifest_path = os.path.join(args.out_dir, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    excluded_path = os.path.join(args.out_dir, 'excluded_blocks.csv')
    with open(excluded_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scope', 'block_index', 'token_id', 'item', 'reason'])
        for e in excluded_blocks_log:
            w.writerow([e['scope'], e['block_index'], e['token_id'], e['item'], e['reason']])
        for e in excluded_rows_log:
            w.writerow([e['scope'], e['block_index'], e['token_id'], e['item'], e['reason']])
        for r in excl_rows:
            if not (r.get('token_id') or '').strip():
                w.writerow(['informational', '', '', r.get('item', ''), r.get('reason', '')])

    print(f"\nWrote:\n  {dataset_path}\n  {manifest_path}\n  {excluded_path}")
    print(f"\nSummary: {len(kept_blocks)}/{len(blocks)} blocks kept, {len(scoreable_rows)} scoreable rows, "
          f"{n_candidates} candidates, {len(cross_split_forms)} token forms duplicated across splits.")
    if split['small_stratum_notes']:
        print("Notes:")
        for n in split['small_stratum_notes']:
            print(f"  - {n}")


if __name__ == '__main__':
    main()
