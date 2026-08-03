# annotation_model.py
"""Pure annotation-state transformation functions, extracted from
cs_annotator_app.py (Stage 2.1). No tkinter import — callable and testable
independent of a running GUI."""

import re


def is_meta_row_token(tok: str) -> bool:
    """Rows that should NOT be counted as tokens for numbering."""
    if tok is None:
        return False
    t = str(tok).strip()
    if not t:
        # blank rows are structural/editable placeholders
        return True
    if t in ("MatrixLang", "EmbedLang"):
        return True
    if t.lower() in ("sentenceid", "sentid", "sentence_id", "sent_id"):
        return True
    return False


def freq_normalize_token(tok: str):
    """Normalize token for frequency counting.
    - Casefold ON
    - Keep hyphenated forms (umbrella-ya stays)
    - Strip leading/trailing punctuation
    """
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    # strip punctuation at edges only
    s = re.sub(r"^[\W_]+|[\W__]+$", "", s, flags=re.UNICODE)
    if not s:
        return None
    return s.casefold()


def compute_word_frequencies(blocks, allowed_labels=None):
    """Compute word frequencies from annotated blocks.
    Returns:
        freq: dict[token] -> total count
        by_label: dict[token] -> {label: count}
        total_tokens: int
    Meta rows (MatrixLang, EmbedLang, SentenceID, blanks) are excluded.
    """
    freq = {}
    by_label = {}
    total = 0

    for blk in blocks or []:
        for r in blk:
            tok = r.get('token', '')
            if is_meta_row_token(tok):
                continue
            norm = freq_normalize_token(tok)
            if not norm:
                continue
            lab = str(r.get('label', '') or '').strip()
            if allowed_labels is not None and lab not in allowed_labels:
                continue

            total += 1
            freq[norm] = freq.get(norm, 0) + 1
            if norm not in by_label:
                by_label[norm] = {}
            if lab:
                by_label[norm][lab] = by_label[norm].get(lab, 0) + 1

    return freq, by_label, total


def sheet_rows_to_txt(rows, headers):
    """Convert grid rows to TXT exactly like the UI: tab-separated rows.
    Blank grid rows become blank lines (block separators).

    Behavior:
    - If idx is blank (meta rows), omit the idx column in the output.
    - Trailing empty fields are removed.
    """
    out_lines = []
    for rr in rows:
        if rr is None:
            out_lines.append("")
            continue

        r = ["" if v is None else str(v) for v in rr]
        ncol = len(headers)
        while len(r) < ncol:
            r.append("")
        r = r[:ncol]

        if all(x.strip() == "" for x in r):
            out_lines.append("")
            continue

        idx = r[0].strip()
        fields = r[1:] if idx == "" else r

        while fields and fields[-1].strip() == "":
            fields.pop()

        out_lines.append("\t".join(fields))

    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    return "\n".join(out_lines)


def renumber_tokens(blocks):
    """Assign sequential token ids only to non-meta rows."""
    g = 1
    for rows in blocks:
        for r in rows:
            tok = r.get("token", "")
            if is_meta_row_token(tok):
                r["idx"] = ""
                continue
            r["idx"] = g
            g += 1


def reconstruct_text_from_blocks(blocks, extra_headers):
    """Fallback TXT reconstruction from the Python model.

    Includes extra user-defined columns. For meta rows (blank idx), idx is omitted.
    Trailing empty fields are trimmed.
    """
    renumber_tokens(blocks)
    out_blocks = []
    for rows in blocks:
        lines = []
        for r in rows:
            idx = str(r.get('idx', '') or '').strip()
            tok = str(r.get('token', '') or '')
            lab = str(r.get('label', '') or '')
            glo = str(r.get('gloss', '') or '')

            if not tok:
                continue

            extras = [str(r.get(h, '') or '') for h in extra_headers]

            if idx == "":
                fields = [tok, lab, glo] + extras
            else:
                fields = [idx, tok, lab, glo] + extras

            while fields and str(fields[-1]).strip() == "":
                fields.pop()

            lines.append("\t".join(fields))
        out_blocks.append("\n".join(lines))
    return "\n\n".join(out_blocks)


def is_matrixembed_locked(token, new_value) -> bool:
    """MatrixLang/EmbedLang rows may only have their Label set to TR or EN."""
    return token in ("MatrixLang", "EmbedLang") and new_value not in ("TR", "EN")


def resolve_row(row_index_map, sep_rows, visible_row):
    """Resolve a visible grid row to its (bidx, ridx) model position.
    Returns (None, None) for separator rows or rows with no mapping."""
    if visible_row in sep_rows:
        return None, None
    return row_index_map.get(visible_row, (None, None))


def iter_visible_rows(blocks, row_index_map, sep_rows):
    """Iterate visible (non-separator) grid rows in sorted order, resolving
    each to its underlying model row. Skips rows that don't resolve to a
    model position (bidx is None)."""
    for vis_r in sorted(row_index_map.keys()):
        if vis_r in sep_rows:
            continue
        bidx, ridx = row_index_map.get(vis_r, (None, None))
        if bidx is None:
            continue
        row = blocks[bidx][ridx]
        yield vis_r, bidx, ridx, row


def collect_label_rows(blocks, row_index_map, sep_rows, labels):
    """Collect every row whose label is in `labels` (an iterable of label
    strings, e.g. {"UID"}), in corpus/visible order. Returns a list of
    {"vis_r", "bidx", "ridx"} dicts. Used by the UID Review Tool for its
    default UID list and, with a different label set, for other label-scoped
    views."""
    wanted = set(labels)
    out = []
    for vis_r, bidx, ridx, row in iter_visible_rows(blocks, row_index_map, sep_rows):
        if str(row.get("label", "") or "").strip() in wanted:
            out.append({"vis_r": vis_r, "bidx": bidx, "ridx": ridx})
    return out


def _normalize_for_match(tok, exact_surface, case_sensitive):
    """Normalize a token for occurrence matching. Default policy (exact_surface=False,
    case_sensitive=False): Unicode-aware casefold plus stripping only leading/trailing
    punctuation -- the same edge-punctuation rule freq_normalize_token uses, not a
    substring match. exact_surface=True compares the raw surface form instead (still
    subject to case_sensitive)."""
    if tok is None:
        return None
    s = str(tok)
    if not exact_surface:
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s, flags=re.UNICODE)
    if not s:
        return None
    if not case_sensitive:
        s = s.casefold()
    return s


def find_occurrences(blocks, row_index_map, sep_rows, query, *,
                      case_sensitive=False, exact_surface=False,
                      label_filter=None, bidx_filter=None):
    """Find every occurrence of `query` in corpus order. Returns a list of
    {"vis_r", "bidx", "ridx"} dicts.

    Default matching policy: normalized form (Unicode casefold + strip only
    leading/trailing punctuation), NOT substring matching. Pass
    exact_surface=True to compare raw surface forms instead (still subject to
    case_sensitive). Pass bidx_filter=<int> to restrict to one sentence/block
    ("current sentence only"). Pass label_filter=<iterable of labels> to
    restrict to specific current labels.
    """
    target = _normalize_for_match(query, exact_surface, case_sensitive)
    if not target:
        return []
    labels = set(label_filter) if label_filter is not None else None
    out = []
    for vis_r, bidx, ridx, row in iter_visible_rows(blocks, row_index_map, sep_rows):
        if bidx_filter is not None and bidx != bidx_filter:
            continue
        tok = row.get("token", "")
        if is_meta_row_token(tok):
            continue
        if _normalize_for_match(tok, exact_surface, case_sensitive) != target:
            continue
        if labels is not None and str(row.get("label", "") or "").strip() not in labels:
            continue
        out.append({"vis_r": vis_r, "bidx": bidx, "ridx": ridx})
    return out


def rows_are_adjacent_same_block(blocks, bidx, ridx_list):
    """True iff ridx_list (2 or more row indices) are all within the same
    block `bidx`, form a contiguous run of positions once sorted, and none
    of them is a meta row (SentenceID/MatrixLang/EmbedLang/blank). Used to
    gate token merging: no cross-sentence merge, no merge across a gap, and
    no merge that would swallow a structural row."""
    if ridx_list is None or len(ridx_list) < 2:
        return False
    if bidx is None or bidx < 0 or bidx >= len(blocks):
        return False
    rows = blocks[bidx]
    sorted_ridx = sorted(ridx_list)
    if len(set(sorted_ridx)) != len(sorted_ridx):
        return False
    for i in range(len(sorted_ridx) - 1):
        if sorted_ridx[i + 1] != sorted_ridx[i] + 1:
            return False
    for ridx in sorted_ridx:
        if ridx < 0 or ridx >= len(rows):
            return False
        if is_meta_row_token(rows[ridx].get("token", "")):
            return False
    return True


def merge_token_rows(blocks, bidx, ridx_list, merged_token, merged_label, merged_gloss):
    """Replace the contiguous rows at blocks[bidx][ridx] for ridx in
    ridx_list with a single merged row. Requires
    rows_are_adjacent_same_block(blocks, bidx, ridx_list) to be True --
    raises ValueError otherwise (no cross-sentence merge, no merge across a
    gap, no merge that includes a meta row). Caller must call
    renumber_tokens(blocks) afterward. Never infers the merged label or
    gloss -- both must be supplied explicitly by the caller (the UI's
    confirmation dialog), matching the "do not automatically infer the
    merged label" requirement."""
    if not rows_are_adjacent_same_block(blocks, bidx, ridx_list):
        raise ValueError(
            "rows to merge must be 2+, contiguous, in the same sentence, "
            "and contain no meta row"
        )
    merged_tok = str(merged_token).strip()
    if not merged_tok:
        raise ValueError("merged token must not be blank")
    rows = blocks[bidx]
    sorted_ridx = sorted(ridx_list)
    first, last = sorted_ridx[0], sorted_ridx[-1]
    new_row = {"idx": "", "token": merged_tok, "label": merged_label, "gloss": merged_gloss}
    blocks[bidx] = rows[:first] + [new_row] + rows[last + 1:]
    return True


def build_grid_view(blocks, extra_headers, skip_separator_after_empty_block):
    """Build tksheet-ready row data plus row_index_map/sep_rows from blocks.
    Returns (data, row_index_map, sep_rows).

    A separator row is inserted after every block except the last.
    If skip_separator_after_empty_block is True, that separator is
    additionally skipped when the block itself has no rows.
    """
    data = []
    row_index_map = {}
    sep_rows = set()
    row_cursor = 0
    for bidx, rows in enumerate(blocks):
        for ridx, r in enumerate(rows):
            for h in extra_headers:
                r.setdefault(h, "")
            idxv = r.get("idx", "")
            idxs = "" if idxv is None else str(idxv)
            vals = [idxs, r.get("token", ""), r.get("label", ""), r.get("gloss", "")]
            for h in extra_headers:
                vals.append(r.get(h, ""))
            data.append(vals)
            row_index_map[row_cursor] = (bidx, ridx)
            row_cursor += 1

        is_last = bidx == len(blocks) - 1
        insert_sep = (not is_last) and (bool(rows) if skip_separator_after_empty_block else True)
        if insert_sep:
            data.append(["" for _ in range(4 + len(extra_headers))])
            row_index_map[row_cursor] = (None, None)
            sep_rows.add(row_cursor)
            row_cursor += 1

    return data, row_index_map, sep_rows
