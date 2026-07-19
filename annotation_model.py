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
