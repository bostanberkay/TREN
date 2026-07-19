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
