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
