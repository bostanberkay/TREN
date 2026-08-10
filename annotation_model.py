# annotation_model.py
"""Pure annotation-state transformation functions, extracted from
cs_annotator_app.py (Stage 2.1). No tkinter import — callable and testable
independent of a running GUI."""

import copy
import json
import os
import re
import uuid


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


_TOKEN_LABEL_RE = re.compile(r"^(\S+)\s+(TR|EN|MIXED|UID|NE|OTHER|LANG3)\s*$")


def parse_annotated_text_to_blocks(text, extra_headers=None):
    """Parse pipeline output (or a reconstructed TXT-style export) into a
    fresh `blocks` list, exactly as `App._populate_table` has always parsed
    it: blocks are separated by a blank line, rows are tab-separated
    (2 fields = token, label; 3+ fields = idx, token, label, ...), with a
    fallback to a whitespace `TOKEN LABEL` pattern for lines with no tabs.
    `idx` is left at the placeholder 0 -- call `renumber_tokens` afterward.
    Does not mutate `extra_headers`."""
    headers = list(extra_headers) if extra_headers else []
    blocks = []
    for b in text.split("\n\n"):
        rows = []
        for ln in b.splitlines():
            ln = ln.strip()
            if not ln:
                continue

            parts = ln.split("\t")
            token = label = None

            if len(parts) >= 2:
                if len(parts) == 2:
                    token, label = parts[0].strip(), parts[1].strip()
                else:
                    token, label = parts[1].strip(), parts[2].strip()
            else:
                m = _TOKEN_LABEL_RE.match(ln)
                if m:
                    token, label = m.group(1), m.group(2)

            if token is None:
                continue

            rr = {"idx": 0, "token": token, "label": label, "gloss": ""}
            for h in headers:
                rr.setdefault(h, "")
            rows.append(rr)
        blocks.append(rows)
    return blocks


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
    """Collect every TOKEN row whose label is in `labels` (an iterable of
    label strings, e.g. {"UID"}), in corpus/visible order. Returns a list of
    {"vis_r", "bidx", "ridx"} dicts. Used by the Confidence Review Tool for
    its default UID list and, with a different label set, for its other
    label-scoped views.

    Meta rows (MatrixLang/EmbedLang/SentenceID) are always excluded, even
    when their own value happens to match `labels` -- e.g. a MatrixLang row
    is stored as {"token": "MatrixLang", "label": "TR"}, so filtering for
    {"TR"} would otherwise incorrectly collect it as if it were a TR TOKEN.
    This never mattered while the only caller filtered for {"UID"} (no meta
    row's value is ever "UID"), but matters as soon as a caller filters for
    TR/EN, so it is fixed here rather than left as a latent trap."""
    wanted = set(labels)
    out = []
    for vis_r, bidx, ridx, row in iter_visible_rows(blocks, row_index_map, sep_rows):
        if is_meta_row_token(row.get("token", "")):
            continue
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


# ---------------------------------------------------------------------------
# Multiple annotation datasets
#
# A project is a list of independent "dataset" dicts plus which one is
# active. Each dataset dict has exactly these keys:
#   id             -- stable internal string id (never shown in the UI)
#   name           -- user-visible name, e.g. "Data 1"
#   source_text    -- the complete raw input text this dataset was built from
#   blocks         -- the annotation blocks (same shape as the single-dataset
#                      `blocks` list documented in docs/file-formats.md)
#   extra_headers  -- this dataset's user-added grid columns
#   source_filename -- optional: the basename (e.g. "corpus.txt") of the
#                      file this dataset's source_text was read from (Add
#                      New Data -> Open New File), for display/reference
#                      only. May be None/absent. Deliberately NEVER a full
#                      path: an absolute path can expose the user's account
#                      name and local directory structure if the project
#                      file is shared, is non-portable, and provides no
#                      reopening benefit since source_text is already the
#                      authoritative, fully self-contained content. Nothing
#                      about reopening a project ever depends on the
#                      original file still existing on disk, still less at
#                      that same path. `make_dataset` defensively takes only
#                      os.path.basename() of whatever is passed here, so a
#                      full path can never end up stored even by accident.
# `make_dataset` always deep-copies `blocks`/`extra_headers` so two datasets
# never share a mutable row dict, even if built from the same source.
# ---------------------------------------------------------------------------

_DEFAULT_DATASET_NAME_RE = re.compile(r"^Data (\d+)$")


def make_dataset(name, source_text="", blocks=None, extra_headers=None, dataset_id=None, source_filename=None):
    """Build a new, fully independent dataset dict. `blocks` and
    `extra_headers` are deep-copied so the new dataset never shares mutable
    row dicts with whatever list was passed in. `source_filename` is
    optional, display-only metadata -- see the module docstring above. Only
    `os.path.basename(source_filename)` is ever stored, even if a full path
    is passed in, so this function itself is the single choke point that
    guarantees a full path can never leak into a saved project."""
    return {
        "id": dataset_id or uuid.uuid4().hex,
        "name": str(name),
        "source_text": "" if source_text is None else str(source_text),
        "blocks": copy.deepcopy(blocks) if blocks else [],
        "extra_headers": list(extra_headers) if extra_headers else [],
        "source_filename": os.path.basename(str(source_filename)) if source_filename else None,
    }


def next_default_dataset_name(existing_names):
    """"Data N" for the smallest N whose default name isn't already used by
    an *auto-generated* name in `existing_names` (a user-renamed dataset
    doesn't reserve a slot). Always returns "Data 1" for an empty project."""
    max_n = 0
    for n in existing_names or []:
        m = _DEFAULT_DATASET_NAME_RE.match(str(n).strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"Data {max_n + 1}"


def sanitize_dataset_filename(name):
    """Filesystem-safe slug derived from a dataset name, for use as (part
    of) an export filename. Never returns an empty string."""
    s = str(name or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or "dataset"


# `.trenproj` schema version history:
#   1 -- single implicit dataset: top-level "blocks"/"input_text"/"extra_headers".
#        REQUIRES this exact shape -- a "datasets" key present alongside
#        "version": 1 is rejected as malformed, never silently accepted.
#   2 -- multiple datasets: top-level "datasets" (list) + "active_dataset_index".
#        REQUIRES a non-empty "datasets" list.
# A project file with no "version" key at all predates versioning and is
# treated as version 1 (so it too requires the legacy shape). The dispatch
# is version-driven, not shape-driven: datasets_from_payload never infers
# the schema from which keys happen to be present. Bump
# CURRENT_PROJECT_SCHEMA_VERSION and extend SUPPORTED_PROJECT_SCHEMA_VERSIONS
# (plus datasets_from_payload below) when the schema changes again -- never
# reinterpret an unrecognized version silently.
CURRENT_PROJECT_SCHEMA_VERSION = 2
SUPPORTED_PROJECT_SCHEMA_VERSIONS = (1, 2)


def datasets_to_payload(datasets, active_index):
    """Build the `.trenproj`-ready portion of a project payload for a list
    of dataset dicts: "version" (always CURRENT_PROJECT_SCHEMA_VERSION for a
    freshly-written save), "datasets", and "active_dataset_index". Merge the
    result into the rest of the save payload (name, cfg, ...). Deliberately
    does NOT include each dataset's session-only undo stacks (see
    App._sync_active_dataset_from_live) -- positional undo/redo records are
    never persisted to disk, only kept in memory for the current session.
    "source_filename" is only written when set -- it's optional, display-
    only metadata (see the module docstring above; a basename only, NEVER a
    full path), never required to reopen the project."""
    out_datasets = []
    for ds in datasets:
        item = {
            "id": ds.get("id") or uuid.uuid4().hex,
            "name": ds.get("name", ""),
            "source_text": ds.get("source_text", ""),
            "blocks": ds.get("blocks", []),
            "extra_headers": ds.get("extra_headers", []),
        }
        source_filename = ds.get("source_filename")
        if source_filename:
            # os.path.basename is a defensive no-op here in the normal
            # case (ds["source_filename"] already went through
            # make_dataset's own basename-ing), but this is the last line
            # of defense before anything gets written to disk -- a full
            # path must never survive to a saved .trenproj no matter how
            # it ended up on the in-memory dataset dict.
            item["source_filename"] = os.path.basename(source_filename)
        out_datasets.append(item)
    return {
        "version": CURRENT_PROJECT_SCHEMA_VERSION,
        "datasets": out_datasets,
        "active_dataset_index": active_index,
    }


def datasets_from_payload(payload):
    """Rebuild a list of independent dataset dicts (see module docstring
    above) plus an active index from a loaded `.trenproj` payload.

    Strict version-driven policy (the shape of the payload never overrides
    what its declared version requires):

    - A missing "version" key defaults to 1 (pre-versioning projects).
    - An invalid version (non-int, bool, or outside
      SUPPORTED_PROJECT_SCHEMA_VERSIONS) raises ValueError immediately,
      rather than guessing at its shape.
    - Version 1 REQUIRES the legacy single-dataset shape (top-level
      "blocks"/"input_text"/"extra_headers") and becomes exactly one
      dataset named "Data 1". A version-1 payload that also contains a
      "datasets" key is rejected as malformed -- version 1 never takes the
      multi-dataset path, even if it happens to have that key (e.g. a
      hand-edited or half-migrated file); there is no "detect by shape"
      fallback here.
    - Version 2 REQUIRES a non-empty "datasets" list; a version-2 payload
      missing "datasets" is rejected as malformed.

    Raises ValueError with a human-readable message on any malformed
    structure so the caller can show an error dialog instead of crashing or
    silently discarding data. Returned datasets never share row dicts with
    the payload or each other, even if the payload itself aliased them.
    Never mutates `payload`."""
    version = payload.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"Invalid project schema version: {version!r}.")
    if version not in SUPPORTED_PROJECT_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported project schema version {version}. This version of "
            f"TREN supports schema versions {SUPPORTED_PROJECT_SCHEMA_VERSIONS}."
        )

    if version == 1:
        if "datasets" in payload:
            raise ValueError(
                "Project declares schema version 1 (legacy single-dataset) "
                "but also contains a 'datasets' list; version 1 payloads "
                "must use the legacy 'blocks'/'input_text'/'extra_headers' "
                "shape, not the multi-dataset shape."
            )
        blocks = payload.get("blocks", [])
        if not isinstance(blocks, list):
            blocks = []
        extra_headers = payload.get("extra_headers", [])
        if not isinstance(extra_headers, list):
            extra_headers = []
        source_text = payload.get("input_text", "")
        if not isinstance(source_text, str):
            source_text = ""
        return [make_dataset("Data 1", source_text, blocks, extra_headers)], 0

    # version == 2 (the only other currently-supported version).
    if "datasets" not in payload:
        raise ValueError(
            f"Project declares schema version {version} but is missing the 'datasets' list."
        )
    raw = payload.get("datasets")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'datasets' must be a non-empty list.")

    datasets = []
    seen_ids = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"dataset #{i + 1} is not an object.")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"dataset #{i + 1} has no valid 'name'.")
        blocks = item.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError(f"dataset #{i + 1} 'blocks' must be a list.")
        extra_headers = item.get("extra_headers", [])
        if not isinstance(extra_headers, list):
            raise ValueError(f"dataset #{i + 1} 'extra_headers' must be a list.")
        source_text = item.get("source_text", "")
        if not isinstance(source_text, str):
            raise ValueError(f"dataset #{i + 1} 'source_text' must be a string.")
        source_filename = item.get("source_filename")
        if source_filename is not None and not isinstance(source_filename, str):
            raise ValueError(f"dataset #{i + 1} 'source_filename' must be a string.")
        if source_filename is None:
            # Best-effort migration for a development-only payload that
            # still used the old "source_path" field name (predating this
            # privacy fix): take only its basename into memory, never the
            # raw path itself, and never write "source_path" back out on
            # the next save (datasets_to_payload no longer knows that key
            # exists). A malformed legacy value is ignored safely rather
            # than rejected -- it's optional fallback data, not a
            # structurally required field.
            legacy_source_path = item.get("source_path")
            if isinstance(legacy_source_path, str) and legacy_source_path:
                source_filename = os.path.basename(legacy_source_path)

        ds_id = item.get("id")
        if not isinstance(ds_id, str) or not ds_id or ds_id in seen_ids:
            ds_id = uuid.uuid4().hex
        seen_ids.add(ds_id)

        datasets.append(make_dataset(name, source_text, blocks, extra_headers,
                                      dataset_id=ds_id, source_filename=source_filename))

    active_index = payload.get("active_dataset_index", 0)
    if not isinstance(active_index, int) or not (0 <= active_index < len(datasets)):
        active_index = 0
    return datasets, active_index


def blocks_to_export_rows(blocks, extra_headers):
    """Grid-style export rows -- [idx, token, label, gloss, *extras] per
    row, with a blank separator row between (but not after) each block --
    for exporting a dataset that is not the currently-rendered grid (so
    there is no live tksheet to read from). Renumbers a deep copy; never
    mutates the `blocks` argument."""
    extra_headers = list(extra_headers or [])
    work = copy.deepcopy(blocks or [])
    renumber_tokens(work)
    ncol = 4 + len(extra_headers)
    rows = []
    for bidx, rows_in_block in enumerate(work):
        for r in rows_in_block:
            idxv = r.get("idx", "")
            idxs = "" if idxv is None else str(idxv)
            vals = [idxs, str(r.get("token", "") or ""), str(r.get("label", "") or ""), str(r.get("gloss", "") or "")]
            for h in extra_headers:
                vals.append(str(r.get(h, "") or ""))
            rows.append(vals)
        if bidx < len(work) - 1:
            rows.append(["" for _ in range(ncol)])
    return rows


# ---------------------------------------------------------------------------
# CoNLL / JSONL export
#
# Both formats are TREN-specific (documented in README.md as "TREN
# CoNLL-style", explicitly NOT CoNLL-U or any external shared-task format).
# Both use 1-based per-sentence token indices (restarting at 1 in every
# sentence/block), not the global running `idx` used by the grid/TXT/CSV
# export -- this matches conventional CoNLL-style tooling and keeps a single
# sentence's rows self-describing without needing the whole document.
# ---------------------------------------------------------------------------


def _export_lexical_rows(rows):
    """Real token rows of a block, in order, excluding meta rows
    (SentenceID/MatrixLang/EmbedLang/blank)."""
    return [r for r in rows if not is_meta_row_token(r.get("token", ""))]


def _export_sentence_meta(rows, fallback_sent_id):
    """(sent_id, matrix_lang, embedded_lang) for a block, read from its meta
    rows. sent_id falls back to `fallback_sent_id` (the block's 1-based
    position) when no SentenceID row is present; matrix_lang/embedded_lang
    fall back to "" when their meta row is absent (never fabricated)."""
    sent_id = str(fallback_sent_id)
    matrix_lang = ""
    embedded_lang = ""
    for r in rows:
        tok = str(r.get("token", "") or "").strip()
        if tok == "SentenceID":
            lab = str(r.get("label", "") or "").strip()
            if lab:
                sent_id = lab
        elif tok == "MatrixLang":
            matrix_lang = str(r.get("label", "") or "").strip()
        elif tok == "EmbedLang":
            embedded_lang = str(r.get("label", "") or "").strip()
    return sent_id, matrix_lang, embedded_lang


def _conll_field(value):
    """Escape a value for one tab-separated CoNLL field: a stray tab or
    newline in a token/label/gloss can never split or extend a line."""
    s = "" if value is None else str(value)
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def blocks_to_conll(blocks):
    """Serialize `blocks` to TREN's CoNLL-style export text (see module
    docstring above). Does not mutate `blocks`. Deterministic for identical
    input. Blocks with zero rows (transient empty blocks) are skipped."""
    lines = ["# TREN CoNLL export", "# columns = TokenIndex Token Label Gloss"]
    sentence_chunks = []
    for bidx, rows in enumerate(blocks or []):
        if not rows:
            continue
        sent_id, matrix_lang, embedded_lang = _export_sentence_meta(rows, bidx + 1)
        chunk = [f"# sent_id = {_conll_field(sent_id)}"]
        if matrix_lang:
            chunk.append(f"# matrix_lang = {_conll_field(matrix_lang)}")
        if embedded_lang:
            chunk.append(f"# embedded_lang = {_conll_field(embedded_lang)}")
        for i, r in enumerate(_export_lexical_rows(rows), start=1):
            tok = _conll_field(r.get("token", ""))
            lab = _conll_field(r.get("label", ""))
            gloss_raw = str(r.get("gloss", "") or "").strip()
            gloss = _conll_field(gloss_raw) if gloss_raw else "_"
            chunk.append("\t".join([str(i), tok, lab, gloss]))
        sentence_chunks.append("\n".join(chunk))

    text = "\n".join(lines)
    if sentence_chunks:
        text += "\n\n" + "\n\n".join(sentence_chunks)
    return text + "\n"


def blocks_to_jsonl_records(blocks, dataset_name):
    """List of plain-dict sentence records (see blocks_to_jsonl for the
    schema), one per non-empty block, in block order. Does not mutate
    `blocks`."""
    records = []
    for bidx, rows in enumerate(blocks or []):
        if not rows:
            continue
        lex = _export_lexical_rows(rows)
        sent_id, matrix_lang, embedded_lang = _export_sentence_meta(rows, bidx + 1)
        source_text = " ".join(str(r.get("token", "") or "") for r in lex)
        tokens = [
            {
                "index": i,
                "token": str(r.get("token", "") or ""),
                "label": str(r.get("label", "") or ""),
                "gloss": str(r.get("gloss", "") or ""),
            }
            for i, r in enumerate(lex, start=1)
        ]
        records.append({
            "sentence_id": sent_id,
            "dataset": str(dataset_name or ""),
            "source_text": source_text,
            "matrix_lang": matrix_lang,
            "embedded_lang": embedded_lang,
            "tokens": tokens,
        })
    return records


def blocks_to_jsonl(blocks, dataset_name):
    """Serialize `blocks` to JSONL text: one `json.dumps` object per
    non-empty sentence/block, one physical line each, UTF-8, Unicode
    preserved (ensure_ascii=False). Does not mutate `blocks`. Deterministic
    for identical input."""
    records = blocks_to_jsonl_records(blocks, dataset_name)
    lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    return "\n".join(lines) + ("\n" if lines else "")
