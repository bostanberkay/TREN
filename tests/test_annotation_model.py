import copy

import pytest

import json

from annotation_model import (
    is_meta_row_token,
    freq_normalize_token,
    compute_word_frequencies,
    sheet_rows_to_txt,
    renumber_tokens,
    reconstruct_text_from_blocks,
    is_matrixembed_locked,
    iter_visible_rows,
    resolve_row,
    build_grid_view,
    collect_label_rows,
    find_occurrences,
    rows_are_adjacent_same_block,
    merge_token_rows,
    parse_annotated_text_to_blocks,
    make_dataset,
    next_default_dataset_name,
    sanitize_dataset_filename,
    datasets_to_payload,
    datasets_from_payload,
    blocks_to_export_rows,
    blocks_to_conll,
    blocks_to_jsonl,
    blocks_to_jsonl_records,
    CURRENT_PROJECT_SCHEMA_VERSION,
    SUPPORTED_PROJECT_SCHEMA_VERSIONS,
)


def _row(tok, lab="", glo="", idx=""):
    return {"idx": idx, "token": tok, "label": lab, "gloss": glo}


# --- is_meta_row_token ---------------------------------------------------

@pytest.mark.parametrize("tok, expected", [
    (None, False),
    ("", True),
    ("   ", True),
    ("MatrixLang", True),
    ("EmbedLang", True),
    ("SentenceID", True),
    ("sentid", True),
    ("Sent_Id", True),
    ("SENTENCE_ID", True),
    ("Bugun", False),
    ("meeting'e", False),
])
def test_is_meta_row_token(tok, expected):
    assert is_meta_row_token(tok) == expected


def test_is_meta_row_token_case_sensitive_matrixembed():
    # Only the SentenceID family is case-insensitive; MatrixLang/EmbedLang are not.
    assert is_meta_row_token("matrixlang") is False
    assert is_meta_row_token("embedlang") is False


# --- freq_normalize_token -------------------------------------------------

@pytest.mark.parametrize("tok, expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("BUGUN", "bugun"),
    ("...hello...", "hello"),
    ("umbrella-ya", "umbrella-ya"),
    ("meeting'e", "meeting'e"),
    ("...", None),
    ("!!!", None),
])
def test_freq_normalize_token(tok, expected):
    assert freq_normalize_token(tok) == expected


def test_freq_normalize_token_turkish_casefold_is_not_locale_aware():
    # Documents current, accepted behavior: Python's str.casefold() is not
    # Turkish-locale-aware, so the dotted capital İ does not casefold to a
    # plain ASCII "i". This is intentional-until-changed, not a bug to fix
    # here -- locking it in so a future stdlib/behavior change is caught.
    assert freq_normalize_token("İstanbul") == "İstanbul".casefold()


# --- compute_word_frequencies ---------------------------------------------

def _freq_row(idx, tok, lab, glo=""):
    return {"idx": idx, "token": tok, "label": lab, "gloss": glo}


def _freq_blocks():
    # SentenceID/MatrixLang/EmbedLang/blank rows must be excluded from counts;
    # "Bugun"/"bugun" must collapse via casefolding; MIXED apostrophe tokens
    # and duplicate EN tokens across sentences are covered too.
    return [
        [
            _freq_row("", "SentenceID", "TR"),
            _freq_row(1, "Bugun", "TR"),
            _freq_row(2, "meeting'e", "MIXED"),
            _freq_row(3, "gitmem", "TR"),
            _freq_row(4, "stressed", "EN"),
            _freq_row(5, "boss'um", "MIXED"),
            _freq_row("", "MatrixLang", "TR"),
            _freq_row("", "EmbedLang", "EN"),
            _freq_row("", "", ""),
        ],
        [
            _freq_row("", "SentenceID", "TR"),
            _freq_row(1, "I", "TR"),
            _freq_row(2, "think", "EN"),
            _freq_row(3, "THINK", "EN"),
            _freq_row(4, "...", "OTHER"),
            _freq_row(5, "!!!", "OTHER"),
            _freq_row(6, "", ""),
            _freq_row(7, "   ", ""),
            _freq_row(8, "Bugun", "TR"),
            _freq_row("", "MatrixLang", "TR"),
        ],
    ]


@pytest.mark.parametrize("allowed_labels, expected", [
    (
        None,
        (
            {"bugun": 2, "meeting'e": 1, "gitmem": 1, "stressed": 1, "boss'um": 1, "i": 1, "think": 2},
            {
                "bugun": {"TR": 2},
                "meeting'e": {"MIXED": 1},
                "gitmem": {"TR": 1},
                "stressed": {"EN": 1},
                "boss'um": {"MIXED": 1},
                "i": {"TR": 1},
                "think": {"EN": 2},
            },
            9,
        ),
    ),
    (
        set(),
        ({}, {}, 0),
    ),
    (
        {"TR"},
        (
            {"bugun": 2, "gitmem": 1, "i": 1},
            {"bugun": {"TR": 2}, "gitmem": {"TR": 1}, "i": {"TR": 1}},
            4,
        ),
    ),
    (
        {"TR", "EN"},
        (
            {"bugun": 2, "gitmem": 1, "stressed": 1, "i": 1, "think": 2},
            {
                "bugun": {"TR": 2},
                "gitmem": {"TR": 1},
                "stressed": {"EN": 1},
                "i": {"TR": 1},
                "think": {"EN": 2},
            },
            7,
        ),
    ),
    (
        {"MIXED"},
        (
            {"meeting'e": 1, "boss'um": 1},
            {"meeting'e": {"MIXED": 1}, "boss'um": {"MIXED": 1}},
            2,
        ),
    ),
])
def test_compute_word_frequencies_allowed_labels(allowed_labels, expected):
    # Also exercises mixed-casing collapse ("Bugun"/"bugun" -> "bugun") and
    # meta-row exclusion (SentenceID/MatrixLang/EmbedLang/blank never counted).
    assert compute_word_frequencies(_freq_blocks(), allowed_labels) == expected


def test_compute_word_frequencies_blocks_none():
    assert compute_word_frequencies(None) == ({}, {}, 0)


def test_compute_word_frequencies_does_not_mutate_blocks():
    blocks = _freq_blocks()
    before = copy.deepcopy(blocks)
    compute_word_frequencies(blocks, allowed_labels=None)
    assert blocks == before


# --- sheet_rows_to_txt ------------------------------------------------------

_HEADERS_STD = ["Token", "Item", "Label", "Gloss"]
_HEADERS_EXTRA = ["Token", "Item", "Label", "Gloss", "Note", "POS"]

_SHEET_ROWS_CASES = [
    (
        "normal_rows",
        _HEADERS_STD,
        [
            ["1", "Bugun", "TR", ""],
            ["2", "meeting'e", "MIXED", "stem-DAT"],
            ["", "SentenceID", "TR", ""],
            None,
            ["3", "stressed", "EN", ""],
        ],
        "1\tBugun\tTR\n2\tmeeting'e\tMIXED\tstem-DAT\nSentenceID\tTR\n\n3\tstressed\tEN",
    ),
    (
        "blank_separator_empty_string_row",
        _HEADERS_STD,
        [
            ["1", "a", "TR", ""],
            ["", "", "", ""],
            ["2", "b", "EN", ""],
        ],
        "1\ta\tTR\n\n2\tb\tEN",
    ),
    (
        "none_cell_values",
        _HEADERS_STD,
        [
            [None, None, None, None],
            ["1", "x", "TR", None],
        ],
        "\n1\tx\tTR",
    ),
    (
        "short_rows_padding",
        _HEADERS_STD,
        [
            ["1", "x"],
            ["", "MatrixLang"],
        ],
        "1\tx\nMatrixLang",
    ),
    (
        "long_rows_truncation",
        _HEADERS_STD,
        [
            ["1", "x", "TR", "gloss1", "extra1", "extra2"],
        ],
        "1\tx\tTR\tgloss1",
    ),
    (
        "trailing_empty_fields_trimmed",
        _HEADERS_STD,
        [
            ["1", "x", "TR", ""],
            ["", "EmbedLang", "EN", ""],
        ],
        "1\tx\tTR\nEmbedLang\tEN",
    ),
    (
        "extra_columns_present",
        _HEADERS_EXTRA,
        [
            ["1", "x", "TR", "g", "note1", ""],
            ["2", "y", "EN", "", "", "NOUN"],
        ],
        "1\tx\tTR\tg\tnote1\n2\ty\tEN\t\t\tNOUN",
    ),
    (
        "all_rows_blank",
        _HEADERS_STD,
        [
            ["", "", "", ""],
            None,
            ["", "", "", ""],
        ],
        "",
    ),
    (
        "empty_rows_list",
        _HEADERS_STD,
        [],
        "",
    ),
    (
        "empty_list_row",
        _HEADERS_STD,
        [
            ["1", "a", "TR", ""],
            [],
            ["2", "b", "EN", ""],
        ],
        "1\ta\tTR\n\n2\tb\tEN",
    ),
    (
        "unicode_apostrophe_punctuation",
        _HEADERS_STD,
        [
            ["1", "boss'um", "MIXED", ""],
            ["2", "İstanbul", "NE", ""],
            ["3", "...", "OTHER", ""],
            ["4", "çok", "TR", ""],
        ],
        "1\tboss'um\tMIXED\n2\tİstanbul\tNE\n3\t...\tOTHER\n4\tçok\tTR",
    ),
]


@pytest.mark.parametrize("name, headers, rows, expected", _SHEET_ROWS_CASES, ids=[c[0] for c in _SHEET_ROWS_CASES])
def test_sheet_rows_to_txt(name, headers, rows, expected):
    # Byte-for-byte output assertion against the exact expected TXT string.
    assert sheet_rows_to_txt(rows, headers) == expected


@pytest.mark.parametrize("name, headers, rows, expected", _SHEET_ROWS_CASES, ids=[c[0] for c in _SHEET_ROWS_CASES])
def test_sheet_rows_to_txt_does_not_mutate_rows(name, headers, rows, expected):
    before = copy.deepcopy(rows)
    sheet_rows_to_txt(rows, headers)
    assert rows == before


# --- renumber_tokens ---------------------------------------------------

def _rt_row(tok, lab="", glo="", idx="PRE", **extra):
    d = {"idx": idx, "token": tok, "label": lab, "gloss": glo}
    d.update(extra)
    return d


def test_renumber_tokens_global_numbering_across_blocks():
    # Numbering must be a single, global counter -- it must NOT reset at
    # each block boundary.
    blocks = [
        [_rt_row("Bugun", "TR"), _rt_row("meeting'e", "MIXED")],
        [_rt_row("I", "TR"), _rt_row("think", "EN"), _rt_row("bir", "TR")],
    ]
    renumber_tokens(blocks)
    assert [[r["idx"] for r in blk] for blk in blocks] == [[1, 2], [3, 4, 5]]


def test_renumber_tokens_skips_metadata_rows():
    blocks = [
        [
            _rt_row("SentenceID"),
            _rt_row("a", "TR"),
            _rt_row("MatrixLang"),
            _rt_row("b", "EN"),
            _rt_row("EmbedLang"),
        ],
    ]
    renumber_tokens(blocks)
    assert [(r["token"], r["idx"]) for blk in blocks for r in blk] == [
        ("SentenceID", ""),
        ("a", 1),
        ("MatrixLang", ""),
        ("b", 2),
        ("EmbedLang", ""),
    ]


def test_renumber_tokens_overwrites_existing_idx_values():
    blocks = [[_rt_row("a", "TR", idx=999), _rt_row("b", "EN", idx=-5)]]
    renumber_tokens(blocks)
    assert [r["idx"] for blk in blocks for r in blk] == [1, 2]


def test_renumber_tokens_row_missing_token_key():
    # A row dict with no "token" key at all defaults to "" (via r.get),
    # which is itself a meta-row token, so it gets idx == "".
    blocks = [[{"idx": "X", "label": "TR", "gloss": ""}, _rt_row("y", "EN")]]
    renumber_tokens(blocks)
    assert [r.get("idx") for blk in blocks for r in blk] == ["", 1]


def test_renumber_tokens_returns_none():
    blocks = [[_rt_row("a", "TR")]]
    assert renumber_tokens(blocks) is None


def test_renumber_tokens_mutates_blocks_in_place():
    blocks = [[_rt_row("a", "TR", idx="PRE")]]
    row_obj = blocks[0][0]
    renumber_tokens(blocks)
    # Same dict object, mutated, not replaced.
    assert blocks[0][0] is row_obj
    assert row_obj["idx"] == 1


# --- reconstruct_text_from_blocks ------------------------------------------

_RECONSTRUCT_CASES = [
    (
        "empty_blocks_list",
        [],
        [],
        "",
    ),
    (
        "single_block_empty_rows",
        [[]],
        [],
        "",
    ),
    (
        "multiple_blocks_normal",
        [
            [_rt_row("Bugun", "TR"), _rt_row("meeting'e", "MIXED", "stem-DAT"), _rt_row("stressed", "EN")],
            [_rt_row("I", "TR"), _rt_row("think", "EN")],
        ],
        [],
        "1\tBugun\tTR\n2\tmeeting'e\tMIXED\tstem-DAT\n3\tstressed\tEN\n\n4\tI\tTR\n5\tthink\tEN",
    ),
    (
        "metadata_rows",
        [
            [_rt_row("SentenceID"), _rt_row("x", "TR"), _rt_row("y", "EN"), _rt_row("MatrixLang", "TR"), _rt_row("EmbedLang", "EN")],
        ],
        [],
        "SentenceID\n1\tx\tTR\n2\ty\tEN\nMatrixLang\tTR\nEmbedLang\tEN",
    ),
    (
        "extra_headers_trailing_trim",
        [
            [_rt_row("a", "TR", "gloss1", note="n1", pos=""), _rt_row("b", "EN", "", note="", pos="")],
        ],
        ["note", "pos"],
        "1\ta\tTR\tgloss1\tn1\n2\tb\tEN",
    ),
    (
        "extra_headers_fully_populated",
        [
            [_rt_row("a", "TR", "g", note="n1", pos="NOUN")],
        ],
        ["note", "pos"],
        "1\ta\tTR\tg\tn1\tNOUN",
    ),
    (
        "whitespace_only_gloss",
        [
            [_rt_row("a", "TR", "   ")],
        ],
        [],
        "1\ta\tTR",
    ),
    (
        "empty_token_rows_skipped_inline",
        [
            [_rt_row("a", "TR"), _rt_row(""), _rt_row("b", "EN")],
            [_rt_row("")],
        ],
        [],
        "1\ta\tTR\n2\tb\tEN\n\n",
    ),
]


@pytest.mark.parametrize(
    "name, blocks, extra_headers, expected",
    _RECONSTRUCT_CASES,
    ids=[c[0] for c in _RECONSTRUCT_CASES],
)
def test_reconstruct_text_from_blocks(name, blocks, extra_headers, expected):
    # Byte-for-byte output assertion against the exact expected TXT string.
    blocks_copy = copy.deepcopy(blocks)
    assert reconstruct_text_from_blocks(blocks_copy, extra_headers) == expected


def test_reconstruct_text_from_blocks_renumbering_side_effect():
    blocks = [
        [_rt_row("SentenceID"), _rt_row("Bugun", "TR"), _rt_row("meeting'e", "MIXED")],
        [_rt_row("I", "TR"), _rt_row("MatrixLang")],
    ]
    reconstruct_text_from_blocks(blocks, [])
    assert [[r["idx"] for r in blk] for blk in blocks] == [["", 1, 2], [3, ""]]


def test_reconstruct_text_from_blocks_delegates_to_renumber_tokens():
    # The numbering side effect must be identical whether renumber_tokens is
    # called directly or reached via reconstruct_text_from_blocks -- proving
    # this function delegates rather than re-implementing numbering.
    blocks = [
        [_rt_row("SentenceID"), _rt_row("Bugun", "TR"), _rt_row("meeting'e", "MIXED")],
        [_rt_row("I", "TR"), _rt_row("MatrixLang")],
    ]

    blocks_direct = copy.deepcopy(blocks)
    renumber_tokens(blocks_direct)

    blocks_via_reconstruct = copy.deepcopy(blocks)
    reconstruct_text_from_blocks(blocks_via_reconstruct, [])

    idx_direct = [[r["idx"] for r in blk] for blk in blocks_direct]
    idx_via = [[r["idx"] for r in blk] for blk in blocks_via_reconstruct]
    assert idx_direct == idx_via


# --- is_matrixembed_locked -------------------------------------------------

@pytest.mark.parametrize("token, new_value, expected", [
    # MatrixLang cases
    ("MatrixLang", "TR", False),
    ("MatrixLang", "EN", False),
    ("MatrixLang", "MIXED", True),
    ("MatrixLang", "", True),
    # EmbedLang cases
    ("EmbedLang", "TR", False),
    ("EmbedLang", "EN", False),
    ("EmbedLang", "UID", True),
    ("EmbedLang", None, True),
    # valid/invalid edits on non-meta tokens (never locked)
    ("SentenceID", "TR", False),
    ("SentenceID", "XYZ", False),
    ("Bugun", "TR", False),
    ("Bugun", "ANYTHING", False),
    # None token
    (None, "TR", False),
    (None, None, False),
    # ordinary/empty token
    ("", "", False),
], ids=[
    "MatrixLang-TR-unlocked", "MatrixLang-EN-unlocked", "MatrixLang-MIXED-locked", "MatrixLang-empty-locked",
    "EmbedLang-TR-unlocked", "EmbedLang-EN-unlocked", "EmbedLang-UID-locked", "EmbedLang-None-locked",
    "SentenceID-TR-unlocked", "SentenceID-XYZ-unlocked", "Bugun-TR-unlocked", "Bugun-ANYTHING-unlocked",
    "None_token-TR-unlocked", "None_token-None-unlocked", "empty_token-empty-unlocked",
])
def test_is_matrixembed_locked(token, new_value, expected):
    assert is_matrixembed_locked(token, new_value) == expected


def test_is_matrixembed_locked_case_sensitive():
    # Only the exact "MatrixLang"/"EmbedLang" strings trigger the lock.
    assert is_matrixembed_locked("matrixlang", "TR") is False
    assert is_matrixembed_locked("embedlang", "XYZ") is False


# --- iter_visible_rows -------------------------------------------------

def test_iter_visible_rows_normal_multiple_blocks():
    blocks = [
        [_row("Bugun", "TR"), _row("meeting'e", "MIXED")],
        [_row("I", "TR"), _row("think", "EN")],
    ]
    row_index_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    sep_rows = set()
    result = list(iter_visible_rows(blocks, row_index_map, sep_rows))
    assert result == [
        (0, 0, 0, blocks[0][0]),
        (1, 0, 1, blocks[0][1]),
        (2, 1, 0, blocks[1][0]),
        (3, 1, 1, blocks[1][1]),
    ]


def test_iter_visible_rows_with_separator_rows():
    blocks = [
        [_row("a", "TR"), _row("b", "EN")],
        [_row("c", "MIXED")],
    ]
    row_index_map = {0: (0, 0), 1: (0, 1), 2: (None, None), 3: (1, 0)}
    sep_rows = {2}
    result = list(iter_visible_rows(blocks, row_index_map, sep_rows))
    assert result == [
        (0, 0, 0, blocks[0][0]),
        (1, 0, 1, blocks[0][1]),
        (3, 1, 0, blocks[1][0]),
    ]


def test_iter_visible_rows_unresolved_rows_not_in_sep_rows():
    # bidx is None but the row isn't flagged as a separator -- must still skip.
    blocks = [[_row("x", "TR")]]
    row_index_map = {0: (0, 0), 1: (None, None), 2: (None, None)}
    sep_rows = set()
    result = list(iter_visible_rows(blocks, row_index_map, sep_rows))
    assert result == [(0, 0, 0, blocks[0][0])]


def test_iter_visible_rows_empty_mappings():
    assert list(iter_visible_rows([], {}, set())) == []


def test_iter_visible_rows_interleaved_separators():
    blocks = [
        [_row("a1", "TR")],
        [_row("b1", "EN"), _row("b2", "MIXED")],
        [_row("c1", "TR")],
    ]
    row_index_map = {0: (0, 0), 1: (None, None), 2: (1, 0), 3: (1, 1), 4: (None, None), 5: (2, 0)}
    sep_rows = {1, 4}
    result = list(iter_visible_rows(blocks, row_index_map, sep_rows))
    assert result == [
        (0, 0, 0, blocks[0][0]),
        (2, 1, 0, blocks[1][0]),
        (3, 1, 1, blocks[1][1]),
        (5, 2, 0, blocks[2][0]),
    ]


def test_iter_visible_rows_stale_row_index_map_raises_index_error():
    # Documents current, accepted behavior (see the S2 disclosure in review):
    # a row_index_map entry pointing past the end of blocks is not silently
    # skipped -- it raises IndexError when the generator is consumed. This
    # only matters if row_index_map and blocks are already out of sync, which
    # shouldn't happen in normal operation.
    blocks = [[_row("a", "MIXED")]]
    row_index_map = {0: (0, 0), 1: (5, 0)}
    sep_rows = set()
    gen = iter_visible_rows(blocks, row_index_map, sep_rows)
    with pytest.raises(IndexError):
        list(gen)


# --- resolve_row -------------------------------------------------------

@pytest.mark.parametrize("row_index_map, sep_rows, visible_row, expected", [
    ({0: (0, 0), 1: (None, None)}, {1}, 1, (None, None)),
    ({0: (0, 0), 1: (0, 1), 2: (1, 0)}, set(), 1, (0, 1)),
    ({0: (0, 0), 1: (0, 1)}, set(), 0, (0, 0)),
    ({0: (0, 0)}, set(), 99, (None, None)),
    ({}, set(), 0, (None, None)),
    ({}, set(), 5, (None, None)),
    ({0: (0, 0)}, {7}, 7, (None, None)),
    # Separator precedence: present in the map but also flagged as a
    # separator -- separator status must win.
    ({3: (2, 1)}, {3}, 3, (None, None)),
], ids=[
    "separator_row", "valid_row", "valid_row_first_entry", "missing_row",
    "empty_mappings", "empty_mappings_row_absent", "row_absent_and_separator",
    "separator_precedence_over_map_presence",
])
def test_resolve_row(row_index_map, sep_rows, visible_row, expected):
    assert resolve_row(row_index_map, sep_rows, visible_row) == expected


# --- build_grid_view ---------------------------------------------------

_GRID_VIEW_CASES = [
    (
        "multiple_non_empty_blocks",
        [
            [_row("Bugun", "TR"), _row("meeting'e", "MIXED")],
            [_row("I", "TR"), _row("think", "EN")],
            [_row("x", "OTHER")],
        ],
    ),
    (
        "empty_block_in_the_middle",
        [
            [_row("a", "TR")],
            [],
            [_row("b", "EN")],
        ],
    ),
    (
        "empty_block_at_the_end",
        [
            [_row("a", "TR")],
            [_row("b", "EN")],
            [],
        ],
    ),
    (
        "empty_block_at_the_start",
        [
            [],
            [_row("a", "TR")],
            [_row("b", "EN")],
        ],
    ),
    (
        "single_block_only",
        [
            [_row("a", "TR"), _row("b", "EN")],
        ],
    ),
    (
        "single_empty_block_only",
        [
            [],
        ],
    ),
    (
        "empty_blocks_list",
        [],
    ),
    (
        "multiple_consecutive_empty_blocks",
        [
            [_row("a", "TR")],
            [],
            [],
            [_row("b", "EN")],
        ],
    ),
]

# Expected (data, row_index_map, sep_rows) per (case, policy), computed by
# direct execution against the real function before writing these tests.
_GRID_VIEW_EXPECTED = {
    ("multiple_non_empty_blocks", True): (
        [["", "Bugun", "TR", ""], ["", "meeting'e", "MIXED", ""], ["", "", "", ""],
         ["", "I", "TR", ""], ["", "think", "EN", ""], ["", "", "", ""], ["", "x", "OTHER", ""]],
        {0: (0, 0), 1: (0, 1), 2: (None, None), 3: (1, 0), 4: (1, 1), 5: (None, None), 6: (2, 0)},
        {2, 5},
    ),
    ("multiple_non_empty_blocks", False): (
        [["", "Bugun", "TR", ""], ["", "meeting'e", "MIXED", ""], ["", "", "", ""],
         ["", "I", "TR", ""], ["", "think", "EN", ""], ["", "", "", ""], ["", "x", "OTHER", ""]],
        {0: (0, 0), 1: (0, 1), 2: (None, None), 3: (1, 0), 4: (1, 1), 5: (None, None), 6: (2, 0)},
        {2, 5},
    ),
    ("empty_block_in_the_middle", True): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "b", "EN", ""]],
        {0: (0, 0), 1: (None, None), 2: (2, 0)},
        {1},
    ),
    ("empty_block_in_the_middle", False): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "", "", ""], ["", "b", "EN", ""]],
        {0: (0, 0), 1: (None, None), 2: (None, None), 3: (2, 0)},
        {1, 2},
    ),
    ("empty_block_at_the_end", True): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "b", "EN", ""], ["", "", "", ""]],
        {0: (0, 0), 1: (None, None), 2: (1, 0), 3: (None, None)},
        {1, 3},
    ),
    ("empty_block_at_the_end", False): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "b", "EN", ""], ["", "", "", ""]],
        {0: (0, 0), 1: (None, None), 2: (1, 0), 3: (None, None)},
        {1, 3},
    ),
    ("empty_block_at_the_start", True): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "b", "EN", ""]],
        {0: (1, 0), 1: (None, None), 2: (2, 0)},
        {1},
    ),
    ("empty_block_at_the_start", False): (
        [["", "", "", ""], ["", "a", "TR", ""], ["", "", "", ""], ["", "b", "EN", ""]],
        {0: (None, None), 1: (1, 0), 2: (None, None), 3: (2, 0)},
        {0, 2},
    ),
    ("single_block_only", True): (
        [["", "a", "TR", ""], ["", "b", "EN", ""]],
        {0: (0, 0), 1: (0, 1)},
        set(),
    ),
    ("single_block_only", False): (
        [["", "a", "TR", ""], ["", "b", "EN", ""]],
        {0: (0, 0), 1: (0, 1)},
        set(),
    ),
    ("single_empty_block_only", True): ([], {}, set()),
    ("single_empty_block_only", False): ([], {}, set()),
    ("empty_blocks_list", True): ([], {}, set()),
    ("empty_blocks_list", False): ([], {}, set()),
    ("multiple_consecutive_empty_blocks", True): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "b", "EN", ""]],
        {0: (0, 0), 1: (None, None), 2: (3, 0)},
        {1},
    ),
    ("multiple_consecutive_empty_blocks", False): (
        [["", "a", "TR", ""], ["", "", "", ""], ["", "", "", ""], ["", "", "", ""], ["", "b", "EN", ""]],
        {0: (0, 0), 1: (None, None), 2: (None, None), 3: (None, None), 4: (3, 0)},
        {1, 2, 3},
    ),
}


@pytest.mark.parametrize("name, blocks", _GRID_VIEW_CASES, ids=[c[0] for c in _GRID_VIEW_CASES])
@pytest.mark.parametrize("skip_policy", [True, False], ids=["skip_empty_sep", "always_sep"])
def test_build_grid_view(name, blocks, skip_policy):
    blocks_copy = copy.deepcopy(blocks)
    result = build_grid_view(blocks_copy, [], skip_separator_after_empty_block=skip_policy)
    assert result == _GRID_VIEW_EXPECTED[(name, skip_policy)]


def test_build_grid_view_separator_policies_diverge_for_empty_middle_block():
    # Regression guard for the discovered _populate_table vs.
    # _rebuild_grid_from_model inconsistency: for a block structure with an
    # empty middle block, the two policies MUST produce different sep_rows.
    # If this test ever passes with skip_policy True and False producing the
    # same result for this input, the divergence this parameter exists to
    # encode has been lost.
    blocks = [[_row("a", "TR")], [], [_row("b", "EN")]]

    _, _, sep_true = build_grid_view(copy.deepcopy(blocks), [], skip_separator_after_empty_block=True)
    _, _, sep_false = build_grid_view(copy.deepcopy(blocks), [], skip_separator_after_empty_block=False)

    assert sep_true == {1}
    assert sep_false == {1, 2}
    assert sep_true != sep_false


def test_build_grid_view_custom_header_backfill_and_mutation():
    blocks = [
        [_row("a", "TR"), {"idx": "", "token": "b", "label": "EN", "gloss": ""}],
    ]
    before_keys = [set(r.keys()) for blk in blocks for r in blk]

    data, row_index_map, sep_rows = build_grid_view(blocks, ["note", "pos"], skip_separator_after_empty_block=True)

    after_keys = [set(r.keys()) for blk in blocks for r in blk]
    # setdefault backfills the missing extra-header keys onto the row dicts
    # in place -- neither row had "note"/"pos" before the call.
    assert before_keys == [
        {"idx", "token", "label", "gloss"},
        {"idx", "token", "label", "gloss"},
    ]
    assert after_keys == [
        {"idx", "token", "label", "gloss", "note", "pos"},
        {"idx", "token", "label", "gloss", "note", "pos"},
    ]
    assert data == [["", "a", "TR", "", "", ""], ["", "b", "EN", "", "", ""]]
    assert row_index_map == {0: (0, 0), 1: (0, 1)}
    assert sep_rows == set()


def _grid(blocks):
    """Shared helper for the UID Review Tool model-function tests below:
    build (row_index_map, sep_rows) for `blocks` the same way the live app
    does via build_grid_view, so these tests exercise the exact same
    corpus-order/visible-row machinery the GUI relies on."""
    _, row_index_map, sep_rows = build_grid_view(copy.deepcopy(blocks), [], skip_separator_after_empty_block=True)
    return row_index_map, sep_rows


# --- collect_label_rows ------------------------------------------------

def test_collect_label_rows_uid_only_in_corpus_order():
    blocks = [
        [_row("Cafeye", "UID"), _row("gittik", "TR"), _row("kahveleri", "UID")],
        [_row("Bodyci", "MIXED"), _row("abiler", "UID")],
    ]
    row_index_map, sep_rows = _grid(blocks)
    result = collect_label_rows(blocks, row_index_map, sep_rows, {"UID"})
    assert [(r["bidx"], r["ridx"]) for r in result] == [(0, 0), (0, 2), (1, 1)]


def test_collect_label_rows_multiple_labels():
    blocks = [[_row("a", "UID"), _row("b", "NE"), _row("c", "TR")]]
    row_index_map, sep_rows = _grid(blocks)
    result = collect_label_rows(blocks, row_index_map, sep_rows, {"UID", "NE"})
    assert [(r["bidx"], r["ridx"]) for r in result] == [(0, 0), (0, 1)]


def test_collect_label_rows_empty_project():
    row_index_map, sep_rows = _grid([])
    assert collect_label_rows([], row_index_map, sep_rows, {"UID"}) == []


# --- find_occurrences ----------------------------------------------------

def test_find_occurrences_default_policy_is_normalized_not_substring():
    # "backend" and "backends" must NOT cross-match under the default
    # (normalized, non-substring) policy.
    blocks = [[_row("Backend", "EN"), _row("backends", "EN"), _row("backend.", "EN")]]
    row_index_map, sep_rows = _grid(blocks)
    result = find_occurrences(blocks, row_index_map, sep_rows, "backend")
    # "Backend" (casefold) and "backend." (edge punctuation stripped) match;
    # "backends" does not, because this is exact normalized matching, not substring.
    assert [(r["bidx"], r["ridx"]) for r in result] == [(0, 0), (0, 2)]


def test_find_occurrences_unicode_casefold():
    blocks = [[_row("İstanbul", "NE"), _row("istanbul", "NE")]]
    row_index_map, sep_rows = _grid(blocks)
    result = find_occurrences(blocks, row_index_map, sep_rows, "i̇stanbul")
    assert len(result) >= 1  # casefold must not crash on Turkish dotted-I; exact set is locale-sensitive


def test_find_occurrences_case_sensitive():
    blocks = [[_row("May", "NE"), _row("may", "TR")]]
    row_index_map, sep_rows = _grid(blocks)
    insensitive = find_occurrences(blocks, row_index_map, sep_rows, "may", case_sensitive=False)
    sensitive = find_occurrences(blocks, row_index_map, sep_rows, "may", case_sensitive=True)
    assert len(insensitive) == 2
    assert [(r["bidx"], r["ridx"]) for r in sensitive] == [(0, 1)]


def test_find_occurrences_exact_surface_form():
    blocks = [[_row("backend.", "EN"), _row("backend", "EN")]]
    row_index_map, sep_rows = _grid(blocks)
    normalized = find_occurrences(blocks, row_index_map, sep_rows, "backend", exact_surface=False)
    exact = find_occurrences(blocks, row_index_map, sep_rows, "backend", exact_surface=True)
    assert len(normalized) == 2  # edge punctuation stripped by default
    assert [(r["bidx"], r["ridx"]) for r in exact] == [(0, 1)]  # only the unpunctuated surface form


def test_find_occurrences_current_sentence_only_vs_entire_project():
    blocks = [[_row("data", "EN")], [_row("data", "TR")]]
    row_index_map, sep_rows = _grid(blocks)
    whole_project = find_occurrences(blocks, row_index_map, sep_rows, "data")
    sentence_0_only = find_occurrences(blocks, row_index_map, sep_rows, "data", bidx_filter=0)
    assert len(whole_project) == 2
    assert [(r["bidx"], r["ridx"]) for r in sentence_0_only] == [(0, 0)]


def test_find_occurrences_label_filter():
    blocks = [[_row("data", "EN"), _row("data", "TR"), _row("data", "UID")]]
    row_index_map, sep_rows = _grid(blocks)
    result = find_occurrences(blocks, row_index_map, sep_rows, "data", label_filter={"EN", "UID"})
    assert [(r["bidx"], r["ridx"]) for r in result] == [(0, 0), (0, 2)]


def test_find_occurrences_skips_meta_rows():
    blocks = [[_row("MatrixLang", "TR"), _row("EmbedLang", "EN")]]
    row_index_map, sep_rows = _grid(blocks)
    assert find_occurrences(blocks, row_index_map, sep_rows, "MatrixLang") == []


def test_find_occurrences_no_query_returns_empty():
    blocks = [[_row("a", "TR")]]
    row_index_map, sep_rows = _grid(blocks)
    assert find_occurrences(blocks, row_index_map, sep_rows, "") == []
    assert find_occurrences(blocks, row_index_map, sep_rows, "***") == []  # normalizes to empty


# --- rows_are_adjacent_same_block -------------------------------------------

def test_rows_are_adjacent_same_block_true_for_contiguous_run():
    blocks = [[_row("node"), _row("ları"), _row("vs")]]
    assert rows_are_adjacent_same_block(blocks, 0, [0, 1]) is True
    assert rows_are_adjacent_same_block(blocks, 0, [1, 2]) is True
    assert rows_are_adjacent_same_block(blocks, 0, [0, 1, 2]) is True


def test_rows_are_adjacent_same_block_false_for_gap():
    blocks = [[_row("a"), _row("b"), _row("c")]]
    assert rows_are_adjacent_same_block(blocks, 0, [0, 2]) is False


def test_rows_are_adjacent_same_block_false_across_sentences():
    blocks = [[_row("a")], [_row("b")]]
    # even if a caller mistakenly passes ridx values from different blocks,
    # this function only ever looks within ONE bidx -- cross-sentence
    # merges must be rejected by construction at the call site (ridx_list
    # can only reference one block's positions), and a single-block index
    # list spanning bidx 0's single row plus a nonexistent bidx-0 position
    # 1 (which is really block 1's row) must fail as out-of-range.
    assert rows_are_adjacent_same_block(blocks, 0, [0, 1]) is False


def test_rows_are_adjacent_same_block_false_for_meta_row():
    blocks = [[_row("a", "TR"), _row("MatrixLang", "TR"), _row("b", "TR")]]
    assert rows_are_adjacent_same_block(blocks, 0, [1, 2]) is False


def test_rows_are_adjacent_same_block_false_for_single_row():
    blocks = [[_row("a")]]
    assert rows_are_adjacent_same_block(blocks, 0, [0]) is False


# --- merge_token_rows -------------------------------------------------------


def test_merge_token_rows_combines_adjacent_rows():
    blocks = [[_row("before", "TR"), _row("node", "EN"), _row("ları", "TR"), _row("after", "TR")]]
    merge_token_rows(blocks, 0, [1, 2], "nodeları", "MIXED", "node-PL")
    assert [r["token"] for r in blocks[0]] == ["before", "nodeları", "after"]
    assert blocks[0][1]["label"] == "MIXED"
    assert blocks[0][1]["gloss"] == "node-PL"


def test_merge_token_rows_renumbers_correctly_after_call():
    blocks = [[_row("before", "TR"), _row("node", "EN"), _row("ları", "TR"), _row("after", "TR")]]
    merge_token_rows(blocks, 0, [1, 2], "nodeları", "MIXED", "")
    renumber_tokens(blocks)
    assert [r["idx"] for r in blocks[0]] == [1, 2, 3]


def test_merge_token_rows_does_not_infer_label_or_gloss():
    # The function signature REQUIRES the caller to supply merged_label and
    # merged_gloss explicitly -- there is no code path that derives them
    # from the constituent rows' own labels/glosses.
    blocks = [[_row("node", "EN", "n."), _row("ları", "TR", "pl.")]]
    merge_token_rows(blocks, 0, [0, 1], "nodeları", "UID", "")
    assert blocks[0][0]["label"] == "UID"  # explicitly supplied, NOT "EN" or "TR" from the originals
    assert blocks[0][0]["gloss"] == ""     # explicitly supplied, NOT "n." or "pl." from the originals


def test_merge_token_rows_rejects_non_adjacent():
    blocks = [[_row("a"), _row("b"), _row("c")]]
    with pytest.raises(ValueError):
        merge_token_rows(blocks, 0, [0, 2], "ac", "TR", "")


def test_merge_token_rows_rejects_cross_sentence():
    blocks = [[_row("a")], [_row("b")]]
    with pytest.raises(ValueError):
        # ridx_list [0, 1] against bidx=0 only has one real row (index 0);
        # index 1 does not exist in block 0 -- rejected as out-of-range,
        # which is exactly the protection that makes a cross-sentence merge
        # (by construction) impossible to express as a single-block call.
        merge_token_rows(blocks, 0, [0, 1], "ab", "TR", "")


def test_merge_token_rows_rejects_meta_row_included():
    blocks = [[_row("a", "TR"), _row("MatrixLang", "TR")]]
    with pytest.raises(ValueError):
        merge_token_rows(blocks, 0, [0, 1], "aMatrixLang", "TR", "")


# --- parse_annotated_text_to_blocks --------------------------------------

def test_parse_annotated_text_to_blocks_two_field_lines():
    text = "Bugun\tTR\nmeeting'e\tMIXED"
    blocks = parse_annotated_text_to_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][0] == {"idx": 0, "token": "Bugun", "label": "TR", "gloss": ""}
    assert blocks[0][1] == {"idx": 0, "token": "meeting'e", "label": "MIXED", "gloss": ""}


def test_parse_annotated_text_to_blocks_three_plus_field_lines_use_middle_two():
    text = "1\tBugun\tTR\tsomegloss"
    blocks = parse_annotated_text_to_blocks(text)
    assert blocks[0][0]["token"] == "Bugun"
    assert blocks[0][0]["label"] == "TR"
    # gloss column in the input is ignored by the parser -- gloss always starts blank
    assert blocks[0][0]["gloss"] == ""


def test_parse_annotated_text_to_blocks_whitespace_fallback():
    text = "Bugun TR"
    blocks = parse_annotated_text_to_blocks(text)
    assert blocks[0][0]["token"] == "Bugun"
    assert blocks[0][0]["label"] == "TR"


def test_parse_annotated_text_to_blocks_unmatched_line_skipped():
    text = "not a valid annotation line"
    blocks = parse_annotated_text_to_blocks(text)
    assert blocks == [[]]


def test_parse_annotated_text_to_blocks_blank_line_separates_blocks():
    text = "a\tTR\n\nb\tEN"
    blocks = parse_annotated_text_to_blocks(text)
    assert len(blocks) == 2
    assert blocks[0][0]["token"] == "a"
    assert blocks[1][0]["token"] == "b"


def test_parse_annotated_text_to_blocks_backfills_extra_headers():
    blocks = parse_annotated_text_to_blocks("a\tTR", extra_headers=["Notes"])
    assert blocks[0][0]["Notes"] == ""


def test_parse_annotated_text_to_blocks_does_not_mutate_extra_headers_arg():
    headers = ["Notes"]
    parse_annotated_text_to_blocks("a\tTR", extra_headers=headers)
    assert headers == ["Notes"]


def test_parse_annotated_text_to_blocks_empty_text():
    assert parse_annotated_text_to_blocks("") == [[]]


# --- make_dataset ----------------------------------------------------------

def test_make_dataset_defaults():
    ds = make_dataset("Data 1")
    assert ds["name"] == "Data 1"
    assert ds["source_text"] == ""
    assert ds["blocks"] == []
    assert ds["extra_headers"] == []
    assert ds["source_filename"] is None
    assert isinstance(ds["id"], str) and ds["id"]


def test_make_dataset_source_filename_optional():
    ds = make_dataset("Data 1", source_filename="example.txt")
    assert ds["source_filename"] == "example.txt"


def test_make_dataset_source_filename_never_stores_a_full_path():
    # Privacy: an absolute path can expose the user's account name and
    # local directory structure if the project is shared. make_dataset is
    # the single choke point that guarantees only the basename is ever
    # stored, even if a caller passes a full path by mistake.
    ds = make_dataset("Data 1", source_filename="/Users/alice/Desktop/private-folder/corpus.txt")
    assert ds["source_filename"] == "corpus.txt"
    assert "/Users/" not in ds["source_filename"]
    assert "private-folder" not in ds["source_filename"]


def test_make_dataset_generates_unique_ids():
    a = make_dataset("Data 1")
    b = make_dataset("Data 2")
    assert a["id"] != b["id"]


def test_make_dataset_explicit_id_preserved():
    ds = make_dataset("Data 1", dataset_id="fixed-id")
    assert ds["id"] == "fixed-id"


def test_make_dataset_deep_copies_blocks():
    original_blocks = [[_row("a", "TR")]]
    ds = make_dataset("Data 1", blocks=original_blocks)
    ds["blocks"][0][0]["label"] = "EN"
    assert original_blocks[0][0]["label"] == "TR"


def test_make_dataset_deep_copies_extra_headers():
    headers = ["Notes"]
    ds = make_dataset("Data 1", extra_headers=headers)
    ds["extra_headers"].append("More")
    assert headers == ["Notes"]


def test_make_dataset_two_datasets_from_same_source_do_not_share_rows():
    src = [[_row("a", "TR")]]
    d1 = make_dataset("Data 1", blocks=src)
    d2 = make_dataset("Data 2", blocks=src)
    d1["blocks"][0][0]["label"] = "MIXED"
    assert d2["blocks"][0][0]["label"] == "TR"


# --- next_default_dataset_name ---------------------------------------------

@pytest.mark.parametrize("existing, expected", [
    ([], "Data 1"),
    (["Data 1"], "Data 2"),
    (["Data 1", "Data 2"], "Data 3"),
    (["Data 1", "Data 3"], "Data 4"),
    (["My Custom Name"], "Data 1"),
    (["Data 1", "My Custom Name"], "Data 2"),
    (["data 1"], "Data 1"),  # case-sensitive: "data 1" isn't the reserved pattern
])
def test_next_default_dataset_name(existing, expected):
    assert next_default_dataset_name(existing) == expected


def test_next_default_dataset_name_none():
    assert next_default_dataset_name(None) == "Data 1"


# --- sanitize_dataset_filename ----------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("Data 1", "Data_1"),
    ("My/Weird:Name*?", "My_Weird_Name"),
    ("  spaced  ", "spaced"),
    ("", "dataset"),
    (None, "dataset"),
    ("Türkçe Veri 1", "Türkçe_Veri_1"),
])
def test_sanitize_dataset_filename(name, expected):
    assert sanitize_dataset_filename(name) == expected


def test_sanitize_dataset_filename_never_empty_for_all_symbols():
    assert sanitize_dataset_filename("!!!") == "dataset"


# --- datasets_to_payload / datasets_from_payload ----------------------------

def test_datasets_to_payload_stamps_current_schema_version():
    payload = datasets_to_payload([make_dataset("Data 1")], active_index=0)
    assert payload["version"] == CURRENT_PROJECT_SCHEMA_VERSION == 2


def test_datasets_to_payload_does_not_serialize_undo_stacks():
    # Positional undo/redo records are session-only (see
    # App._sync_active_dataset_from_live) and must never be written to disk.
    ds = make_dataset("Data 1")
    ds["uid_undo_stack"] = [{"bidx": 0, "ridx": 0, "old": {}, "new": {}}]
    ds["merge_cells_undo_stack"] = [{"bidx": 0, "old_rows": []}]
    payload = datasets_to_payload([ds], active_index=0)
    saved_ds = payload["datasets"][0]
    assert "uid_undo_stack" not in saved_ds
    assert "merge_cells_undo_stack" not in saved_ds


def test_datasets_round_trip_preserves_data():
    ds1 = make_dataset("Data 1", "hello", [[_row("a", "TR")]], ["Notes"])
    ds2 = make_dataset("Data 2", "world", [[_row("b", "EN")]])
    payload = {"version": 1, "cfg": {}}
    payload.update(datasets_to_payload([ds1, ds2], active_index=1))
    assert payload["version"] == CURRENT_PROJECT_SCHEMA_VERSION  # stamped by datasets_to_payload

    restored, active_index = datasets_from_payload(payload)
    assert active_index == 1
    assert len(restored) == 2
    assert restored[0]["name"] == "Data 1"
    assert restored[0]["source_text"] == "hello"
    assert restored[0]["blocks"][0][0]["token"] == "a"
    assert restored[0]["extra_headers"] == ["Notes"]
    assert restored[1]["name"] == "Data 2"


def test_datasets_round_trip_preserves_source_filename_when_set():
    ds = make_dataset("Data 1", "hello", source_filename="example.txt")
    payload = datasets_to_payload([ds], active_index=0)
    assert payload["datasets"][0]["source_filename"] == "example.txt"

    restored, _ = datasets_from_payload(payload)
    assert restored[0]["source_filename"] == "example.txt"


def test_datasets_to_payload_omits_source_filename_when_unset():
    ds = make_dataset("Data 1", "hello")
    payload = datasets_to_payload([ds], active_index=0)
    assert "source_filename" not in payload["datasets"][0]


def test_datasets_from_payload_v2_without_source_filename_still_loads():
    # Existing version-2 projects saved before source_filename existed.
    payload = {
        "version": 2,
        "datasets": [{"name": "Data 1", "blocks": [[_row("a", "TR")]]}],
        "active_dataset_index": 0,
    }
    restored, _ = datasets_from_payload(payload)
    assert restored[0]["source_filename"] is None


def test_datasets_from_payload_reopening_never_requires_source_file_to_exist():
    payload = {
        "version": 2,
        "datasets": [{
            "name": "Data 1",
            "source_text": "the real content",
            "source_filename": "anywhere.txt",
            "blocks": [[_row("a", "TR")]],
        }],
        "active_dataset_index": 0,
    }
    restored, _ = datasets_from_payload(payload)  # must not raise or touch the filesystem
    assert restored[0]["source_text"] == "the real content"
    assert restored[0]["source_filename"] == "anywhere.txt"


def test_datasets_from_payload_invalid_source_filename_type_raises():
    payload = {
        "version": 2,
        "datasets": [{"name": "Data 1", "blocks": [], "source_filename": 123}],
    }
    with pytest.raises(ValueError):
        datasets_from_payload(payload)


# --- Privacy: source_filename must never carry a full local path ----------

def test_datasets_to_payload_serialized_json_contains_only_basename_not_full_path():
    # The exact regression this fix exists for: a shared .trenproj must
    # never leak the user's account name or local directory structure.
    full_path = "/Users/alice/Desktop/private-folder/corpus.txt"
    ds = make_dataset("Data 1", "hello", source_filename=full_path)
    payload = datasets_to_payload([ds], active_index=0)

    serialized = json.dumps(payload)
    assert '"corpus.txt"' in serialized
    assert "/Users/" not in serialized
    assert "private-folder" not in serialized
    assert full_path not in serialized
    assert payload["datasets"][0]["source_filename"] == "corpus.txt"


def test_datasets_from_payload_migrates_legacy_source_path_to_basename_only():
    # A development-only payload from before this fix might still use the
    # old "source_path" field name with a full path. It must be migrated to
    # source_filename (basename only) in memory, and never re-persisted as
    # "source_path" or as a full path on the next save.
    legacy_full_path = "/Users/alice/Desktop/private-folder/corpus.txt"
    payload = {
        "version": 2,
        "datasets": [{
            "name": "Data 1",
            "source_text": "hello",
            "source_path": legacy_full_path,
            "blocks": [],
        }],
        "active_dataset_index": 0,
    }
    restored, active_index = datasets_from_payload(payload)
    assert restored[0]["source_filename"] == "corpus.txt"
    assert "source_path" not in restored[0]

    re_saved = datasets_to_payload(restored, active_index)
    serialized = json.dumps(re_saved)
    assert "source_path" not in serialized
    assert "/Users/" not in serialized
    assert "private-folder" not in serialized
    assert re_saved["datasets"][0]["source_filename"] == "corpus.txt"


def test_datasets_from_payload_malformed_legacy_source_path_ignored_safely():
    payload = {
        "version": 2,
        "datasets": [{"name": "Data 1", "blocks": [], "source_path": 12345}],
    }
    restored, _ = datasets_from_payload(payload)  # must not raise
    assert restored[0]["source_filename"] is None


def test_datasets_from_payload_legacy_single_dataset_becomes_data_1():
    payload = {
        "version": 1,
        "input_text": "hello world",
        "blocks": [[_row("hello", "EN")]],
        "extra_headers": ["Notes"],
    }
    datasets, active_index = datasets_from_payload(payload)
    assert active_index == 0
    assert len(datasets) == 1
    assert datasets[0]["name"] == "Data 1"
    assert datasets[0]["source_text"] == "hello world"
    assert datasets[0]["blocks"][0][0]["token"] == "hello"
    assert datasets[0]["extra_headers"] == ["Notes"]


def test_datasets_from_payload_legacy_missing_keys_use_safe_defaults():
    datasets, active_index = datasets_from_payload({"version": 1})
    assert active_index == 0
    assert datasets[0]["name"] == "Data 1"
    assert datasets[0]["blocks"] == []
    assert datasets[0]["source_text"] == ""


# --- .trenproj schema version handling --------------------------------

def test_datasets_from_payload_version_1_uses_legacy_path_even_with_extra_keys():
    payload = {"version": 1, "input_text": "hi", "blocks": [[_row("hi", "EN")]]}
    datasets, active_index = datasets_from_payload(payload)
    assert active_index == 0
    assert datasets[0]["name"] == "Data 1"
    assert datasets[0]["source_text"] == "hi"


def test_datasets_from_payload_version_2_uses_datasets_path():
    payload = {
        "version": 2,
        "datasets": [{"name": "Data 1", "blocks": [[_row("a", "TR")]]}],
        "active_dataset_index": 0,
    }
    datasets, active_index = datasets_from_payload(payload)
    assert len(datasets) == 1
    assert datasets[0]["blocks"][0][0]["token"] == "a"


def test_datasets_from_payload_missing_version_with_datasets_key_rejected():
    # Missing "version" defaults to 1, and version 1 REQUIRES the legacy
    # shape -- a "datasets" key present (e.g. a hand-edited or
    # half-migrated file) must be rejected as malformed, not silently
    # accepted via shape-detection.
    payload = {"datasets": [{"name": "Data 1", "blocks": [[_row("a", "TR")]]}]}
    with pytest.raises(ValueError):
        datasets_from_payload(payload)


def test_datasets_from_payload_version_1_with_datasets_key_rejected():
    payload = {
        "version": 1,
        "datasets": [{"name": "Data 1", "blocks": [[_row("a", "TR")]]}],
    }
    with pytest.raises(ValueError):
        datasets_from_payload(payload)


def test_datasets_from_payload_missing_version_with_legacy_shape_is_data_1():
    payload = {"input_text": "hi", "blocks": [[_row("hi", "EN")]]}
    datasets, active_index = datasets_from_payload(payload)
    assert active_index == 0
    assert datasets[0]["name"] == "Data 1"
    assert datasets[0]["source_text"] == "hi"


def test_datasets_from_payload_version_2_missing_datasets_key_raises():
    with pytest.raises(ValueError):
        datasets_from_payload({"version": 2, "input_text": "hi", "blocks": []})


@pytest.mark.parametrize("bad_version", ["two", 2.0, None, [2], True])
def test_datasets_from_payload_malformed_version_raises(bad_version):
    with pytest.raises(ValueError):
        datasets_from_payload({"version": bad_version, "datasets": [{"name": "Data 1", "blocks": []}]})


@pytest.mark.parametrize("future_version", [0, 3, 99])
def test_datasets_from_payload_unsupported_future_version_raises(future_version):
    with pytest.raises(ValueError):
        datasets_from_payload({"version": future_version, "datasets": [{"name": "Data 1", "blocks": []}]})


def test_supported_project_schema_versions_matches_current():
    assert CURRENT_PROJECT_SCHEMA_VERSION in SUPPORTED_PROJECT_SCHEMA_VERSIONS
    assert SUPPORTED_PROJECT_SCHEMA_VERSIONS == (1, 2)


@pytest.mark.parametrize("payload", [
    {"version": 2, "datasets": []},
    {"version": 2, "datasets": "not-a-list"},
    {"version": 2, "datasets": [{"name": "", "blocks": []}]},
    {"version": 2, "datasets": [{"blocks": []}]},
    {"version": 2, "datasets": [{"name": "Data 1", "blocks": "not-a-list"}]},
    {"version": 2, "datasets": [{"name": "Data 1", "blocks": [], "extra_headers": "not-a-list"}]},
    {"version": 2, "datasets": [{"name": "Data 1", "blocks": [], "source_text": 123}]},
    {"version": 2, "datasets": ["not-an-object"]},
])
def test_datasets_from_payload_malformed_raises_value_error(payload):
    with pytest.raises(ValueError):
        datasets_from_payload(payload)


def test_datasets_from_payload_out_of_range_active_index_falls_back_to_zero():
    payload = {"version": 2, "datasets": [{"name": "Data 1", "blocks": []}], "active_dataset_index": 99}
    datasets, active_index = datasets_from_payload(payload)
    assert active_index == 0


def test_datasets_from_payload_duplicate_ids_are_regenerated():
    payload = {
        "version": 2,
        "datasets": [
            {"id": "same", "name": "Data 1", "blocks": []},
            {"id": "same", "name": "Data 2", "blocks": []},
        ]
    }
    datasets, _ = datasets_from_payload(payload)
    assert datasets[0]["id"] != datasets[1]["id"]


def test_datasets_from_payload_restored_datasets_do_not_share_row_dicts():
    shared_row_source = [[_row("a", "TR")]]
    payload = {
        "version": 2,
        "datasets": [
            {"name": "Data 1", "blocks": shared_row_source},
            {"name": "Data 2", "blocks": shared_row_source},
        ]
    }
    datasets, _ = datasets_from_payload(payload)
    datasets[0]["blocks"][0][0]["label"] = "MIXED"
    assert datasets[1]["blocks"][0][0]["label"] == "TR"


# --- blocks_to_export_rows ---------------------------------------------------

def test_blocks_to_export_rows_basic_with_separator():
    blocks = [[_row("a", "TR")], [_row("b", "EN")]]
    rows = blocks_to_export_rows(blocks, [])
    assert rows == [["1", "a", "TR", ""], ["", "", "", ""], ["2", "b", "EN", ""]]


def test_blocks_to_export_rows_no_trailing_separator():
    blocks = [[_row("a", "TR")]]
    rows = blocks_to_export_rows(blocks, [])
    assert rows == [["1", "a", "TR", ""]]


def test_blocks_to_export_rows_includes_extra_headers():
    blocks = [[{"idx": 0, "token": "a", "label": "TR", "gloss": "", "Notes": "n1"}]]
    rows = blocks_to_export_rows(blocks, ["Notes"])
    assert rows == [["1", "a", "TR", "", "n1"]]


def test_blocks_to_export_rows_does_not_mutate_input():
    blocks = [[_row("a", "TR", idx="")]]
    blocks_to_export_rows(blocks, [])
    assert blocks[0][0]["idx"] == ""


# --- blocks_to_conll -----------------------------------------------------

def _conll_block(sent_id="1", matrix="TR", embed="EN", tokens=(("Bugun", "TR", ""), ("meeting'e", "MIXED", "meeting-DAT"))):
    rows = [{"idx": "", "token": "SentenceID", "label": sent_id, "gloss": ""}]
    for tok, lab, glo in tokens:
        rows.append(_row(tok, lab, glo))
    if matrix:
        rows.append({"idx": "", "token": "MatrixLang", "label": matrix, "gloss": ""})
    if embed:
        rows.append({"idx": "", "token": "EmbedLang", "label": embed, "gloss": ""})
    return rows


def test_blocks_to_conll_header_lines():
    text = blocks_to_conll([_conll_block()])
    lines = text.splitlines()
    assert lines[0] == "# TREN CoNLL export"
    assert lines[1] == "# columns = TokenIndex Token Label Gloss"


def test_blocks_to_conll_sentence_comments_and_tab_separated_tokens():
    text = blocks_to_conll([_conll_block()])
    assert "# sent_id = 1" in text
    assert "# matrix_lang = TR" in text
    assert "# embedded_lang = EN" in text
    assert "1\tBugun\tTR\t_" in text
    assert "2\tmeeting'e\tMIXED\tmeeting-DAT" in text


def test_blocks_to_conll_empty_gloss_becomes_underscore():
    text = blocks_to_conll([_conll_block(tokens=(("a", "TR", ""),))])
    assert "\ta\tTR\t_" in text


def test_blocks_to_conll_blank_line_separates_sentences():
    b1 = _conll_block(sent_id="1")
    b2 = _conll_block(sent_id="2")
    text = blocks_to_conll([b1, b2])
    body = text.split("# columns = TokenIndex Token Label Gloss\n\n", 1)[1]
    sentences = body.rstrip("\n").split("\n\n")
    assert len(sentences) == 2
    assert sentences[0].startswith("# sent_id = 1")
    assert sentences[1].startswith("# sent_id = 2")


def test_blocks_to_conll_skips_empty_blocks():
    text = blocks_to_conll([_conll_block(sent_id="1"), [], _conll_block(sent_id="2")])
    assert text.count("# sent_id") == 2


def test_blocks_to_conll_no_metadata_rows_as_lexical_tokens():
    text = blocks_to_conll([_conll_block()])
    for ln in text.splitlines():
        if "\t" in ln and not ln.startswith("#"):
            fields = ln.split("\t")
            assert fields[1] not in ("SentenceID", "MatrixLang", "EmbedLang")


def test_blocks_to_conll_missing_matrix_embed_meta_rows_omits_those_comments():
    text = blocks_to_conll([_conll_block(matrix=None, embed=None)])
    assert "matrix_lang" not in text
    assert "embedded_lang" not in text


def test_blocks_to_conll_falls_back_to_position_when_no_sentence_id_row():
    rows = [_row("a", "TR")]
    text = blocks_to_conll([rows])
    assert "# sent_id = 1" in text


def test_blocks_to_conll_does_not_mutate_input():
    block = _conll_block()
    before = copy.deepcopy(block)
    blocks_to_conll([block])
    assert block == before


def test_blocks_to_conll_escapes_embedded_tabs_and_newlines():
    rows = [_row("weird\ttoken\nwith\rnewline", "TR", "gl\tos\ns")]
    text = blocks_to_conll([rows])
    line = [ln for ln in text.splitlines() if ln.startswith("1\t")][0]
    fields = line.split("\t")
    assert len(fields) == 4  # token/label/gloss internal tabs must not add fields
    assert "\n" not in line and "\r" not in line


def test_blocks_to_conll_unicode_preserved():
    rows = [_row("Türkçe", "TR")]
    text = blocks_to_conll([rows])
    assert "Türkçe" in text


def test_blocks_to_conll_deterministic():
    blocks = [_conll_block()]
    assert blocks_to_conll(blocks) == blocks_to_conll(blocks)


def test_blocks_to_conll_empty_project():
    text = blocks_to_conll([])
    assert text.startswith("# TREN CoNLL export\n# columns = TokenIndex Token Label Gloss\n")
    assert "sent_id" not in text


# --- blocks_to_jsonl / blocks_to_jsonl_records -------------------------------

def test_blocks_to_jsonl_one_line_per_sentence_and_valid_json():
    blocks = [_conll_block(sent_id="1"), _conll_block(sent_id="2")]
    text = blocks_to_jsonl(blocks, "Data 1")
    lines = text.rstrip("\n").split("\n")
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)  # must not raise


def test_blocks_to_jsonl_token_schema():
    blocks = [_conll_block()]
    records = blocks_to_jsonl_records(blocks, "Data 1")
    rec = records[0]
    assert rec["sentence_id"] == "1"
    assert rec["dataset"] == "Data 1"
    assert rec["matrix_lang"] == "TR"
    assert rec["embedded_lang"] == "EN"
    assert rec["source_text"] == "Bugun meeting'e"
    assert rec["tokens"] == [
        {"index": 1, "token": "Bugun", "label": "TR", "gloss": ""},
        {"index": 2, "token": "meeting'e", "label": "MIXED", "gloss": "meeting-DAT"},
    ]


def test_blocks_to_jsonl_excludes_meta_rows_from_tokens():
    records = blocks_to_jsonl_records([_conll_block()], "Data 1")
    tokens_text = [t["token"] for t in records[0]["tokens"]]
    assert "SentenceID" not in tokens_text
    assert "MatrixLang" not in tokens_text
    assert "EmbedLang" not in tokens_text


def test_blocks_to_jsonl_empty_gloss_is_empty_string_not_underscore():
    records = blocks_to_jsonl_records([[_row("a", "TR", "")]], "Data 1")
    assert records[0]["tokens"][0]["gloss"] == ""


def test_blocks_to_jsonl_skips_empty_blocks():
    records = blocks_to_jsonl_records([_conll_block(), [], _conll_block()], "Data 1")
    assert len(records) == 2


def test_blocks_to_jsonl_unicode_not_ascii_escaped():
    text = blocks_to_jsonl([[_row("Türkçe", "TR")]], "Data 1")
    assert "Türkçe" in text
    assert "\\u" not in text


def test_blocks_to_jsonl_deterministic():
    blocks = [_conll_block()]
    assert blocks_to_jsonl(blocks, "Data 1") == blocks_to_jsonl(blocks, "Data 1")


def test_blocks_to_jsonl_does_not_mutate_input():
    block = _conll_block()
    before = copy.deepcopy(block)
    blocks_to_jsonl([block], "Data 1")
    assert block == before


def test_blocks_to_jsonl_no_private_or_internal_fields():
    records = blocks_to_jsonl_records([_conll_block()], "Data 1")
    rec = records[0]
    assert set(rec.keys()) == {"sentence_id", "dataset", "source_text", "matrix_lang", "embedded_lang", "tokens"}
    for tok in rec["tokens"]:
        assert set(tok.keys()) == {"index", "token", "label", "gloss"}


def test_blocks_to_jsonl_empty_project():
    assert blocks_to_jsonl([], "Data 1") == ""
