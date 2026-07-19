import copy

import pytest

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
