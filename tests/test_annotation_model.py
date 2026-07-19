import copy

import pytest

from annotation_model import (
    is_meta_row_token,
    freq_normalize_token,
    compute_word_frequencies,
    sheet_rows_to_txt,
)


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
