import pytest

from annotation_model import is_meta_row_token, freq_normalize_token


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
