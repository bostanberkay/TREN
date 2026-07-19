from unittest import mock

import pytest

from cs_pipeline import Annotator, DEFAULTS


def _make_annotator(turkish_top=(), turkish_all=(), english_words=()):
    # Bypass __init__ entirely -- no frequency-file reads, no
    # fasttext.load_model() call. _choose_label only needs these three sets.
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set(turkish_top)
    obj.turkish_freq_all = set(turkish_all)
    obj.english_freq_words = set(english_words)
    return obj


# --- ordinary branch cases, one per lexicon tier --------------------------

def test_choose_label_turkish_top_1000():
    obj = _make_annotator(turkish_top={"bugun"})
    assert obj._choose_label("bugun", DEFAULTS) == "TR"


def test_choose_label_english_list():
    obj = _make_annotator(english_words={"stressed"})
    assert obj._choose_label("stressed", DEFAULTS) == "EN"


def test_choose_label_turkish_full_list():
    obj = _make_annotator(turkish_all={"cocuk"})
    assert obj._choose_label("cocuk", DEFAULTS) == "TR"


# --- precedence between lexicon tiers --------------------------------------

def test_choose_label_top1000_wins_over_english_and_full_list():
    # A token present in all three lexicons must resolve via branch 1 (TR),
    # not branch 2 (EN) or branch 3 (TR-full). This is the exact structural
    # rule behind the documented "I"/"i" mislabeling (see below) -- tested
    # here with a synthetic fixture, not the real 50k-line resource files.
    obj = _make_annotator(turkish_top={"i"}, english_words={"i"}, turkish_all={"i"})
    assert obj._choose_label("i", DEFAULTS) == "TR"


def test_choose_label_english_list_wins_over_turkish_full_list():
    # A token in both the English list and the Turkish full list (but not
    # top-1000) must resolve EN -- branch 2 is checked before branch 3.
    obj = _make_annotator(english_words={"gitmem"}, turkish_all={"gitmem"})
    assert obj._choose_label("gitmem", DEFAULTS) == "EN"


# --- fastText fallback: ordinary + inclusive/exclusive threshold boundary --

@pytest.mark.parametrize("ft_result, expected", [
    (("EN", 0.95), "EN"),           # clearly above FT_EN_MIN
    (("EN", 0.80), "EN"),           # exactly FT_EN_MIN -- inclusive (>=)
    (("EN", 0.7999999), "UID"),     # just under -- must not be EN
    (("TR", 0.95), "TR"),           # clearly above FT_TR_MIN
    (("TR", 0.80), "TR"),           # exactly FT_TR_MIN -- inclusive (>=)
    (("TR", 0.7999999), "UID"),     # just under -- must not be TR
    (("EN", 0.5), "UID"),           # below threshold: no fallback to TR
    (("FR", 0.99), "UID"),          # neither EN nor TR, regardless of confidence
], ids=[
    "en_clearly_above", "en_exact_boundary", "en_just_below",
    "tr_clearly_above", "tr_exact_boundary", "tr_just_below",
    "en_below_threshold_no_tr_fallback", "third_language_always_uid",
])
def test_choose_label_fasttext_fallback(ft_result, expected):
    obj = _make_annotator()  # token in no lexicon -- forces the fastText path
    with mock.patch.object(obj, "_ft_predict", return_value=ft_result):
        assert obj._choose_label("unseen_token", DEFAULTS) == expected


def test_choose_label_fasttext_fallback_empty_token():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.5)):
        assert obj._choose_label("", DEFAULTS) == "UID"


def test_choose_label_respects_custom_cfg_thresholds():
    # Confirms FT_EN_MIN/FT_TR_MIN are read from cfg, not hardcoded --
    # 0.6 would be UID under the default 0.80 threshold but EN under a
    # looser custom one.
    obj = _make_annotator()
    cfg = dict(DEFAULTS, FT_EN_MIN=0.5)
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.6)):
        assert obj._choose_label("unseen_token", cfg) == "EN"


# --- priority order: lexicon hits must short-circuit before fastText ------

@pytest.mark.parametrize("make_obj, token", [
    (lambda: _make_annotator(turkish_top={"bugun"}), "bugun"),
    (lambda: _make_annotator(english_words={"stressed"}), "stressed"),
    (lambda: _make_annotator(turkish_all={"cocuk"}), "cocuk"),
], ids=["turkish_top_short_circuits", "english_list_short_circuits", "turkish_full_short_circuits"])
def test_choose_label_lexicon_hit_never_calls_ft_predict(make_obj, token):
    obj = make_obj()
    with mock.patch.object(obj, "_ft_predict") as mocked:
        obj._choose_label(token, DEFAULTS)
    mocked.assert_not_called()
