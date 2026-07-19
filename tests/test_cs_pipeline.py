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


# --- _split_mixed_apostrophe ------------------------------------------------
# Neither this function nor _parse_tr_suffixes_full touches `self` at all, so
# a bare Annotator.__new__() instance with no attributes set is sufficient.

@pytest.mark.parametrize("token, expected", [
    ("meeting'e", ("meeting", "e")),      # straight apostrophe
    ("meeting’e", ("meeting", "e")),      # curly/typographic apostrophe (U+2019)
    ("nothingatall", (None, None)),       # no apostrophe at all
    ("'twas", (None, None)),              # leading apostrophe -> empty first part
    ("word'", (None, None)),              # trailing apostrophe -> empty second part
    ("a'b'c", (None, None)),              # 2+ apostrophes -> more than 2 parts
], ids=[
    "straight_apostrophe_split", "curly_apostrophe_split", "no_apostrophe",
    "leading_apostrophe_empty_base", "trailing_apostrophe_empty_suffix",
    "multiple_apostrophes",
])
def test_split_mixed_apostrophe(token, expected):
    obj = _make_annotator()
    assert obj._split_mixed_apostrophe(token) == expected


@pytest.mark.parametrize("token", [
    "he's", "we're", "I've", "I'm", "we'll", "he'd", "don't",
], ids=["s", "re", "ve", "m", "ll", "d", "t_via_dont"])
def test_split_mixed_apostrophe_english_contractions_rejected(token):
    obj = _make_annotator()
    assert obj._split_mixed_apostrophe(token) == (None, None)


def test_split_mixed_apostrophe_contraction_check_is_case_insensitive():
    obj = _make_annotator()
    assert obj._split_mixed_apostrophe("word'S") == (None, None)


def test_split_mixed_apostrophe_nt_branch_is_unreachable():
    # Documents a confirmed dead branch: `sfx.endswith("n't")` can never be
    # True, because re.split(r"[’']", token) splits on every apostrophe,
    # so the suffix half can never itself contain an apostrophe when exactly
    # 2 parts result (the precondition to reach this check at all). Real
    # contractions like "don't" split to suffix "t", already caught by the
    # EN_CONTRACTIONS set membership check. Not a bug to fix here -- locking
    # in the current, verified behavior.
    obj = _make_annotator()
    assert obj._split_mixed_apostrophe("don't")[1] != "n't"


# --- _parse_tr_suffixes_full -------------------------------------------------

@pytest.mark.parametrize("suffix, expected", [
    ("", ([], set(), set(), set())),
    ("e", (["e"], {"Case=Dat"}, set(), set())),
    ("ne", (["ne"], {"Case=Dat"}, set(), set())),   # buffer-n dative, one unit
    ("na", (["na"], {"Case=Dat"}, set(), set())),   # buffer-n dative, one unit
    ("ımız", (["ımız"], {"Poss=Yes", "Person[psor]=1", "Number[psor]=Plur"}, set(), set())),
    ("lar", (["lar"], {"Number=Plur"}, set(), set())),
    ("lik", (["lik"], set(), {"Deriv=LIK", "DerivPOS=NOUN"}, set())),
    ("xyz", (["xyz"], set(), {"Unparsed=Leftover"}, set())),
], ids=[
    "empty_string", "single_case_ending", "buffer_n_dative_ne", "buffer_n_dative_na",
    "possessive_long", "plural", "derivational", "unparseable_leftover",
])
def test_parse_tr_suffixes_full(suffix, expected):
    obj = _make_annotator()
    assert obj._parse_tr_suffixes_full(suffix) == expected


def test_parse_tr_suffixes_full_multistage_chain():
    # Exercises all four stages chaining together in one input:
    # deriv ("lık") + plural ("lar") + case ("ı").
    obj = _make_annotator()
    segments, ud, deriv, amb = obj._parse_tr_suffixes_full("lıkları")
    assert segments == ["lık", "lar", "ı"]
    assert ud == {"Case=Acc", "Number=Plur"}
    assert deriv == {"Deriv=LIK", "DerivPOS=NOUN"}
    assert amb == set()


@pytest.mark.parametrize("suffix", ["ıım", "iim", "uum", "uüm"], ids=[
    "ambiguous_ii_im", "ambiguous_i_im", "ambiguous_u_um", "ambiguous_u_umlaut_m",
])
def test_parse_tr_suffixes_full_ambiguous_vowel_branch_is_reachable(suffix):
    # Confirms the Amb=P3sg_or_Acc branch is reachable (verified by brute
    # force search), contrary to what a static read might suggest: stage 1's
    # CASE_ENDINGS already includes bare i/i/u/u-umlaut, but stage 2 can
    # strip a POSS_SHORT suffix (e.g. "ım") and expose a *new* trailing
    # vowel that stage 1 never had the chance to see, since stage 1 already
    # finished running before stage 2 started.
    obj = _make_annotator()
    segments, ud, deriv, amb = obj._parse_tr_suffixes_full(suffix)
    assert amb == {"Amb=P3sg_or_Acc"}
