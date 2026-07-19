from unittest import mock

import pytest

import cs_pipeline
from cs_pipeline import Annotator, DEFAULTS, tokenize


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


# --- _detect_mixed_no_apostrophe --------------------------------------------
# Uses Annotator.__new__(Annotator) + synthetic turkish_freq_all /
# english_freq_words, and mock.patch.object for _ft_predict, matching the
# same pattern used for _choose_label.

def test_detect_mixed_no_apostrophe_whole_token_already_turkish():
    obj = _make_annotator(turkish_all={"evim"})
    assert obj._detect_mixed_no_apostrophe("evim", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_base_resolved_via_english_lexicon():
    obj = _make_annotator(english_words={"stress"})
    assert obj._detect_mixed_no_apostrophe("stressim", DEFAULTS) == ("stress", "im")


def test_detect_mixed_no_apostrophe_base_resolved_via_fasttext():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.9)):
        assert obj._detect_mixed_no_apostrophe("bossum", DEFAULTS) == ("boss", "um")


@pytest.mark.parametrize("prob, expected", [
    (0.80, ("boss", "um")),        # exactly FT_EN_MIN -- inclusive (>=)
    (0.7999999, (None, None)),     # just under -- must not be accepted
], ids=["exact_boundary", "just_below_boundary"])
def test_detect_mixed_no_apostrophe_fasttext_threshold_boundary(prob, expected):
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", prob)):
        assert obj._detect_mixed_no_apostrophe("bossum", DEFAULTS) == expected


def test_detect_mixed_no_apostrophe_no_valid_split():
    obj = _make_annotator(english_words={"hello"})
    assert obj._detect_mixed_no_apostrophe("hello", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_longest_suffix_candidate_wins():
    # "lilanın" ends in both "nın" (3 chars, a real CASE_ENDINGS key) and
    # "ın" (2 chars, also a real CASE_ENDINGS key) -- both bases are valid
    # English words here, so only trial order decides the result. Candidates
    # are tried longest-first, so "nın" (base "lila") must win over "ın"
    # (base "lilan").
    obj = _make_annotator(english_words={"lila", "lilan"})
    assert obj._detect_mixed_no_apostrophe("lilanın", DEFAULTS) == ("lila", "nın")


def test_detect_mixed_no_apostrophe_first_valid_candidate_wins_after_longest_fails():
    # Same "nın"/"ın" overlap as above, but this time only the SHORTER
    # split's base ("lilan") is a known English word -- the longer split's
    # base ("lila") is not, and is mocked to clearly fail the fastText
    # fallback too. The loop must continue past the failed longest candidate
    # to the next-shorter one, not stop at the first (longest) attempt.
    obj = _make_annotator(english_words={"lilan"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.1)):
        assert obj._detect_mixed_no_apostrophe("lilanın", DEFAULTS) == ("lilan", "ın")


def test_detect_mixed_no_apostrophe_base_length_exactly_2_accepted():
    obj = _make_annotator(english_words={"ok"})
    assert obj._detect_mixed_no_apostrophe("oklar", DEFAULTS) == ("ok", "lar")


def test_detect_mixed_no_apostrophe_base_length_1_rejected():
    obj = _make_annotator(english_words={"a"})
    assert obj._detect_mixed_no_apostrophe("aim", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_single_char_suffix_candidates_excluded():
    # "stresse" = "stress" + "e". "e" is a valid CASE_ENDINGS key (Case=Dat)
    # and would otherwise split into a real English base, but candidates
    # shorter than 2 chars are structurally skipped (`if len(suf) < 2:
    # continue`), so no split is ever attempted here.
    obj = _make_annotator(english_words={"stress"})
    assert obj._detect_mixed_no_apostrophe("stresse", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_yn_initial_suffix_vowel_final_base_accepted():
    obj = _make_annotator(english_words={"feta"})
    assert obj._detect_mixed_no_apostrophe("fetayla", DEFAULTS) == ("feta", "yla")


def test_detect_mixed_no_apostrophe_yn_initial_suffix_consonant_final_base_rejected():
    obj = _make_annotator(english_words={"boss"})
    assert obj._detect_mixed_no_apostrophe("bossya", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_suffix_must_produce_a_feature():
    # Every real suffix dictionary key (len >= 2) always parses into at
    # least one UD/deriv/ambiguity feature via _parse_tr_suffixes_full, so
    # this guard never actually rejects a real candidate in practice. To
    # exercise the guard itself (not fix or reinterpret it), force a
    # no-feature parse via mocking and confirm the candidate is rejected.
    obj = _make_annotator(english_words={"boss"})
    with mock.patch.object(obj, "_parse_tr_suffixes_full", return_value=([], set(), set(), set())):
        assert obj._detect_mixed_no_apostrophe("bossum", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_mixed_strict_rejects_turkish_known_base():
    obj = _make_annotator(turkish_all={"kritik"}, english_words={"kritik"})
    assert obj._detect_mixed_no_apostrophe("kritikim", DEFAULTS) == (None, None)


def test_detect_mixed_no_apostrophe_mixed_strict_false_accepts_same_candidate():
    obj = _make_annotator(turkish_all={"kritik"}, english_words={"kritik"})
    cfg = dict(DEFAULTS, MIXED_STRICT=False)
    assert obj._detect_mixed_no_apostrophe("kritikim", cfg) == ("kritik", "im")


def test_detect_mixed_no_apostrophe_ft_predict_not_called_when_base_in_lexicon():
    obj = _make_annotator(english_words={"stress"})
    with mock.patch.object(obj, "_ft_predict") as mocked:
        result = obj._detect_mixed_no_apostrophe("stressim", DEFAULTS)
    assert result == ("stress", "im")
    mocked.assert_not_called()


# --- _build_ne_map -----------------------------------------------------
# Touches no `self.*` state at all -- Annotator.__new__() with zero
# attributes set is sufficient. Only doc.ents (a list) and each entity's
# .text (a string) are ever read; nothing about Stanza's real API surface
# (spans, offsets, per-token objects) matters to this function.

class _FakeEnt:
    def __init__(self, text):
        self.text = text


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


@pytest.mark.parametrize("ents", [None, []], ids=["ents_none", "ents_empty_list"])
def test_build_ne_map_falsy_ents(ents):
    obj = _make_annotator()
    assert obj._build_ne_map(_FakeDoc(ents), ["a", "b"]) == {}


def test_build_ne_map_doc_missing_ents_attribute():
    class NoEntsDoc:
        pass
    obj = _make_annotator()
    assert obj._build_ne_map(NoEntsDoc(), ["a", "b"]) == {}


def test_build_ne_map_normal_multiword_entity():
    obj = _make_annotator()
    tokens = tokenize("New York is nice")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("New York")]), tokens) == {
        "New": "NE", "York": "NE",
    }


def test_build_ne_map_single_word_entity():
    obj = _make_annotator()
    tokens = tokenize("Ankara is nice")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Ankara")]), tokens) == {"Ankara": "NE"}


def test_build_ne_map_empty_entity_text():
    obj = _make_annotator()
    tokens = tokenize("New York is nice")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("")]), tokens) == {}


def test_build_ne_map_malformed_entity_missing_text_attribute_raises():
    # Documents current, unguarded behavior: there is no try/except here, so
    # a malformed entity object propagates a bare AttributeError. Not a bug
    # to fix in this step -- locking in the current behavior.
    class BadEnt:
        pass
    obj = _make_annotator()
    tokens = tokenize("New York is nice")
    with pytest.raises(AttributeError):
        obj._build_ne_map(_FakeDoc([BadEnt()]), tokens)


def test_build_ne_map_case_sensitive_matching():
    obj = _make_annotator()
    tokens = tokenize("Washington met washington")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Washington")]), tokens) == {
        "Washington": "NE",
    }


def test_build_ne_map_duplicate_token_text_limitation():
    # Documents a real, current alignment limitation: matching is by literal
    # token TEXT, not by span/position. A token string that legitimately
    # appears once as part of an entity and once elsewhere in the same line
    # cannot be distinguished here -- both would be treated as NE when this
    # map is consumed by annotate()'s per-token loop. Not a bug to fix in
    # this step -- locking in the current behavior.
    obj = _make_annotator()
    tokens = tokenize("Paris loves Paris")
    ne_map = obj._build_ne_map(_FakeDoc([_FakeEnt("Paris")]), tokens)
    assert ne_map == {"Paris": "NE"}
    assert all(tok in ne_map for tok in tokens if tok == "Paris")


def test_build_ne_map_entity_text_matches_nothing_in_line():
    obj = _make_annotator()
    tokens = tokenize("unrelated sentence here")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Berlin")]), tokens) == {}


def test_build_ne_map_multiple_entities_aggregate_via_union():
    obj = _make_annotator()
    tokens = tokenize("Ankara and Istanbul are cities")
    ne_map = obj._build_ne_map(
        _FakeDoc([_FakeEnt("Ankara"), _FakeEnt("Istanbul")]), tokens
    )
    assert ne_map == {"Ankara": "NE", "Istanbul": "NE"}


def test_build_ne_map_entity_text_with_trailing_punctuation():
    # The entity-piece regex strips punctuation not attached to \w chars,
    # so a trailing comma in the entity's raw text doesn't prevent matching.
    obj = _make_annotator()
    tokens = tokenize("I love New York today")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("New York,")]), tokens) == {
        "New": "NE", "York": "NE",
    }


def test_build_ne_map_apostrophe_containing_entity_name():
    obj = _make_annotator()
    tokens = tokenize("O'Brien lives here")
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("O'Brien")]), tokens) == {
        "O'Brien": "NE",
    }


def test_build_ne_map_lone_apostrophe_entity_text_regex_gap():
    # Documents a confirmed inconsistency: tokenize()'s regex includes a
    # lone-apostrophe alternative (`|['’]`), but _build_ne_map's own
    # inline regex (r"\w+['’]?\w*|\w+") does not. A standalone
    # apostrophe token from tokenize() can never be matched via this path.
    # Narrow practical impact; not a bug to fix in this step.
    obj = _make_annotator()
    assert tokenize("'") == ["'"]
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("'")]), ["'", "x"]) == {}


# --- _ensure_ner ---------------------------------------------------------

def test_ensure_ner_disabled_leaves_self_ner_untouched():
    obj = _make_annotator()
    obj.ner = None
    obj._ensure_ner(enabled=False)
    assert obj.ner is None


def test_ensure_ner_disabled_does_not_overwrite_existing_ner():
    obj = _make_annotator()
    sentinel = object()
    obj.ner = sentinel
    obj._ensure_ner(enabled=False)
    assert obj.ner is sentinel


def test_ensure_ner_enabled_does_not_overwrite_existing_ner():
    # Lazy construction: only builds a pipeline the first time, when
    # self.ner is still None.
    obj = _make_annotator()
    sentinel = object()
    obj.ner = sentinel
    obj._ensure_ner(enabled=True)
    assert obj.ner is sentinel


def test_ensure_ner_enabled_lazily_constructs_pipeline_when_none():
    obj = _make_annotator()
    obj.ner = None
    with mock.patch.object(cs_pipeline, "stanza") as mocked_stanza:
        mocked_stanza.Pipeline.return_value = "fake_pipeline_instance"
        obj._ensure_ner(enabled=True)
    assert obj.ner == "fake_pipeline_instance"
    mocked_stanza.Pipeline.assert_called_once_with(
        "tr", processors="tokenize,ner", use_gpu=False
    )
