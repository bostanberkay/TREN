from unittest import mock

import pytest

from tren import cs_pipeline
from tren.cs_pipeline import Annotator, DEFAULTS, is_other_token, tokenize


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
    # Locks in a dead branch: re.split on every apostrophe means the suffix can
    # never contain "n't"; real contractions hit EN_CONTRACTIONS instead.
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
    # Reachable despite appearances: stage 2 can strip a POSS_SHORT suffix and
    # expose a trailing vowel that stage 1 already ran past.
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
    # Both "nın" and "ın" splits have English bases; longest-first must pick "nın".
    obj = _make_annotator(english_words={"lila", "lilan"})
    assert obj._detect_mixed_no_apostrophe("lilanın", DEFAULTS) == ("lila", "nın")


def test_detect_mixed_no_apostrophe_first_valid_candidate_wins_after_longest_fails():
    # Only the shorter split's base is English: the loop must continue past the
    # failed longest candidate.
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
    # Real suffixes always yield a feature, so force a no-feature parse to
    # exercise the guard itself.
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
# Reads only doc.ents and each entity's .text; no Annotator state needed.

class _FakeEnt:
    def __init__(self, text, type="PERSON"):
        # PERSON survives Policy C and blocks Policy D, so non-C/D tests keep
        # "always NE"; C/D tests pass an explicit type.
        self.text = text
        self.type = type


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
    # Known limitation, locked in: matching is by token text, not position, so
    # the same string elsewhere in the line is also treated as NE.
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
    # Known inconsistency, locked in: tokenize() emits lone apostrophes but this
    # regex cannot match them.
    obj = _make_annotator()
    assert tokenize("'") == ["'"]
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("'")]), ["'", "x"]) == {}


# --- NE arbitration: Policy C (TIME subtype) / Policy D (bare English
# lexical exception) -- offline-validated against the real corpus and
# synthetic benchmark before this production integration; see CHANGELOG.

def test_build_ne_map_policy_c_corrects_time_false_positives():
    # Real-corpus false positives: gold TR, Stanza tags a date/weekday
    # phrase as one TIME entity.
    obj = _make_annotator(turkish_top={"aralık", "mayıs", "pazartesi"})
    tokens = ["25", "Aralık"]
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("25 Aralık", type="TIME")]), tokens) == {}
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("2025 mayıs", type="TIME")]), ["2025", "mayıs"]) == {}
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("pazartesi", type="TIME")]), ["pazartesi"]) == {}


def test_build_ne_map_policy_c_does_not_touch_money():
    # Real-corpus genuine gold=NE MONEY matches ("dolar"/"dolarlık") --
    # MONEY was evaluated and deliberately excluded from Policy C's scope;
    # must stay NE.
    obj = _make_annotator()
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("milyar dolarlık", type="MONEY")]),
                              ["milyar", "dolarlık"]) == {"milyar": "NE", "dolarlık": "NE"}
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("15 dolar", type="MONEY")]),
                              ["15", "dolar"]) == {"15": "NE", "dolar": "NE"}


def test_build_ne_map_policy_c_keeps_ne_when_also_matched_by_allowed_subtype():
    # A piece matched by BOTH a TIME entity and a non-excluded-subtype
    # entity keeps NE -- conservative, matches only when EVERY match is
    # an excluded subtype.
    obj = _make_annotator()
    entities = [_FakeEnt("25 Aralık", type="TIME"), _FakeEnt("Aralık", type="ORGANIZATION")]
    assert obj._build_ne_map(_FakeDoc(entities), ["25", "Aralık"]) == {"Aralık": "NE"}


@pytest.mark.parametrize("token", ["Achievement", "Research", "Early", "Queer", "detachment"])
def test_build_ne_map_policy_d_corrects_ordinary_english_words(token):
    obj = _make_annotator(english_words={token.lower()})
    assert obj._build_ne_map(_FakeDoc([_FakeEnt(token, type="ORGANIZATION")]), [token]) == {}


def test_build_ne_map_policy_d_retains_genuine_multiword_compounds():
    # Regression (found and fixed during offline validation): pieces of a
    # genuine compound organization name must NOT be overridden even if
    # individually they'd otherwise qualify.
    obj = _make_annotator(english_words={"comic", "plus", "after", "effect"})
    tokens = ["Comic", "Plus"]
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Comic Plus", type="ORGANIZATION")]), tokens) == {
        "Comic": "NE", "Plus": "NE",
    }
    tokens2 = ["After", "Effect"]
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("After Effect", type="ORGANIZATION")]), tokens2) == {
        "After": "NE", "Effect": "NE",
    }


def test_build_ne_map_policy_d_still_fixes_ner_boundary_noise():
    # Span-boundary noise in the other piece must not block the override on the
    # piece that qualifies.
    obj = _make_annotator(english_words={"achievement", "research", "detachment"})
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Achievement examın", type="ORGANIZATION")]),
                              ["Achievement", "examın"]) == {"examın": "NE"}
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Research Quality'de", type="ORGANIZATION")]),
                              ["Research", "Quality'de"]) == {"Quality'de": "NE"}
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("3 detachment", type="MONEY")]),
                              ["3", "detachment"]) == {"3": "NE"}


def test_build_ne_map_policy_d_never_touches_apostrophed_tokens():
    obj = _make_annotator(english_words={"store"})
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Store'da", type="ORGANIZATION")]), ["Store'da"]) == {
        "Store'da": "NE",
    }


def test_build_ne_map_policy_d_never_touches_acronyms_or_redacted():
    obj = _make_annotator(english_words={"ai", "redacted"})
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("AI", type="ORGANIZATION")]), ["AI"]) == {"AI": "NE"}
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("REDACTED", type="ORGANIZATION")]), ["REDACTED"]) == {
        "REDACTED": "NE",
    }


def test_build_ne_map_policy_d_declines_when_valid_turkish_morphology_exists():
    # A bare English-lexicon word that ALSO has a plausible Turkish
    # nominal-suffix reading must not be overridden (condition 7).
    obj = _make_annotator(english_words={"kolaj"})
    # "kolajlar" = "kolaj" + "lar" (Number=Plur) -- a valid Turkish parse.
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("kolajlar", type="ORGANIZATION")]), ["kolajlar"]) == {
        "kolajlar": "NE",
    }


@pytest.mark.parametrize("entity_type", ["PERSON", "LOCATION", "ORGANIZATION"])
def test_build_ne_map_retains_genuine_entities_of_all_three_types(entity_type):
    obj = _make_annotator()
    assert obj._build_ne_map(_FakeDoc([_FakeEnt("Bocconi", type=entity_type)]), ["Bocconi"]) == {
        "Bocconi": "NE",
    }


def test_annotate_ai_siz_remains_mixed_despite_ai_entity_in_sentence():
    # Regression: an entity "AI" must not pull "AI'sız" into NE; it stays on the
    # apostrophe-MIXED path.
    obj = _make_annotator(english_words={"ai"})
    obj.ner = lambda line: _FakeDoc([_FakeEnt("AI", type="ORGANIZATION")])
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("AI'sız", DEFAULTS)
    assert "AI'sız\tMIXED" in out.splitlines()


def test_annotate_apostrophe_mixed_forms_unaffected_by_policy_c_d():
    # Pre-existing apostrophe-MIXED behavior (no entity involved at all)
    # must be byte-identical after the Policy C/D change.
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("meeting'e", _CFG_NO_NER)
    assert out == "SentenceID\t1\nmeeting'e\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n"


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
        "tr", processors="tokenize,ner", use_gpu=False,
        download_method=mocked_stanza.DownloadMethod.REUSE_RESOURCES,
    )


def test_ensure_ner_reuses_cached_resources_instead_of_checking_online():
    # Regression: the default DOWNLOAD_RESOURCES re-fetched resources.json on
    # every start, so annotation with NER on failed offline even with every
    # model already cached.
    obj = _make_annotator()
    obj.ner = None
    with mock.patch.object(cs_pipeline, "stanza") as mocked_stanza:
        obj._ensure_ner(enabled=True)
    kwargs = mocked_stanza.Pipeline.call_args.kwargs
    assert kwargs["download_method"] is mocked_stanza.DownloadMethod.REUSE_RESOURCES


def test_ensure_ner_failure_raises_clear_error_and_keeps_ner_unset():
    obj = _make_annotator()
    obj.ner = None
    with mock.patch.object(cs_pipeline, "stanza") as mocked_stanza:
        mocked_stanza.Pipeline.side_effect = ConnectionError("no network")
        with pytest.raises(cs_pipeline.NERUnavailableError) as excinfo:
            obj._ensure_ner(enabled=True)
    msg = str(excinfo.value)
    assert "internet connection" in msg
    assert "turn off the NER option" in msg
    assert "no network" in msg
    assert obj.ner is None


def test_annotate_with_ner_enabled_does_not_silently_skip_ner_when_unavailable():
    obj = _make_annotator()
    obj.ner = None
    with mock.patch.object(cs_pipeline, "stanza") as mocked_stanza:
        mocked_stanza.Pipeline.side_effect = OSError("models missing")
        with pytest.raises(cs_pipeline.NERUnavailableError):
            obj.annotate("Ahmet geldi", {"NER_ENABLED": True})


# --- _decide_matrix_embed ----------------------------------------------
# Touches no `self.*` state at all -- Annotator.__new__() with zero
# attributes set is sufficient. No models, files, Stanza objects, or
# lexicons are involved anywhere in this function.

@pytest.mark.parametrize("labels, expected", [
    ([], ("TR", "-")),                                  # empty label list -- 0>=0 tie-break to TR, no EN/MIXED -> "-"
    (["TR", "TR", "TR"], ("TR", "-")),                   # TR-only
    (["EN", "EN"], ("EN", "-")),                         # EN-only
    (["TR", "TR", "EN"], ("TR", "EN")),                  # TR majority
    (["EN", "EN", "TR"], ("EN", "TR")),                  # EN majority
    (["TR", "EN"], ("TR", "EN")),                        # exact TR/EN tie -> current TR tie-break
    (["MIXED"], ("TR", "EN")),                           # one MIXED, default weights (0.6 TR / 0.4 EN)
    (["TR", "MIXED"], ("TR", "EN")),                     # MIXED combined with TR
    (["EN", "MIXED"], ("EN", "TR")),                     # MIXED combined with EN
    (["TR", "NE", "OTHER", "UID"], ("TR", "-")),         # NE/OTHER/UID ignored -- same result as TR-only
    (["NE", "OTHER", "UID"], ("TR", "-")),               # only NE/OTHER/UID -- same result as empty list
], ids=[
    "empty_label_list", "tr_only", "en_only", "tr_majority", "en_majority",
    "exact_tr_en_tie", "one_mixed_default_weights", "mixed_combined_with_tr",
    "mixed_combined_with_en", "ne_other_uid_ignored_alongside_tr",
    "labels_containing_only_ne_other_uid",
])
def test_decide_matrix_embed(labels, expected):
    obj = _make_annotator()
    assert obj._decide_matrix_embed(labels, DEFAULTS) == expected


def test_decide_matrix_embed_weighted_tie_via_mixed():
    # Exact float tie via MIXED weighting (3.0 vs 3.0) resolves to TR.
    obj = _make_annotator()
    labels = ["EN"] + ["MIXED"] * 5
    assert obj._decide_matrix_embed(labels, DEFAULTS) == ("TR", "EN")


def test_decide_matrix_embed_custom_mixed_tr_weight_can_flip_outcome():
    # At default weights, ["EN", "MIXED"] resolves EN (score_en=1.4 >
    # score_tr=0.6). Boosting MIXED_TR_WEIGHT flips the outcome to TR,
    # proving the cfg value is actually read, not hardcoded.
    obj = _make_annotator()
    labels = ["EN", "MIXED"]
    assert obj._decide_matrix_embed(labels, DEFAULTS) == ("EN", "TR")
    cfg = dict(DEFAULTS, MIXED_TR_WEIGHT=2.0)
    assert obj._decide_matrix_embed(labels, cfg) == ("TR", "EN")


def test_decide_matrix_embed_custom_mixed_en_weight_can_flip_outcome():
    # At default weights, ["TR", "MIXED"] resolves TR (score_tr=1.6 >
    # score_en=0.4). Boosting MIXED_EN_WEIGHT flips the outcome to EN,
    # proving the cfg value is actually read, not hardcoded.
    obj = _make_annotator()
    labels = ["TR", "MIXED"]
    assert obj._decide_matrix_embed(labels, DEFAULTS) == ("TR", "EN")
    cfg = dict(DEFAULTS, MIXED_EN_WEIGHT=2.0)
    assert obj._decide_matrix_embed(labels, cfg) == ("EN", "TR")


def test_decide_matrix_embed_returns_a_two_element_string_tuple():
    obj = _make_annotator()
    result = obj._decide_matrix_embed(["TR"], DEFAULTS)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(x, str) for x in result)


@pytest.mark.parametrize("labels, expected_embed", [
    (["TR", "TR"], "-"),   # matrix resolves TR, no EN/MIXED present -> "-"
    (["EN", "EN"], "-"),   # matrix resolves EN, no TR/MIXED present -> "-"
], ids=["no_embed_when_matrix_tr_and_no_en_or_mixed", "no_embed_when_matrix_en_and_no_tr_or_mixed"])
def test_decide_matrix_embed_dash_sentinel_both_directions(labels, expected_embed):
    obj = _make_annotator()
    matrix, embed = obj._decide_matrix_embed(labels, DEFAULTS)
    assert embed == expected_embed


# --- annotate() control-flow skeleton ---------------------------------
# Proves ordering, branching and output construction only; helper internals
# have their own unit tests above.

def test_annotate_empty_input():
    obj = _make_annotator()
    obj.ner = lambda line: _FakeDoc([])
    assert obj.annotate("", DEFAULTS) == ""


def test_annotate_whitespace_only_single_line():
    obj = _make_annotator()
    obj.ner = lambda line: _FakeDoc([])
    assert obj.annotate("   ", DEFAULTS) == ""


def test_annotate_whitespace_only_multiple_lines():
    obj = _make_annotator()
    obj.ner = lambda line: _FakeDoc([])
    assert obj.annotate("  \n\t\n", DEFAULTS) == "\n"


def test_annotate_blank_line_preserved_as_output_separator():
    obj = _make_annotator(turkish_top={"bugun", "gunaydin"})
    obj.ner = lambda line: _FakeDoc([])
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun\n\ngunaydin", DEFAULTS)
    assert out == (
        "SentenceID\t1\nbugun\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
        "\n"
        "\nSentenceID\t2\ngunaydin\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
    )


def test_annotate_blank_line_does_not_increment_sentence_id():
    obj = _make_annotator(turkish_top={"bugun", "gunaydin"})
    obj.ner = lambda line: _FakeDoc([])
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun\n\ngunaydin", DEFAULTS)
    assert "SentenceID\t1" in out
    assert "SentenceID\t2" in out
    assert "SentenceID\t3" not in out


@pytest.mark.parametrize("flag, expect_row", [(True, True), (False, False)], ids=["enabled", "disabled"])
def test_annotate_feature_sentence_id_flag(flag, expect_row):
    obj = _make_annotator(turkish_top={"bugun"})
    obj.ner = lambda line: _FakeDoc([])
    cfg = dict(DEFAULTS, FEATURE_SENTENCE_ID=flag)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun", cfg)
    assert ("SentenceID\t1" in out.splitlines()) == expect_row


def test_annotate_multiline_sentence_id_counting():
    obj = _make_annotator(turkish_top={"bugun", "gunaydin", "iyi"})
    obj.ner = lambda line: _FakeDoc([])
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun\ngunaydin\niyi", DEFAULTS)
    assert "SentenceID\t1" in out.splitlines()
    assert "SentenceID\t2" in out.splitlines()
    assert "SentenceID\t3" in out.splitlines()


def test_annotate_ner_disabled_never_calls_self_ner():
    obj = _make_annotator(turkish_top={"bugun"})
    ner_mock = mock.Mock()
    obj.ner = ner_mock
    cfg = dict(DEFAULTS, NER_ENABLED=False)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        obj.annotate("bugun", cfg)
    ner_mock.assert_not_called()


def test_annotate_ner_enabled_called_once_per_nonblank_line():
    obj = _make_annotator(turkish_top={"bugun", "gunaydin"})
    ner_mock = mock.Mock(return_value=_FakeDoc([]))
    obj.ner = ner_mock
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        obj.annotate("bugun\n\ngunaydin", DEFAULTS)  # NER_ENABLED defaults True
    assert ner_mock.call_count == 2
    ner_mock.assert_has_calls([mock.call("bugun"), mock.call("gunaydin")])


def test_annotate_other_precedence_over_ne():
    # Proves is_other_token is checked before ne_map. _ft_predict is deliberately
    # not mocked, so a precedence regression fails loudly.
    obj = _make_annotator()
    obj.ner = lambda line: _FakeDoc([])
    with mock.patch.object(obj, "_build_ne_map", return_value={"42": "NE"}):
        out = obj.annotate("42", DEFAULTS)
    lines = out.splitlines()
    assert "42\tOTHER" in lines
    assert "42\tNE" not in lines


def test_annotate_ne_precedence_over_choose_label():
    # Patches _build_ne_map to force NE membership, and spies on
    # _choose_label to prove it is never reached for an NE-matched token --
    # this tests branch ordering, not _choose_label's own (already-tested)
    # internal logic.
    obj = _make_annotator()
    obj.ner = lambda line: _FakeDoc([])
    with mock.patch.object(obj, "_build_ne_map", return_value={"Istanbul": "NE"}):
        with mock.patch.object(obj, "_choose_label") as mocked_choose:
            out = obj.annotate("Istanbul", DEFAULTS)
    assert "Istanbul\tNE" in out.splitlines()
    mocked_choose.assert_not_called()


@pytest.mark.parametrize("flag, expect_token_row", [(True, True), (False, False)], ids=[
    "per_item_true_emits_rows", "per_item_false_emits_no_rows",
])
def test_annotate_feature_language_per_item_row_emission(flag, expect_token_row):
    obj = _make_annotator(turkish_top={"bugun"})
    obj.ner = lambda line: _FakeDoc([])
    cfg = dict(DEFAULTS, FEATURE_LANGUAGE_PER_ITEM=flag)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun", cfg)
    lines = out.splitlines()
    assert ("bugun\tTR" in lines) == expect_token_row
    # meta rows (SentenceID/MatrixLang) are unaffected either way
    assert "SentenceID\t1" in lines
    assert "MatrixLang\tTR" in lines


@pytest.mark.parametrize("matrix_flag, embed_flag, expect_matrix_row, expect_embed_row", [
    (True, True, True, True),
    (True, False, True, False),
    (False, True, False, True),
    (False, False, False, False),
], ids=[
    "matrix_true_embed_true", "matrix_true_embed_false",
    "matrix_false_embed_true", "matrix_false_embed_false",
])
def test_annotate_matrix_embed_flag_combinations(
    matrix_flag, embed_flag, expect_matrix_row, expect_embed_row
):
    obj = _make_annotator(turkish_top={"bugun"})
    obj.ner = lambda line: _FakeDoc([])
    cfg = dict(DEFAULTS, FEATURE_MATRIX_LANGUAGE=matrix_flag, FEATURE_EMBEDDED_LANGUAGE=embed_flag)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun", cfg)
    lines = out.splitlines()
    assert any(l.startswith("MatrixLang\t") for l in lines) == expect_matrix_row
    assert any(l.startswith("EmbedLang\t") for l in lines) == expect_embed_row


# --- annotate() detailed integration branches --------------------------
# NER_ENABLED=False throughout; already-tested helpers are patched out.

_CFG_NO_NER = dict(DEFAULTS, NER_ENABLED=False)


# -- Apostrophe MIXED --

def test_annotate_apostrophe_mixed_english_base_produces_mixed_row():
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("meeting'e", _CFG_NO_NER)
    assert out == "SentenceID\t1\nmeeting'e\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n"


def test_annotate_apostrophe_mixed_curly_apostrophe_variant():
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("meeting’e", _CFG_NO_NER)
    assert out == "SentenceID\t1\nmeeting’e\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n"


def test_annotate_apostrophe_mixed_requires_parsed_features():
    # The gate is always true in practice; force a no-feature parse to check that
    # MIXED is skipped and labeling falls through to UID.
    obj = _make_annotator(english_words={"meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        with mock.patch.object(obj, "_parse_tr_suffixes_full", return_value=([], set(), set(), set())):
            with mock.patch.object(obj, "_detect_mixed_no_apostrophe", return_value=(None, None)):
                out = obj.annotate("meeting'e", _CFG_NO_NER)
    assert out == "SentenceID\t1\nmeeting'e\tUID\nMatrixLang\tTR\nEmbedLang\t-\n"


def test_annotate_apostrophe_split_turkish_base_follows_tr_branch():
    # base_label == "TR" is a distinct branch from both the MIXED path and
    # the ordinary-labeling fallback -- it emits the ORIGINAL token text
    # (with apostrophe) labeled TR, not just the base.
    obj = _make_annotator(turkish_top={"kedi"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("kedi'ye", _CFG_NO_NER)
    assert out == "SentenceID\t1\nkedi'ye\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"


def test_annotate_apostrophe_split_non_qualifying_falls_through():
    # base_label UID: the apostrophe branch must fall through to ordinary labeling
    # (helpers patched to isolate the wiring).
    obj = _make_annotator()
    with mock.patch.object(obj, "_split_mixed_apostrophe", return_value=("weird", "zzz")):
        with mock.patch.object(obj, "_choose_label", return_value="UID") as mocked_choose:
            with mock.patch.object(obj, "_detect_mixed_no_apostrophe", return_value=(None, None)) as mocked_detect:
                out = obj.annotate("weird'zzz", _CFG_NO_NER)
    assert out == "SentenceID\t1\nweird'zzz\tUID\nMatrixLang\tTR\nEmbedLang\t-\n"
    assert mocked_choose.call_count == 2  # whole token, then base
    mocked_detect.assert_called_once_with("weird'zzz", _CFG_NO_NER)


# -- Non-apostrophe MIXED --

def test_annotate_non_apostrophe_mixed_produces_mixed_row():
    obj = _make_annotator(english_words={"stress"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("stressim", _CFG_NO_NER)
    assert out == "SentenceID\t1\nstressim\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n"


def test_annotate_non_apostrophe_mixed_wiring_via_mocked_detection():
    # Patches _detect_mixed_no_apostrophe directly (already unit-tested on
    # its own) to prove annotate() trusts its returned (base, suffix) and
    # emits the MIXED row accordingly -- pure wiring, not re-derivation of
    # that function's own feature-parsing logic.
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        with mock.patch.object(obj, "_detect_mixed_no_apostrophe", return_value=("stress", "im")) as mocked:
            out = obj.annotate("stressim", _CFG_NO_NER)
    assert out == "SentenceID\t1\nstressim\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n"
    mocked.assert_called_once_with("stressim", _CFG_NO_NER)


def test_annotate_non_apostrophe_mixed_skipped_when_label_is_tr():
    obj = _make_annotator(turkish_top={"bugun"})
    with mock.patch.object(obj, "_detect_mixed_no_apostrophe") as mocked:
        obj.annotate("bugun", _CFG_NO_NER)
    mocked.assert_not_called()


def test_annotate_non_apostrophe_mixed_consulted_for_non_tr_labels():
    obj = _make_annotator(english_words={"stressed"})
    with mock.patch.object(obj, "_detect_mixed_no_apostrophe", return_value=(None, None)) as mocked:
        obj.annotate("stressed", _CFG_NO_NER)
    mocked.assert_called_once_with("stressed", _CFG_NO_NER)


# -- UID and voting --

def test_annotate_uid_emits_token_row_when_per_item_true():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("xyzzy", _CFG_NO_NER)
    assert out == "SentenceID\t1\nxyzzy\tUID\nMatrixLang\tTR\nEmbedLang\t-\n"


def test_annotate_uid_does_not_contribute_to_voting():
    # A TR token plus a UID token: MatrixLang must reflect only the TR
    # contribution, exactly as if the UID token were TR-only (i.e. UID
    # contributes nothing to either score).
    obj = _make_annotator(turkish_top={"bugun"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun xyzzy", _CFG_NO_NER)
    assert out == "SentenceID\t1\nbugun\tTR\nxyzzy\tUID\nMatrixLang\tTR\nEmbedLang\t-\n"


def test_annotate_all_uid_sentence_yields_dash_sentinels():
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("xyzzy plugh", _CFG_NO_NER)
    assert out == "SentenceID\t1\nxyzzy\tUID\nplugh\tUID\nMatrixLang\tTR\nEmbedLang\t-\n"


def test_annotate_uid_rows_disappear_when_per_item_false():
    obj = _make_annotator()
    cfg = dict(_CFG_NO_NER, FEATURE_LANGUAGE_PER_ITEM=False)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("xyzzy", cfg)
    assert out == "SentenceID\t1\nMatrixLang\tTR\nEmbedLang\t-\n"
    assert "UID" not in out


# -- Final output construction --

def test_annotate_token_rows_precede_matrix_and_embed_rows():
    obj = _make_annotator(turkish_top={"bugun"}, english_words={"stressed"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun stressed", _CFG_NO_NER)
    lines = out.splitlines()
    assert lines == [
        "SentenceID\t1", "bugun\tTR", "stressed\tEN", "MatrixLang\tTR", "EmbedLang\tEN",
    ]
    matrix_idx = lines.index("MatrixLang\tTR")
    embed_idx = lines.index("EmbedLang\tEN")
    token_indices = [lines.index("bugun\tTR"), lines.index("stressed\tEN")]
    assert all(i < matrix_idx for i in token_indices)
    assert matrix_idx < embed_idx


def test_annotate_each_sentence_block_ends_with_blank_separator():
    obj = _make_annotator(turkish_top={"bugun", "gunaydin"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out_single = obj.annotate("bugun", _CFG_NO_NER)
        out_multi = obj.annotate("bugun\ngunaydin", _CFG_NO_NER)
    assert out_single == "SentenceID\t1\nbugun\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
    assert out_multi == (
        "SentenceID\t1\nbugun\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
        "\nSentenceID\t2\ngunaydin\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
    )


def test_annotate_representative_tr_en_mixed_sentence_exact_output():
    obj = _make_annotator(turkish_top={"bugun"}, english_words={"stressed", "meeting"})
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("bugun stressed meeting'e", _CFG_NO_NER)
    assert out == (
        "SentenceID\t1\n"
        "bugun\tTR\n"
        "stressed\tEN\n"
        "meeting'e\tMIXED\n"
        "MatrixLang\tTR\n"
        "EmbedLang\tEN\n"
    )


# --- tokenize() ----------------------------------------------------------
# Current regex behavior is locked in as-is, including non-obvious findings.

@pytest.mark.parametrize("text, expected", [
    ("hello world", ["hello", "world"]),
    ("123 456", ["123", "456"]),
    ("under_score", ["under_score"]),                       # underscore is a \w char
    ("word_123_test", ["word_123_test"]),
    ("çalışma İstanbul ğüşöç", ["çalışma", "İstanbul", "ğüşöç"]),  # Turkish letters are \w
    ("Bugün meeting'e gitmem lazım", ["Bugün", "meeting'e", "gitmem", "lazım"]),
    ("", []),
    ("   ", []),
    ("\n\t", []),
    ("a\tb\nc", ["a", "b", "c"]),
    ("a-b", ["a-b"]),           # hyphen-joined alnum segments now kept as one token
                                 # (so machine codes like "A7K-204" survive intact)
    ("---", []),                # pure punctuation run -- matches nothing at all
], ids=[
    "ordinary_whitespace_separated", "digits", "underscore", "underscore_with_digits",
    "turkish_characters", "mixed_turkish_english", "empty_string", "whitespace_only",
    "newlines_and_tabs", "tab_and_newline_separated_tokens", "hyphen_between_words",
    "pure_punctuation_run",
])
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def test_tokenize_punctuation_is_dropped_not_returned_as_tokens():
    # Punctuation matches none of the regex's three alternatives, so it is
    # silently absent from the result -- NOT returned as its own token(s).
    # This is current, verified behavior, not "punctuation tokens" in the
    # sense of separate punctuation entries appearing in the output.
    assert tokenize("hello, world!") == ["hello", "world"]
    assert tokenize("hello , world !") == ["hello", "world"]


# --- regression: @mentions, #hashtags, URLs, emoji and joined codes must
# survive tokenize() whole (\w never matches '@', '#' or emoji).

def test_tokenize_preserves_mention():
    assert tokenize("hey @berkay check this") == ["hey", "@berkay", "check", "this"]


def test_tokenize_preserves_hashtag():
    assert tokenize("great day #tbt indeed") == ["great", "day", "#tbt", "indeed"]


def test_tokenize_preserves_url():
    assert tokenize("see https://example.com/path?x=1 now") == [
        "see", "https://example.com/path?x=1", "now",
    ]


def test_tokenize_preserves_www_url():
    assert tokenize("visit www.example.com today") == ["visit", "www.example.com", "today"]


def test_tokenize_preserves_emoji_as_one_token():
    assert tokenize("nice 😀 work") == ["nice", "😀", "work"]


def test_tokenize_preserves_hyphenated_code():
    assert tokenize("kod A7K-204 ve TR-9081-ZX burada") == [
        "kod", "A7K-204", "ve", "TR-9081-ZX", "burada",
    ]


def test_tokenize_preserves_numeric_with_separators():
    # Mirrors NUMERIC_RE's own separator support (.,:/- between digit
    # groups); previously unreachable because tokenize() fragmented on
    # those separators before is_other_token() ever ran.
    assert tokenize("saat 10:30 fiyat 3.14") == ["saat", "10:30", "fiyat", "3.14"]


def test_tokenize_digit_prefixed_word_not_split_by_numeric_alternative():
    # Regression (found in a real-corpus diff): the separator-number alternative
    # must not split "20li"/"3d" into two tokens.
    assert tokenize("kod 20li ve 3d ve 6da") == ["kod", "20li", "ve", "3d", "ve", "6da"]


# --- is_other_token(): the new CODE_RE branch --------------------------

def test_is_other_token_hyphenated_alnum_code():
    assert is_other_token("A7K-204") is True
    assert is_other_token("TR-9081-ZX") is True


def test_is_other_token_underscore_alnum_code():
    assert is_other_token("QR_77B") is True


def test_is_other_token_code_requires_both_letter_and_digit():
    # A hyphen-joined token with only letters is not a "code" in this
    # sense -- plain hyphenated words stay off the OTHER path.
    assert is_other_token("well-known") is False


def test_is_other_token_pure_digit_hyphenated_uses_numeric_re_not_code_re():
    # "12-34" is still classified OTHER, but via the pre-existing
    # NUMERIC_RE branch (digits + separators), not the new CODE_RE branch
    # (which requires at least one letter).
    assert is_other_token("12-34") is True


def test_is_other_token_mention_and_hashtag_reachable_after_tokenize():
    # Historically dead: MENTION_RE/HASHTAG_RE checked tokens that had
    # already lost their '@'/'#' inside tokenize(). Confirms the full
    # pipeline (tokenize -> is_other_token) now classifies them OTHER.
    tokens = tokenize("hey @berkay #tbt")
    assert [is_other_token(t) for t in tokens] == [False, True, True]


@pytest.mark.parametrize("text, expected", [
    ("meeting's", ["meeting's"]),
    ("meeting’s", ["meeting’s"]),
], ids=["straight_apostrophe_inside_word", "curly_apostrophe_inside_word"])
def test_tokenize_apostrophe_inside_word(text, expected):
    assert tokenize(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("'", ["'"]),
    ("’", ["’"]),
], ids=["standalone_straight_apostrophe", "standalone_curly_apostrophe"])
def test_tokenize_standalone_apostrophe(text, expected):
    assert tokenize(text) == expected


def test_tokenize_leading_vs_trailing_apostrophe_are_asymmetric():
    # Asymmetric, locked in: a trailing apostrophe joins the word, a leading one
    # becomes its own token.
    assert tokenize("hello'") == ["hello'"]
    assert tokenize("'hello") == ["'", "hello"]


def test_tokenize_multiple_apostrophes_split_three_ways():
    # The first apostrophe in a word is absorbed into that token; a second,
    # unabsorbable apostrophe becomes its own standalone token; scanning
    # then resumes plainly from there.
    assert tokenize("a'b'c") == ["a'b", "'", "c"]
    assert tokenize("it's a'b'c") == ["it's", "a'b", "'", "c"]


def test_tokenize_preserves_order():
    text = "Bugün meeting'e gitmem lazım çünkü boss'um çok stressed"
    assert tokenize(text) == [
        "Bugün", "meeting'e", "gitmem", "lazım", "çünkü", "boss'um", "çok", "stressed",
    ]


def test_tokenize_returns_a_list_of_strings():
    result = tokenize("hello world")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


# --- end-to-end smoke test -----------------------------------------------
# Only _ft_predict is patched; every other pipeline step runs for real.

def test_annotate_end_to_end_smoke_tr_en_mixed_sentence():
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = {"kitap"}
    obj.turkish_freq_all = set()
    obj.english_freq_words = {"amazing", "boss"}

    cfg = dict(DEFAULTS, NER_ENABLED=False)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("kitap amazing boss'um", cfg)

    assert out == (
        "SentenceID\t1\n"      # sentence counter
        "kitap\tTR\n"          # real _choose_label: Turkish top-1000 hit
        "amazing\tEN\n"        # real _choose_label: English lexicon hit
        "boss'um\tMIXED\n"     # real apostrophe split + suffix parsing: boss (EN) + um (Poss=Yes)
        "MatrixLang\tTR\n"     # real _decide_matrix_embed voting
        "EmbedLang\tEN\n"      # real _decide_matrix_embed voting
    )


def test_annotate_end_to_end_smoke_multiline_independent_sentences():
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = {"kitap"}
    obj.turkish_freq_all = set()
    obj.english_freq_words = {"amazing", "boss"}

    cfg = dict(DEFAULTS, NER_ENABLED=False)
    with mock.patch.object(obj, "_ft_predict", return_value=("UID", 0.0)):
        out = obj.annotate("kitap amazing\nboss'um", cfg)

    assert out == (
        "SentenceID\t1\n"
        "kitap\tTR\n"
        "amazing\tEN\n"
        "MatrixLang\tTR\n"
        "EmbedLang\tEN\n"
        "\n"                   # blank separator between independent sentence blocks
        "SentenceID\t2\n"      # SentenceID increments across lines
        "boss'um\tMIXED\n"
        "MatrixLang\tTR\n"
        "EmbedLang\tEN\n"
    )
