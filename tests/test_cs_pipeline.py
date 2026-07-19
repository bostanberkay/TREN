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
    # 1 EN + 5 MIXED, at default weights (0.6/0.4), produces an EXACT
    # floating-point tie: score_tr = 0.6*5 = 3.0, score_en = 1 + 0.4*5 = 3.0
    # (confirmed no float-precision surprise). Ties resolve to TR, same as
    # the plain exact_tr_en_tie case above but reached via MIXED weighting
    # rather than raw TR/EN counts.
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
# First-time integration-level tests: exercise the real annotate() method
# end-to-end. Lower-level helper internals (choose_label priority order,
# apostrophe/non-apostrophe MIXED detection, suffix parsing, NE-map
# matching, matrix/embed voting) are already covered by dedicated unit
# tests above and are NOT re-verified here. Detailed MIXED/UID/suffix
# branch coverage through annotate() is deliberately out of scope for this
# commit -- these tests only prove the control-flow skeleton: ordering,
# branching, and output construction.

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
    # Patches _build_ne_map directly (already unit-tested on its own) so
    # this test proves annotate()'s branch ORDERING -- is_other_token is
    # checked before ne_map membership -- rather than re-testing NE-match
    # logic. Deliberately does NOT mock _ft_predict: if OTHER precedence
    # ever broke and this token reached the language-choice path instead,
    # the test would fail loudly (AttributeError) rather than silently
    # passing via a mock.
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
