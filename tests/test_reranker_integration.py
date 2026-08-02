# tests/test_reranker_integration.py
"""Phase 6.1/6.2: tests for reranker_integration.py -- loading infrastructure
(Phase 6.1) and apply_reranker() (Phase 6.2). No GUI, no production pipeline
involved. Uses the real resources/models/ artifacts (small, tracked files)
directly rather than mocking joblib/the model, since they're cheap to load
and give genuine end-to-end confidence.

Phase 6.2 tests reuse the exact same bypass-__init__ Annotator convention
already established in tests/test_cs_pipeline.py and
tests/test_mixed_reranker.py (Annotator.__new__ + mocked _ft_predict) rather
than inventing new test infrastructure."""

import json
import os
import shutil
import sys
from unittest import mock

import joblib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs_pipeline import Annotator, DEFAULTS
import reranker_integration as ri

REAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "models")


def _real_metadata():
    with open(os.path.join(REAL_MODEL_DIR, ri.METADATA_FILENAME), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# validate_metadata
# ---------------------------------------------------------------------------

def test_validate_metadata_accepts_real_frozen_metadata():
    is_valid, reasons = ri.validate_metadata(_real_metadata())
    assert is_valid is True
    assert reasons == []


def test_validate_metadata_rejects_non_dict():
    is_valid, reasons = ri.validate_metadata(["not", "a", "dict"])
    assert is_valid is False
    assert reasons


def test_validate_metadata_rejects_missing_batch_a():
    m = _real_metadata()
    sf = m["feature_configuration"]["structured_features"]
    m["feature_configuration"]["structured_features"] = [k for k in sf if k not in ri._BATCH_A_KEYS]
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("Batch A" in r for r in reasons)


def test_validate_metadata_rejects_missing_batch_c():
    m = _real_metadata()
    sf = m["feature_configuration"]["structured_features"]
    m["feature_configuration"]["structured_features"] = [k for k in sf if k not in ri._BATCH_C_KEYS]
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("Batch C" in r for r in reasons)


def test_validate_metadata_rejects_missing_batch_g():
    m = _real_metadata()
    sf = m["feature_configuration"]["structured_features"]
    m["feature_configuration"]["structured_features"] = [k for k in sf if k not in ri._BATCH_G_PRUNED_KEYS]
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("Batch G" in r for r in reasons)


def test_validate_metadata_rejects_unpruned_batch_g():
    m = _real_metadata()
    m["feature_configuration"]["structured_features"].append("distinct_stem_count")
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("distinct_stem_count" in r for r in reasons)


def test_validate_metadata_rejects_batch_b_present():
    m = _real_metadata()
    m["feature_configuration"]["structured_features"].append("morph_tag_count")
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("Batch B" in r for r in reasons)


def test_validate_metadata_rejects_batch_d_present():
    m = _real_metadata()
    m["feature_configuration"]["structured_features"].append("stem_english_confidence")
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("Batch D" in r for r in reasons)


def test_validate_metadata_rejects_wrong_threshold():
    m = _real_metadata()
    m["selected_thresholds"]["active"] = 0.81
    m["selected_thresholds"]["best_precision_ge_0.75"] = 0.81
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("threshold" in r for r in reasons)


def test_validate_metadata_rejects_wrong_policy():
    m = _real_metadata()
    m["selected_thresholds"]["policy"] = "max_f1"
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("policy" in r for r in reasons)


def test_validate_metadata_rejects_wrong_model_type():
    m = _real_metadata()
    m["feature_configuration"]["model"]["type"] = "RandomForestClassifier"
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert any("model type" in r for r in reasons)


def test_validate_metadata_reports_every_failure_not_just_first():
    m = _real_metadata()
    m["feature_configuration"]["structured_features"].append("morph_tag_count")
    m["selected_thresholds"]["active"] = 0.5
    is_valid, reasons = ri.validate_metadata(m)
    assert is_valid is False
    assert len(reasons) >= 2


# ---------------------------------------------------------------------------
# get_threshold
# ---------------------------------------------------------------------------

def test_get_threshold_returns_expected_value_for_real_metadata():
    assert ri.get_threshold(_real_metadata()) == pytest.approx(0.85)


def test_get_threshold_returns_none_when_key_missing():
    assert ri.get_threshold({"selected_thresholds": {}}) is None


def test_get_threshold_returns_none_when_not_a_dict():
    assert ri.get_threshold(None) is None
    assert ri.get_threshold("not a dict") is None


def test_get_threshold_returns_none_for_non_numeric_value():
    assert ri.get_threshold({"selected_thresholds": {"best_precision_ge_0.75": "high"}}) is None


# ---------------------------------------------------------------------------
# load_reranker_bundle -- success path (real resources/models/ artifacts)
# ---------------------------------------------------------------------------

def test_load_reranker_bundle_success_with_real_artifacts():
    bundle = ri.load_reranker_bundle(REAL_MODEL_DIR)
    assert bundle is not None
    assert isinstance(bundle, ri.ReRankerBundle)
    assert bundle.threshold == pytest.approx(0.85)
    assert bundle.model is not None
    assert bundle.tfidf is not None
    assert bundle.dictvec is not None
    assert isinstance(bundle.metadata, dict)


def test_load_reranker_bundle_default_model_dir_is_resources_models():
    assert ri.DEFAULT_MODEL_DIR == REAL_MODEL_DIR
    bundle = ri.load_reranker_bundle()  # uses the default
    assert bundle is not None


# ---------------------------------------------------------------------------
# load_reranker_bundle -- failure paths, all must return None (never raise)
# ---------------------------------------------------------------------------

def test_load_reranker_bundle_missing_directory(tmp_path):
    bundle = ri.load_reranker_bundle(str(tmp_path / "does_not_exist"))
    assert bundle is None


def test_load_reranker_bundle_missing_metadata_file(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.MODEL_FILENAME), tmp_path / ri.MODEL_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.VECTORIZER_FILENAME), tmp_path / ri.VECTORIZER_FILENAME)
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_missing_model_file(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.METADATA_FILENAME), tmp_path / ri.METADATA_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.VECTORIZER_FILENAME), tmp_path / ri.VECTORIZER_FILENAME)
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_invalid_metadata(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.MODEL_FILENAME), tmp_path / ri.MODEL_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.VECTORIZER_FILENAME), tmp_path / ri.VECTORIZER_FILENAME)
    bad_metadata = _real_metadata()
    bad_metadata["selected_thresholds"]["active"] = 0.5
    with open(tmp_path / ri.METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(bad_metadata, f)
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_malformed_metadata_json(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.MODEL_FILENAME), tmp_path / ri.MODEL_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.VECTORIZER_FILENAME), tmp_path / ri.VECTORIZER_FILENAME)
    (tmp_path / ri.METADATA_FILENAME).write_text("{not valid json", encoding="utf-8")
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_corrupted_model_file(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.METADATA_FILENAME), tmp_path / ri.METADATA_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.VECTORIZER_FILENAME), tmp_path / ri.VECTORIZER_FILENAME)
    (tmp_path / ri.MODEL_FILENAME).write_bytes(b"not a real joblib pickle at all")
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_corrupted_vectorizer_file(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.METADATA_FILENAME), tmp_path / ri.METADATA_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.MODEL_FILENAME), tmp_path / ri.MODEL_FILENAME)
    (tmp_path / ri.VECTORIZER_FILENAME).write_bytes(b"not a real joblib pickle at all")
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_vectorizer_wrong_type(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.METADATA_FILENAME), tmp_path / ri.METADATA_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.MODEL_FILENAME), tmp_path / ri.MODEL_FILENAME)
    joblib.dump(["not", "a", "dict"], tmp_path / ri.VECTORIZER_FILENAME)
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_vectorizer_missing_keys(tmp_path):
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.METADATA_FILENAME), tmp_path / ri.METADATA_FILENAME)
    shutil.copy(os.path.join(REAL_MODEL_DIR, ri.MODEL_FILENAME), tmp_path / ri.MODEL_FILENAME)
    joblib.dump({"tfidf": object()}, tmp_path / ri.VECTORIZER_FILENAME)  # missing 'dictvec'
    bundle = ri.load_reranker_bundle(str(tmp_path))
    assert bundle is None


def test_load_reranker_bundle_missing_dependency(tmp_path, monkeypatch):
    # Simulate joblib/scikit-learn not being installed, even though a
    # perfectly valid model_dir is supplied.
    monkeypatch.setitem(sys.modules, "joblib", None)
    bundle = ri.load_reranker_bundle(REAL_MODEL_DIR)
    assert bundle is None


# ---------------------------------------------------------------------------
# Lazy-loading contract
# ---------------------------------------------------------------------------

def test_load_reranker_bundle_never_raises_on_arbitrary_bad_input(tmp_path):
    # Belt-and-braces: a directory that doesn't even exist as a string path
    # component (e.g. contains null bytes) must still come back as None,
    # not propagate an exception up to a caller.
    try:
        bundle = ri.load_reranker_bundle(str(tmp_path) + "\x00bad")
    except Exception as e:  # pragma: no cover - this is exactly what must NOT happen
        pytest.fail(f"load_reranker_bundle raised instead of returning None: {e}")
    assert bundle is None


def test_module_does_not_import_joblib_at_top_level():
    # Structural guarantee: the only "import joblib" in reranker_integration.py
    # must be inside a function body, not at module scope, so importing this
    # module never requires joblib/scikit-learn to be installed.
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(ri))
    module_level_imports = set()
    for node in tree.body:  # only top-level statements, not nested in functions
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_level_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_level_imports.add(node.module.split(".")[0])
    assert "joblib" not in module_level_imports
    assert "sklearn" not in module_level_imports


def test_reranker_bundle_is_immutable_namedtuple():
    fields = ri.ReRankerBundle._fields
    assert fields == ("model", "tfidf", "dictvec", "threshold", "metadata")


# ---------------------------------------------------------------------------
# Phase 6.2: apply_reranker()
#
# Fixture tokens below were empirically verified once against the real,
# frozen resources/models/ bundle (not guessed): "boostlamak"/UID (stem
# "boost" in english_words, mocked EN fastText 0.95) -> P(MIXED)=0.985,
# clears the 0.85 threshold. "zzqxwv"/UID (no lexicon/fastText evidence) ->
# P(MIXED)=0.045, a candidate that stays rejected. "applar"/TR (stem "app"
# in english_words) -> P(MIXED)=0.819, a candidate rejected only because it
# falls just short of 0.85 -- distinct from "masada"/TR (already in
# turkish_all, no non-Turkish stem evidence at all) which is never even a
# candidate. These four cover every branch of _rerank_token_label.
# ---------------------------------------------------------------------------

def _make_annotator(turkish_top=(), turkish_all=(), english_words=()):
    """Same bypass-__init__ convention as tests/test_cs_pipeline.py and
    tests/test_mixed_reranker.py -- no frequency-file reads, no
    fasttext.load_model() call."""
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = set(turkish_top)
    obj.turkish_freq_all = set(turkish_all)
    obj.english_freq_words = set(english_words)
    return obj


@pytest.fixture(scope="module")
def real_bundle():
    bundle = ri.load_reranker_bundle(REAL_MODEL_DIR)
    assert bundle is not None  # sanity: the fixture itself must load successfully
    return bundle


def _sentence_block(sent_id, token_label_pairs, matrix=None, embed=None):
    """Build one block of annotate()'s raw text output by hand: SentenceID,
    then one 'Item\\tLabel' line per pair, then MatrixLang/EmbedLang if
    given (omitted entirely when None, matching annotate()'s own
    FEATURE_MATRIX_LANGUAGE/FEATURE_EMBEDDED_LANGUAGE toggles)."""
    lines = [f"SentenceID\t{sent_id}"]
    for item, label in token_label_pairs:
        lines.append(f"{item}\t{label}")
    if matrix is not None:
        lines.append(f"MatrixLang\t{matrix}")
    if embed is not None:
        lines.append(f"EmbedLang\t{embed}")
    return "\n".join(lines)


def test_apply_reranker_bundle_none_returns_text_unchanged():
    text = _sentence_block(1, [("kitap", "TR"), ("amazing", "EN")], matrix="TR", embed="EN")
    obj = _make_annotator()
    result = ri.apply_reranker(text, obj, DEFAULTS, None)
    assert result == text


def test_apply_reranker_no_candidate_tokens_unchanged():
    # EN/OTHER/MIXED/LANG3 labels are never candidate-eligible -- nothing
    # for the reranker to inspect, block must come back byte-identical.
    text = _sentence_block(1, [("amazing", "EN"), ("42", "OTHER"), ("boss'um", "MIXED")],
                            matrix="EN", embed="TR")
    obj = _make_annotator()
    result = ri.apply_reranker(text, obj, DEFAULTS, mock.Mock())  # bundle content irrelevant here
    assert result == text


def test_apply_reranker_does_not_call_classify_candidate_for_ineligible_labels():
    text = _sentence_block(1, [("amazing", "EN"), ("42", "OTHER")], matrix="EN", embed="-")
    obj = _make_annotator()
    with mock.patch("reranker_integration.mr.classify_candidate") as mock_classify:
        ri.apply_reranker(text, obj, DEFAULTS, mock.Mock())
    mock_classify.assert_not_called()


def test_apply_reranker_candidate_rejected_below_threshold_stays_same_label(real_bundle):
    obj = _make_annotator(english_words={"app"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(_sentence_block(1, [("applar", "TR")], matrix="TR", embed="-"),
                                    obj, DEFAULTS, real_bundle)
    assert "applar\tTR" in result  # P=0.819 < 0.85 -- rejected, label unchanged


def test_apply_reranker_candidate_rejected_weak_evidence_stays_same_label(real_bundle):
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.1)):
        result = ri.apply_reranker(_sentence_block(1, [("zzqxwv", "UID")], matrix="-", embed="-"),
                                    obj, DEFAULTS, real_bundle)
    assert "zzqxwv\tUID" in result  # P=0.045 -- clearly rejected


def test_apply_reranker_non_candidate_bucket_never_reranked(real_bundle):
    # "masada" is already fully Turkish (in turkish_all) -- classify_candidate
    # never grants it candidacy at all, regardless of model probability.
    obj = _make_annotator(turkish_all={"masada"})
    with mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
        result = ri.apply_reranker(_sentence_block(1, [("masada", "TR")], matrix="TR", embed="-"),
                                    obj, DEFAULTS, real_bundle)
    assert "masada\tTR" in result


def test_apply_reranker_candidate_promoted_to_mixed(real_bundle):
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="-", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "boostlamak\tMIXED" in result
    assert "boostlamak\tUID" not in result


def test_apply_reranker_promotion_is_deterministic(real_bundle):
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="-", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result1 = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
        result2 = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert result1 == result2


def test_apply_reranker_recomputes_matrix_embed_when_label_changes(real_bundle):
    # Sentence has one TR token and the promoted-to-MIXED token; before
    # promotion labels_in_sent=["TR"] (UID never counted) -> Matrix=TR,
    # Embed="-". After promotion labels_in_sent=["TR","MIXED"] -> Embed
    # must become "EN" (matrix stays TR, MIXED now counts as an EN-ish
    # signal for embed purposes per Annotator._decide_matrix_embed).
    obj = _make_annotator(english_words={"boost"}, turkish_all={"kitap"})
    original = _sentence_block(1, [("kitap", "TR"), ("boostlamak", "UID")], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "boostlamak\tMIXED" in result
    assert "MatrixLang\tTR" in result
    assert "EmbedLang\tEN" in result
    assert "EmbedLang\t-" not in result


def test_apply_reranker_leaves_matrix_embed_untouched_when_nothing_changes(real_bundle):
    obj = _make_annotator()
    original = _sentence_block(1, [("zzqxwv", "UID")], matrix="TR", embed="EN")
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.1)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert result == original  # byte-for-byte identical, block never touched


def test_apply_reranker_only_rewrites_meta_rows_that_exist(real_bundle):
    # MatrixLang present, EmbedLang absent (as if FEATURE_EMBEDDED_LANGUAGE
    # was False when annotate() ran) -- must not invent an EmbedLang row.
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="TR", embed=None)
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "boostlamak\tMIXED" in result
    assert "EmbedLang" not in result


def test_apply_reranker_multiple_blocks_are_independent(real_bundle):
    block1 = _sentence_block(1, [("boostlamak", "UID")], matrix="-", embed="-")
    block2 = _sentence_block(2, [("zzqxwv", "UID")], matrix="-", embed="-")
    text = block1 + "\n\n" + block2
    obj = _make_annotator(english_words={"boost"})
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(text, obj, DEFAULTS, real_bundle)
    parts = result.split("\n\n")
    assert len(parts) == 2
    assert "boostlamak\tMIXED" in parts[0]
    # block2's token used a different mocked fastText response in isolation
    # elsewhere, but here it shares obj/mock with block1 (EN, 0.95) --
    # what matters is block2 is processed independently, not bundled into
    # block1's decision.
    assert "SentenceID\t2" in parts[1]


def test_apply_reranker_preserves_sentence_id_and_other_meta_lines_verbatim(real_bundle):
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="TR", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "SentenceID\t1" in result.split("\n")


def test_apply_reranker_exact_output_format_no_extra_fields(real_bundle):
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="TR", embed="EN")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    for line in result.split("\n"):
        assert line.count("\t") == 1  # every non-blank line is still exactly 2 tab-separated fields


def test_apply_reranker_ne_candidate_path(real_bundle):
    # NE tokens can also be reranked (CANDIDATE_REASON_NE_SUFFIX), distinct
    # from the UID and TR paths already covered above.
    obj = _make_annotator()
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.1)):
        result = ri.apply_reranker(_sentence_block(1, [("zzqxwv", "NE")], matrix="-", embed="-"),
                                    obj, DEFAULTS, real_bundle)
    assert "zzqxwv\tNE" in result  # weak evidence -- rejected, but path is exercised


def test_apply_reranker_never_raises_on_malformed_block(real_bundle):
    # A block with an unexpected line shape (e.g. 3 tab-separated fields)
    # must be passed through verbatim rather than raise.
    text = "SentenceID\t1\nweird\tline\twith\textra\tfields"
    obj = _make_annotator()
    result = ri.apply_reranker(text, obj, DEFAULTS, real_bundle)
    assert "weird\tline\twith\textra\tfields" in result


# ---------------------------------------------------------------------------
# Phase 6.3: GUI wiring regression guard. cs_annotator_app.py's GUI methods
# have no other automated coverage (its App class needs a display to
# instantiate), so this checks the wiring structurally via source
# inspection -- the same technique test_module_does_not_import_joblib_at_top_level
# already uses above -- rather than building new GUI test infrastructure.
# Importing cs_annotator_app itself needs no display; only instantiating
# App() would.
# ---------------------------------------------------------------------------

def test_run_pipeline_calls_reranker_in_the_expected_order():
    import inspect
    import cs_annotator_app
    src = inspect.getsource(cs_annotator_app.App.run_pipeline)
    load_pos = src.index("reranker_integration.load_reranker_bundle")
    annotate_pos = src.index(".annotate(")
    apply_pos = src.index("reranker_integration.apply_reranker")
    consistency_pos = src.index("_ensure_matrix_embed_consistency")
    populate_pos = src.index("_populate_table")
    assert load_pos < annotate_pos < apply_pos < consistency_pos < populate_pos


def test_app_init_declares_reranker_cache_slots():
    import inspect
    import cs_annotator_app
    src = inspect.getsource(cs_annotator_app.App.__init__)
    assert "self._reranker_bundle = None" in src
    assert "self._reranker_load_attempted = False" in src


# ---------------------------------------------------------------------------
# Residual verbal MIXED detector -- production wiring inside apply_reranker().
# Strict evidence only (mixed_reranker.evaluate_residual_verbal_promotion's
# default); this file only tests PLACEMENT, fail-safety, and Matrix/Embed
# interaction with the frozen-reranker stage -- the detection RULE itself
# (parsing/evidence/gating) is tested exhaustively in test_mixed_reranker.py.
# ---------------------------------------------------------------------------

def test_apply_reranker_residual_stage_runs_after_rerank_loop_in_source():
    # Item 14: structural placement guarantee, same technique as
    # test_module_does_not_import_joblib_at_top_level above.
    import inspect
    src = inspect.getsource(ri.apply_reranker)
    rerank_pos = src.index("_rerank_token_label(")
    residual_pos = src.index("_apply_residual_verbal_promotion(")
    assert rerank_pos < residual_pos


def test_residual_stage_never_reconsiders_a_token_already_promoted_by_frozen_reranker(real_bundle):
    # Item 14 (functional): "boostlamak" would ALSO qualify for the
    # residual stage on its own (stem "boost" + verbalizer+infinitive) --
    # this proves the residual stage's eligibility check sees the
    # ALREADY-promoted post-rerank label and never even calls the residual
    # evaluator for it, rather than merely producing the same answer twice.
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="-", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)), \
         mock.patch("reranker_integration.mr.evaluate_residual_verbal_promotion") as mock_residual:
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "boostlamak\tMIXED" in result
    mock_residual.assert_not_called()


def test_residual_promotion_after_frozen_reranker_declines(real_bundle):
    # The frozen reranker itself is disabled here (classify_candidate always
    # non-candidate) so the promotion below is attributable ONLY to the
    # residual stage -- exercises the real production
    # evaluate_residual_verbal_promotion end to end, not a mock.
    obj = _make_annotator(english_words={"design"})
    original = _sentence_block(1, [("designladık", "TR")], matrix="-", embed="-")
    with mock.patch("reranker_integration.mr.classify_candidate", return_value=(False, None, None)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "designladık\tMIXED" in result


def test_matrix_embed_recomputed_when_only_residual_stage_promotes(real_bundle):
    # Item 15: Matrix/Embedded Language recomputation must fire even when
    # the ONLY label change in the block came from the residual stage, not
    # the frozen reranker.
    obj = _make_annotator(english_words={"design"}, turkish_all={"kitap"})
    original = _sentence_block(1, [("kitap", "TR"), ("designladık", "TR")], matrix="TR", embed="-")
    with mock.patch("reranker_integration.mr.classify_candidate", return_value=(False, None, None)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "designladık\tMIXED" in result
    assert "MatrixLang\tTR" in result
    assert "EmbedLang\tEN" in result
    assert "EmbedLang\t-" not in result


def test_residual_stage_never_touches_non_uid_tr_labels(real_bundle):
    # Item 16: MIXED/EN/OTHER/NE/LANG3 rows must never be inspected by the
    # residual stage, even when their text would otherwise structurally
    # qualify (e.g. "designladık" already labeled MIXED).
    obj = _make_annotator(english_words={"design"})
    original = _sentence_block(1, [
        ("designladık", "MIXED"),
        ("amazing", "EN"),
        ("42", "OTHER"),
        ("Ankara", "NE"),
        ("bonjour", "LANG3"),
    ], matrix="-", embed="-")
    with mock.patch("reranker_integration.mr.classify_candidate", return_value=(False, None, None)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert result == original  # byte-for-byte identical -- nothing eligible for either stage


def test_residual_promotion_never_raises_preserves_label_and_continues(real_bundle):
    # Item 13: fail-safe contract -- an unexpected exception from the
    # residual evaluator for one token must not interrupt the rest of the
    # block, and must leave that token's own label untouched.
    obj = _make_annotator(english_words={"design"})
    original = _sentence_block(1, [("designladık", "TR"), ("kitap", "TR")], matrix="TR", embed="-")
    with mock.patch("reranker_integration.mr.classify_candidate", return_value=(False, None, None)), \
         mock.patch("reranker_integration.mr.evaluate_residual_verbal_promotion",
                     side_effect=RuntimeError("boom")):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "designladık\tTR" in result
    assert "kitap\tTR" in result


def test_frozen_reranker_output_unchanged_by_presence_of_residual_stage(real_bundle):
    # Item 17: with the residual stage forced to never promote, the
    # existing frozen-reranker-only fixtures (applar/zzqxwv/masada/
    # boostlamak) must behave exactly as they did before this integration
    # -- i.e. the frozen model's own decision-making is untouched.
    obj = _make_annotator(english_words={"boost"})
    original = _sentence_block(1, [("boostlamak", "UID")], matrix="-", embed="-")
    with mock.patch.object(obj, "_ft_predict", return_value=("EN", 0.95)), \
         mock.patch("reranker_integration.mr.evaluate_residual_verbal_promotion",
                     return_value=(False, None, "no_verbal_candidate")):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert "boostlamak\tMIXED" in result


# ---------------------------------------------------------------------------
# Residual verbal MIXED detector -- production regression targets (item 11)
# and native-Turkish controls (item 12), exercised through the FULL
# apply_reranker() wiring (frozen reranker disabled, so promotions below are
# attributable only to the residual stage).
# ---------------------------------------------------------------------------

_RESIDUAL_PROD_TARGETS = [
    "uploadlamışım", "designladık", "inviteladık", "filterladık", "forwardladı",
    "mutelamışım", "refreshledik", "cancelladık", "editledik", "dropladı",
]

_RESIDUAL_PROD_CONTROLS = [
    "anladık", "yapmışım", "bekledik", "konuşuyoruz", "gideceksin", "dinledim",
    "başladık", "temizledik", "eklemişim", "yazdırdık", "kapattılar",
    "paylaşmışım", "güncelledik", "alamadık", "alıyoruz", "biliyonuz",
    "dediğimiz", "düşünüyoruz", "giremiyoruz", "gittik", "istiyoruz",
    "kaldık", "olduğumuz", "olmuyorsunuz", "rahatlarız", "bilirdik",
]

_RESIDUAL_PROD_STEMS = {"upload", "design", "invite", "filter", "forward",
                         "mute", "refresh", "cancel", "edit", "drop"}


@pytest.mark.parametrize("token", _RESIDUAL_PROD_TARGETS)
def test_residual_production_targets_promote_via_apply_reranker(token, real_bundle):
    obj = _make_annotator(english_words=_RESIDUAL_PROD_STEMS)
    original = _sentence_block(1, [(token, "TR")], matrix="-", embed="-")
    with mock.patch("reranker_integration.mr.classify_candidate", return_value=(False, None, None)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert f"{token}\tMIXED" in result, f"{token} was not promoted: {result}"


@pytest.mark.parametrize("token", _RESIDUAL_PROD_CONTROLS)
def test_residual_production_native_controls_never_promote_via_apply_reranker(token, real_bundle):
    obj = _make_annotator(english_words=_RESIDUAL_PROD_STEMS)
    original = _sentence_block(1, [(token, "TR")], matrix="-", embed="-")
    with mock.patch("reranker_integration.mr.classify_candidate", return_value=(False, None, None)):
        result = ri.apply_reranker(original, obj, DEFAULTS, real_bundle)
    assert f"{token}\tMIXED" not in result, f"{token} was incorrectly promoted: {result}"
