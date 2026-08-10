# tests/test_confidence_integration.py
"""Integration tests for the confidence/review-tool layer: pipeline wiring,
review-tool filtering (label + confidence band), main-table synchronization,
dataset switching, and .trenproj save/load compatibility (including legacy
projects predating this layer).

Requires a real, working Tk display for the GUI-driving tests below (same
convention as tests/test_confidence_review_tool_gui.py); those are skipped
entirely if none is available. The save/load tests are pure (no Tk) and
always run.
"""
import copy
import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import annotation_model
import confidence as cf


def _tk_available():
    try:
        import tkinter as tk
        probe = tk.Tk()
        probe.destroy()
        return True
    except Exception:
        return False


TK_AVAILABLE = _tk_available()


# ---------------------------------------------------------------------------
# .trenproj save/load compatibility (pure, no Tk)
# ---------------------------------------------------------------------------

def _dataset_with_confidence():
    blocks = [[
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "backend", "label": "UID", "gloss": "",
         "confidence": {"token": "backend", "rule_based_label": "UID", "final_label": "UID",
                         "confidence_score": 0.42, "confidence_band": "LOW",
                         "uncertainty_reasons": ["label_uid_by_definition_below_lid_confidence_threshold"],
                         "evidence_summary": [], "review_recommended": True, "promoted_by": None,
                         "calibration_note": cf.CALIBRATION_NOTE},
         "reviewed": False},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
    ]]
    return annotation_model.make_dataset("Data 1", "backend.", blocks, [])


def test_confidence_and_reviewed_survive_json_round_trip():
    ds = _dataset_with_confidence()
    payload = annotation_model.datasets_to_payload([ds], 0)
    raw = json.loads(json.dumps(payload))  # exactly what save_project_progress/open_project_save do
    datasets, active_index = annotation_model.datasets_from_payload(raw)
    row = datasets[0]["blocks"][0][1]
    assert row["confidence"]["confidence_band"] == "LOW"
    assert row["confidence"]["confidence_score"] == 0.42
    assert row["reviewed"] is False


def test_confidence_survives_after_manual_edit_round_trip():
    ds = _dataset_with_confidence()
    row = ds["blocks"][0][1]
    row["label"] = "TR"
    cf.note_manual_edit(row)
    payload = annotation_model.datasets_to_payload([ds], 0)
    raw = json.loads(json.dumps(payload))
    datasets, _ = annotation_model.datasets_from_payload(raw)
    loaded_row = datasets[0]["blocks"][0][1]
    assert loaded_row["label"] == "TR"
    assert loaded_row["confidence"]["promoted_by"] == "manual_edit"
    assert loaded_row["reviewed"] is True


def test_legacy_version_1_project_without_confidence_loads_cleanly():
    # A version-1 (or version-less) payload predates this layer entirely --
    # no "confidence"/"reviewed" keys anywhere on any row.
    legacy_payload = {
        "version": 1,
        "blocks": [[
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "kitap", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
        ]],
        "input_text": "kitap.",
        "extra_headers": [],
    }
    datasets, active_index = annotation_model.datasets_from_payload(legacy_payload)
    assert active_index == 0
    row = datasets[0]["blocks"][0][1]
    assert "confidence" not in row
    # The review tool's accessors must treat this as "not yet computed",
    # never crash and never fabricate a fake result.
    assert cf.get_confidence(row) is None
    assert cf.is_reviewed(row) is False
    assert cf.band_of(row) is None


def test_legacy_project_without_version_key_defaults_to_v1_and_loads():
    legacy_payload = {
        "blocks": [[{"idx": 1, "token": "x", "label": "TR", "gloss": ""}]],
        "input_text": "x",
        "extra_headers": [],
    }
    datasets, active_index = annotation_model.datasets_from_payload(legacy_payload)
    assert datasets[0]["blocks"][0][0]["label"] == "TR"
    assert cf.get_confidence(datasets[0]["blocks"][0][0]) is None


# ---------------------------------------------------------------------------
# No automatic label changes: attach_confidence_to_blocks over a realistic,
# multi-label block never mutates any label/token/gloss.
# ---------------------------------------------------------------------------

def test_attach_confidence_over_full_label_set_never_mutates_rows():
    from cs_pipeline import Annotator, DEFAULTS
    obj = Annotator.__new__(Annotator)
    obj.turkish_freq_top = {"kitap"}
    obj.turkish_freq_all = {"kitap", "mey"}
    obj.english_freq_words = {"cool", "boost"}
    obj.ner = None

    rows = [
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "kitap", "label": "TR", "gloss": ""},
        {"idx": 2, "token": "cool", "label": "EN", "gloss": ""},
        {"idx": 3, "token": "boss'um", "label": "MIXED", "gloss": ""},
        {"idx": 4, "token": "zzqxwv", "label": "UID", "gloss": ""},
        {"idx": 5, "token": "Ankara", "label": "NE", "gloss": ""},
        {"idx": 6, "token": "42", "label": "OTHER", "gloss": ""},
        {"idx": 7, "token": "bonjour", "label": "LANG3", "gloss": ""},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
    ]
    before = copy.deepcopy(rows)
    with mock.patch.object(obj, "_ft_predict", return_value=("XX", 0.0)):
        cf.attach_confidence_to_blocks([rows], [rows], obj, DEFAULTS)

    for r_before, r_after in zip(before, rows):
        assert r_before["token"] == r_after["token"]
        assert r_before["label"] == r_after["label"]
        assert r_before["gloss"] == r_after["gloss"]


# ---------------------------------------------------------------------------
# GUI-driven: review-tool label/band filtering, main-table sync, dataset
# switching. Skipped without a real Tk display.
# ---------------------------------------------------------------------------

pytestmark_gui = pytest.mark.skipif(not TK_AVAILABLE, reason="No Tk display available.")

if TK_AVAILABLE:
    import cs_annotator_app as caa


def _confidence_dict(band, score, reviewed_reasons=None):
    return {"token": "", "rule_based_label": None, "final_label": "", "confidence_score": score,
            "confidence_band": band, "uncertainty_reasons": reviewed_reasons or [],
            "evidence_summary": [], "review_recommended": band != "HIGH", "promoted_by": None,
            "calibration_note": cf.CALIBRATION_NOTE}


def _mixed_label_blocks():
    return [[
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "backend", "label": "UID", "gloss": "",
         "confidence": _confidence_dict("LOW", 0.4), "reviewed": False},
        {"idx": 2, "token": "kelime", "label": "TR", "gloss": "",
         "confidence": _confidence_dict("MEDIUM", 0.7), "reviewed": False},
        {"idx": 3, "token": "cool", "label": "EN", "gloss": "",
         "confidence": _confidence_dict("HIGH", 0.95), "reviewed": True},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
    ]]


def make_app(blocks):
    app = caa.App()
    app.blocks = blocks
    app._renumber_tokens()
    app._rebuild_grid_from_model()
    app.update()
    return app


@pytestmark_gui
def test_review_tool_default_combobox_value_is_all_uncertain():
    """Requirement: the default combobox value is "All Uncertain" -- the
    literal string, exactly, with no interaction needed to reach it."""
    app = make_app(_mixed_label_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        assert app._review_view_var.get() == "All Uncertain"
        assert app._review_view_combo.get() == "All Uncertain"
        assert app._review_view_mode == 'all_uncertain'
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_default_view_is_all_uncertain_across_labels():
    """The tool's default view is now "All Uncertain": every token (any
    label) whose confidence record has review_recommended=True -- backend
    (UID, LOW) and kelime (TR, MEDIUM) both qualify; cool (EN, HIGH,
    review_recommended=False) must not appear even though it's already
    marked reviewed=True is irrelevant here -- it's excluded purely for
    being confident, not because it was reviewed."""
    app = make_app(_mixed_label_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        assert app._review_view_var.get() == app.REVIEW_VIEW_ALL_UNCERTAIN
        tokens = sorted(app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items)
        assert tokens == ['backend', 'kelime']
        assert 'cool' not in tokens
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_uid_only_view_still_reproduces_old_default():
    """Switching the view combobox to "UID Only" must reproduce exactly
    the tool's original pre-"All Uncertain" default: UID-labeled tokens
    only, regardless of confidence band."""
    app = make_app(_mixed_label_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        app._review_view_var.set(app.REVIEW_VIEW_UID_ONLY)
        app._review_view_combo.event_generate('<<ComboboxSelected>>')
        app.update()
        tokens = [app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items]
        assert tokens == ['backend']
    finally:
        app.destroy()


def _all_seven_labels_blocks():
    """One row per schema label, each with an explicit confidence record --
    the odd-indexed ones (index 1, 3, 5) marked confident/HIGH
    (review_recommended=False), the rest marked uncertain
    (review_recommended=True) -- so the expected "All Uncertain" result set
    spans TR/MIXED/UID/LANG3 (uncertain) while EN/NE/OTHER (confident) are
    excluded, deliberately not correlated with any single label."""
    def conf(label, uncertain):
        return {
            "token": "", "rule_based_label": label, "final_label": label,
            "confidence_score": 0.4 if uncertain else 0.95,
            "confidence_band": "LOW" if uncertain else "HIGH",
            "uncertainty_reasons": [], "evidence_summary": [],
            "review_recommended": uncertain, "promoted_by": None,
            "calibration_note": cf.CALIBRATION_NOTE,
        }

    rows = [{"idx": "", "token": "SentenceID", "label": "1", "gloss": ""}]
    plan = [
        ("tok_tr", "TR", True), ("tok_en", "EN", False), ("tok_mixed", "MIXED", True),
        ("tok_ne", "NE", False), ("tok_uid", "UID", True), ("tok_other", "OTHER", False),
        ("tok_lang3", "LANG3", True),
    ]
    for i, (tok, label, uncertain) in enumerate(plan, start=1):
        rows.append({"idx": i, "token": tok, "label": label, "gloss": "",
                      "confidence": conf(label, uncertain)})
    rows.append({"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""})
    rows.append({"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""})
    return [rows]


@pytestmark_gui
def test_review_tool_default_view_includes_uncertain_tokens_from_every_label():
    """Requirement: tokens from different labels appear in the initial
    (default "All Uncertain") list purely because review_recommended=True
    is set on their confidence record -- TR/MIXED/UID/LANG3 here -- never
    because of what their label literally is."""
    app = make_app(_all_seven_labels_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        assert app._review_view_var.get() == app.REVIEW_VIEW_ALL_UNCERTAIN
        tokens_and_labels = sorted(
            (app.blocks[it['bidx']][it['ridx']]['token'], app.blocks[it['bidx']][it['ridx']]['label'])
            for it in app._uid_items
        )
        assert tokens_and_labels == [
            ("tok_lang3", "LANG3"), ("tok_mixed", "MIXED"), ("tok_tr", "TR"), ("tok_uid", "UID"),
        ]
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_default_view_excludes_confident_tokens_from_every_label():
    """Requirement: confident tokens (review_recommended=False) do not
    appear in the default view, regardless of label -- EN/NE/OTHER here."""
    app = make_app(_all_seven_labels_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        tokens = {app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items}
        assert tokens.isdisjoint({"tok_en", "tok_ne", "tok_other"})
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_all_uncertain_view_handles_empty_queue_cleanly():
    """When nothing is flagged for review (every row confident, or no
    confidence data at all), the default "All Uncertain" view must show an
    empty, valid list -- no crash, a "0 of 0" title, and a cleared editor
    -- never a stale/garbage selection."""
    blocks = [[
        {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
        {"idx": 1, "token": "kitap", "label": "TR", "gloss": "",
         "confidence": {"token": "kitap", "rule_based_label": "TR", "final_label": "TR",
                        "confidence_score": 0.95, "confidence_band": "HIGH",
                        "uncertainty_reasons": [], "evidence_summary": [],
                        "review_recommended": False, "promoted_by": None,
                        "calibration_note": cf.CALIBRATION_NOTE}},
        {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
        {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
    ]]
    app = make_app(blocks)
    try:
        app.open_uid_review_tool()
        app.update()
        assert app._review_view_var.get() == app.REVIEW_VIEW_ALL_UNCERTAIN
        assert app._uid_items == []
        assert app._uid_i == 0
        assert app._uid_win.title().endswith("0 of 0")
        assert app._uid_label_var.get() == ""
        assert app._uid_gloss_var.get() == ""
        assert app._uid_context_var.get() == ""
        # First/Previous/Next/Last/Apply/Undo must all be safe no-ops, not crashes.
        app._uid_first()
        app._uid_prev()
        app._uid_next()
        app._uid_last()
        app._uid_apply()
        app._uid_undo_last()
        app.update()
        assert app._uid_items == []
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_label_and_band_filters_via_real_checkbutton_clicks():
    app = make_app(_mixed_label_blocks())
    try:
        app.open_uid_review_tool()
        app.update()

        # Uncheck UID, check TR and EN (real clicks on the actual Checkbuttons)
        def find_checkbuttons(w):
            out = {}
            for c in w.winfo_children():
                if isinstance(c, caa.ttk.Checkbutton):
                    out[c.cget('text')] = c
                out.update(find_checkbuttons(c))
            return out

        buttons = find_checkbuttons(app._uid_win)
        for lab in ("TR", "EN", "MIXED", "NE", "OTHER", "LANG3"):
            buttons[lab].invoke()
        buttons["UID"].invoke()  # toggle off
        app.update()

        tokens = sorted(app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items)
        assert tokens == ['cool', 'kelime']  # backend (UID) excluded

        # Now restrict confidence band to Low
        buttons["Low"].invoke()
        app.update()
        tokens2 = [app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items]
        assert tokens2 == []  # neither TR(MEDIUM) nor EN(HIGH) is LOW

        buttons["Low"].invoke()  # back off
        buttons["Medium"].invoke()
        app.update()
        tokens3 = [app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items]
        assert tokens3 == ['kelime']  # only the MEDIUM-band TR token
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_hide_reviewed_checkbox():
    app = make_app(_mixed_label_blocks())
    try:
        app.open_uid_review_tool()
        app.update()

        def find_checkbuttons(w):
            out = {}
            for c in w.winfo_children():
                if isinstance(c, caa.ttk.Checkbutton):
                    out[c.cget('text')] = c
                out.update(find_checkbuttons(c))
            return out

        buttons = find_checkbuttons(app._uid_win)
        for lab in ("TR", "EN"):
            buttons[lab].invoke()
        buttons["Hide reviewed"].invoke()
        app.update()

        tokens = sorted(app.blocks[it['bidx']][it['ridx']]['token'] for it in app._uid_items)
        assert 'cool' not in tokens  # reviewed=True, hidden
        assert 'kelime' in tokens
    finally:
        app.destroy()


@pytestmark_gui
def test_apply_from_review_tool_marks_manual_edit_and_syncs_main_table():
    app = make_app(_mixed_label_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        it = app._uid_items[0]
        bidx, ridx, vis_r = it['bidx'], it['ridx'], it['vis_r']
        app._uid_label_var.set('TR')
        app._uid_gloss_var.set('arka uc')
        app._uid_apply()
        app.update()

        row = app.blocks[bidx][ridx]
        assert row['label'] == 'TR'
        assert row['confidence']['promoted_by'] == 'manual_edit'
        assert row['reviewed'] is True
        # main sheet reflects the change immediately
        assert app.sheet.get_cell_data(vis_r, 2) == 'TR'
    finally:
        app.destroy()


@pytestmark_gui
def test_review_tool_never_touches_other_datasets():
    app = caa.App()
    try:
        ds1 = annotation_model.make_dataset("Data 1", "", _mixed_label_blocks(), [])
        ds2_blocks = [[
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "other", "label": "UID", "gloss": "",
             "confidence": _confidence_dict("LOW", 0.3), "reviewed": False},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
        ]]
        ds2 = annotation_model.make_dataset("Data 2", "", ds2_blocks, [])
        app.datasets = [ds1, ds2]
        app._load_dataset_into_live(0)
        app.update()

        app.open_uid_review_tool()
        app.update()
        it = app._uid_items[0]
        app._uid_label_var.set('EN')
        app._uid_apply()
        app.update()

        # Dataset 2 (never made active) must be byte-identical to before.
        assert app.datasets[1]['blocks'] == ds2_blocks
        assert app.datasets[1]['blocks'][0][1]['label'] == 'UID'
    finally:
        app.destroy()


@pytestmark_gui
def test_dataset_switching_preserves_independent_confidence_data():
    app = caa.App()
    try:
        ds1 = annotation_model.make_dataset("Data 1", "", _mixed_label_blocks(), [])
        ds2_blocks = [[
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "other", "label": "TR", "gloss": "",
             "confidence": _confidence_dict("HIGH", 0.9), "reviewed": True},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
        ]]
        ds2 = annotation_model.make_dataset("Data 2", "", ds2_blocks, [])
        app.datasets = [ds1, ds2]

        app._load_dataset_into_live(0)
        app.update()
        assert app.blocks[0][1]['confidence']['confidence_band'] == 'LOW'

        app._load_dataset_into_live(1)
        app.update()
        assert app.blocks[0][1]['token'] == 'other'
        assert app.blocks[0][1]['confidence']['confidence_band'] == 'HIGH'
        assert app.blocks[0][1]['reviewed'] is True

        app._load_dataset_into_live(0)
        app.update()
        assert app.blocks[0][1]['token'] == 'backend'
        assert app.blocks[0][1]['confidence']['confidence_band'] == 'LOW'
    finally:
        app.destroy()


@pytestmark_gui
def test_run_pipeline_attaches_confidence_without_changing_labels():
    """End-to-end pipeline wiring: run_pipeline() -> _populate_table() ->
    _attach_confidence() must produce confidence data for every token row
    without altering the labels the (stubbed) production pipeline produced.
    """
    from cs_pipeline import Annotator, DEFAULTS

    app = caa.App()
    try:
        obj = Annotator.__new__(Annotator)
        obj.turkish_freq_top = {"kitap"}
        obj.turkish_freq_all = {"kitap"}
        obj.english_freq_words = {"cool"}
        obj.ner = None
        app.annotator = obj
        app._reranker_bundle = None
        app._reranker_load_attempted = True
        app.cfg = dict(DEFAULTS)
        app.cfg["NER_ENABLED"] = False

        pipeline_text = (
            "SentenceID\t1\n"
            "kitap\tTR\n"
            "cool\tEN\n"
            "MatrixLang\tTR\n"
            "EmbedLang\tEN\n"
        )
        with mock.patch.object(app, "_run_annotation_pipeline", return_value=pipeline_text) as mocked, \
             mock.patch.object(obj, "_ft_predict", return_value=("TR", 0.9)):
            app._last_rule_based_output = pipeline_text  # what the real method would have stashed
            app.txt_input.delete("1.0", "end")
            app.txt_input.insert("1.0", "kitap cool.")
            app.run_pipeline()
            app.update()
        mocked.assert_called_once()

        labels_by_token = {r['token']: r['label'] for r in app.blocks[0]
                            if not annotation_model.is_meta_row_token(r['token'])}
        assert labels_by_token == {"kitap": "TR", "cool": "EN"}
        for r in app.blocks[0]:
            if not annotation_model.is_meta_row_token(r['token']):
                assert cf.get_confidence(r) is not None
    finally:
        app.destroy()
