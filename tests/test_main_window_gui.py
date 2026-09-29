"""GUI regression tests for main-window fixes: quit paths honoring the
unsaved-changes guard, the relabel panel's busy flag, the Run-over-existing-
annotations confirmation, Label-column validation, toolbar/config sync after
opening a project, Open Input read errors, and the reranker status indicator.
Real Tk; skipped without a display (CI runs it under xvfb).
"""
import copy
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk  # noqa: E402

import cs_annotator_app as caa  # noqa: E402
import reranking  # noqa: E402


def _tk_available():
    try:
        probe = tk.Tk()
        probe.destroy()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _tk_available(),
    reason="No Tk display available in this environment; GUI tests require a real display.",
)


@pytest.fixture(autouse=True)
def _isolated_app_dir(monkeypatch, tmp_path):
    app_dir = tmp_path / "cs_annotator_home"
    monkeypatch.setattr(caa, "APP_DIR", str(app_dir))
    monkeypatch.setattr(caa, "LAST_PROJECT_PTR", str(app_dir / "last_project.json"))


ANNOTATED = (
    "SentenceID\t1\nkitap\tTR\namazing\tEN\nboss'um\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n\n"
    "SentenceID\t2\nbugün\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
)


class _Dialogs:
    """Records every modal messagebox call; each returns a preset answer, so
    no test can block on a real dialog."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.answers = {"askokcancel": True, "askyesnocancel": None, "askyesno": True}
        for name in ("showwarning", "showerror", "showinfo",
                     "askokcancel", "askyesnocancel", "askyesno"):
            monkeypatch.setattr(caa.messagebox, name, self._make(name))

    def _make(self, name):
        def _dialog(*a, **k):
            self.calls.append((name, a, k))
            return self.answers.get(name)
        return _dialog

    def names(self):
        return [c[0] for c in self.calls]


def make_app(monkeypatch, pipeline_output=ANNOTATED):
    dialogs = _Dialogs(monkeypatch)
    app = caa.App()
    app.update()
    monkeypatch.setattr(app, "_run_annotation_pipeline", lambda text: pipeline_output)
    monkeypatch.setattr(app, "_attach_confidence", lambda blocks: None)
    return app, dialogs


def run_once(app, text="kitap amazing boss'um\n\nbugün"):
    app._set_txt_input_text(text)
    app.run_pipeline()
    app.update()


def vis_row(app, token):
    return next(v for v, (b, r) in app._row_index_map.items()
                if b is not None and app.blocks[b][r].get('token') == token)


def row(app, token):
    return next(r for b in app.blocks for r in b if r.get('token') == token)


def track_destroy(app):
    calls = {"n": 0}
    app.destroy = lambda: calls.__setitem__("n", calls["n"] + 1)
    return calls


def file_menu(app):
    menubar = app.nametowidget(app['menu'])
    return app.nametowidget(menubar.entrycget(0, 'menu'))


def menu_labels(menu):
    return [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)
            if menu.type(i) == 'command']


def menu_index(menu, label):
    return next(i for i in range(menu.index('end') + 1)
                if menu.type(i) == 'command' and menu.entrycget(i, 'label') == label)


# =========================================================================
# Quit paths: File > Exit and macOS Quit (Cmd+Q) share the close guard
# =========================================================================

def test_file_exit_with_unsaved_changes_prompts_and_cancel_keeps_app_open(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    destroyed = track_destroy(app)
    try:
        run_once(app)
        assert app._dirty is True
        dialogs.answers["askyesnocancel"] = None  # Cancel
        menu = file_menu(app)
        menu.invoke(menu_index(menu, "Exit"))
        assert "askyesnocancel" in dialogs.names()
        assert destroyed["n"] == 0
        assert app._dirty is True
    finally:
        tk.Tk.destroy(app)


def test_file_exit_when_clean_closes_without_prompt(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    destroyed = track_destroy(app)
    try:
        menu = file_menu(app)
        menu.invoke(menu_index(menu, "Exit"))
        assert "askyesnocancel" not in dialogs.names()
        assert destroyed["n"] == 1
    finally:
        tk.Tk.destroy(app)


def test_file_exit_failed_save_keeps_app_open(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    destroyed = track_destroy(app)
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: None)  # save dialog cancelled
    try:
        run_once(app)
        dialogs.answers["askyesnocancel"] = True  # "Save"
        menu = file_menu(app)
        menu.invoke(menu_index(menu, "Exit"))
        assert destroyed["n"] == 0
        assert app._dirty is True
    finally:
        tk.Tk.destroy(app)


def test_mac_quit_command_goes_through_close_guard(monkeypatch):
    # Cmd+Q / app-menu Quit / Dock Quit invoke ::tk::mac::Quit on macOS;
    # without a handler Tk exits immediately, skipping WM_DELETE_WINDOW.
    app, dialogs = make_app(monkeypatch)
    destroyed = track_destroy(app)
    try:
        run_once(app)
        dialogs.answers["askyesnocancel"] = None  # Cancel
        app.tk.call("::tk::mac::Quit")
        assert "askyesnocancel" in dialogs.names()
        assert destroyed["n"] == 0

        dialogs.answers["askyesnocancel"] = False  # Discard
        app.tk.call("::tk::mac::Quit")
        assert destroyed["n"] == 1
    finally:
        tk.Tk.destroy(app)


# =========================================================================
# Relabel panel busy flag
# =========================================================================

def test_relabel_rejected_on_meta_row_does_not_disable_panel(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        app.sheet.select_cell(vis_row(app, "MatrixLang"), 2)
        app.paste_to_label("MIXED")  # locked: MatrixLang only accepts TR/EN
        app.update()
        assert row(app, "MatrixLang")['label'] == "TR"
        assert app._relabel_busy is False

        app.sheet.select_cell(vis_row(app, "kitap"), 2)
        app.paste_to_label("EN")
        app.update()
        assert row(app, "kitap")['label'] == "EN"
    finally:
        app.destroy()


def test_relabel_busy_flag_cleared_after_exception(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)

        def _boom(*a, **k):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(app, "_ensure_valid_selection", _boom)
        with pytest.raises(RuntimeError):
            app.paste_to_label("EN")
        app.update()
        assert app._relabel_busy is False
    finally:
        app.destroy()


def test_relabel_with_no_rows_does_not_disable_panel(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        app.paste_to_label("EN")  # empty grid: nothing to relabel
        app.update()
        assert app._relabel_busy is False
    finally:
        app.destroy()


# =========================================================================
# Run over existing annotations
# =========================================================================

def _make_manual_edits(app):
    app._add_new_column("Notes")
    r = row(app, "kitap")
    r['label'] = "EN"
    r['gloss'] = "book"
    r['Notes'] = "checked"
    r['reviewed'] = True
    app._sync_active_dataset_from_live()


def test_run_over_existing_annotations_cancel_preserves_everything(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    try:
        run_once(app)
        _make_manual_edits(app)
        app._mark_clean()
        before_blocks = copy.deepcopy(app.blocks)
        before_ds = copy.deepcopy(app.datasets)
        try:
            before_grid = app.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            before_grid = app.sheet.get_sheet_data()

        dialogs.answers["askokcancel"] = False
        app.txt_input.insert("end", " extra")
        app.update()
        app._mark_clean()
        app.run_pipeline()
        app.update()

        assert "askokcancel" in dialogs.names()
        assert app.blocks == before_blocks
        assert app.datasets[0]['blocks'] == before_ds[0]['blocks']
        try:
            after_grid = app.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            after_grid = app.sheet.get_sheet_data()
        assert after_grid == before_grid
        assert app._dirty is False
    finally:
        app.destroy()


def test_run_over_existing_annotations_confirm_replaces(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    try:
        run_once(app)
        _make_manual_edits(app)
        dialogs.answers["askokcancel"] = True
        app.run_pipeline()
        app.update()
        assert row(app, "kitap")['label'] == "TR"
        assert row(app, "kitap").get('gloss', '') == ""
    finally:
        app.destroy()


def test_first_run_on_empty_dataset_does_not_ask(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    try:
        run_once(app)
        assert "askokcancel" not in dialogs.names()
        assert row(app, "kitap")['label'] == "TR"
    finally:
        app.destroy()


# =========================================================================
# Label validation
# =========================================================================

def _end_edit(app, vis, col, value):
    app.sheet.set_cell_data(vis, col, value)
    app._on_sheet_end_edit(types.SimpleNamespace(row=vis, column=col))
    return app.sheet.get_cell_data(vis, col)


def test_typed_invalid_label_is_rejected_and_reverted(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        v = vis_row(app, "boss'um")
        shown = _end_edit(app, v, 2, "mixd")
        assert row(app, "boss'um")['label'] == "MIXED"
        assert shown == "MIXED"
    finally:
        app.destroy()


def test_typed_label_is_canonicalized(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        v = vis_row(app, "kitap")
        shown = _end_edit(app, v, 2, " lang3 ")
        assert row(app, "kitap")['label'] == "LANG3"
        assert shown == "LANG3"
    finally:
        app.destroy()


def test_typed_empty_label_is_allowed_like_clear(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        _end_edit(app, vis_row(app, "kitap"), 2, "")
        assert row(app, "kitap")['label'] == ""
    finally:
        app.destroy()


def test_meta_rows_keep_their_special_values(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        # EmbedLang "-" produced by the pipeline survives an unrelated edit and export.
        assert next(r for b in app.blocks for r in b if r.get('token') == "EmbedLang"
                    and r.get('label') == "-")
        # SentenceID's Label cell holds the sentence id, not a schema label.
        sid_vis = next(v for v, (b, r) in app._row_index_map.items()
                       if b is not None and app.blocks[b][r].get('token') == "SentenceID")
        _end_edit(app, sid_vis, 2, "7")
        assert app.blocks[app._row_index_map[sid_vis][0]][app._row_index_map[sid_vis][1]]['label'] == "7"
        # MatrixLang keeps its existing TR/EN-only lock.
        _end_edit(app, vis_row(app, "MatrixLang"), 2, "MIXED")
        assert row(app, "MatrixLang")['label'] == "TR"
        _end_edit(app, vis_row(app, "MatrixLang"), 2, "EN")
        assert row(app, "MatrixLang")['label'] == "EN"
    finally:
        app.destroy()


def test_paste_invalid_label_is_skipped(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        v = vis_row(app, "kitap")
        app._grid_clipboard = {"rows": 1, "cols": 2, "data": [["garbage", "a-gloss"]]}
        app.sheet.select_cell(v, 2)
        app.update()
        app.paste_selected_cells()
        assert row(app, "kitap")['label'] == "TR"
        assert row(app, "kitap")['gloss'] == "a-gloss"

        app._grid_clipboard = {"rows": 1, "cols": 1, "data": [["en"]]}
        app.sheet.select_cell(v, 2)
        app.update()
        app.paste_selected_cells()
        assert row(app, "kitap")['label'] == "EN"
        assert app.sheet.get_cell_data(v, 2) == "EN"
    finally:
        app.destroy()


def test_invalid_label_never_reaches_export(monkeypatch, tmp_path):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        _end_edit(app, vis_row(app, "boss'um"), 2, "mixd")
        app._sync_active_dataset_from_live()
        for fmt in ("TXT", "CSV", "CoNLL", "JSONL"):
            p = tmp_path / f"out.{fmt}"
            app._write_dataset_export(app.datasets[0], fmt, str(p))
            assert "mixd" not in p.read_text(encoding="utf-8")
    finally:
        app.destroy()


# =========================================================================
# Toolbar checkboxes follow the loaded configuration
# =========================================================================

def _write_project(path, cfg):
    payload = {
        "version": 2,
        "cfg": cfg,
        "datasets": [{"name": "Data 1", "blocks": []}],
        "active_dataset_index": 0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_open_project_syncs_toolbar_checkboxes(monkeypatch, tmp_path):
    path = tmp_path / "p.trenproj"
    _write_project(path, {"NER_ENABLED": False, "FEATURE_LANGUAGE_PER_ITEM": True,
                          "FEATURE_MATRIX_LANGUAGE": False, "FEATURE_EMBEDDED_LANGUAGE": True})
    app, _ = make_app(monkeypatch)
    try:
        assert app.v_ner.get() is True
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        assert app.cfg["NER_ENABLED"] is False
        assert app.v_ner.get() is False
        assert app.v_mlx.get() is False
        assert app.v_lang.get() is True
        assert app.v_emb.get() is True
        assert app._dirty is False
    finally:
        app.destroy()


def test_auto_restore_syncs_toolbar_checkboxes(monkeypatch, tmp_path):
    path = tmp_path / "p.trenproj"
    _write_project(path, {"NER_ENABLED": False})
    os.makedirs(caa.APP_DIR, exist_ok=True)
    with open(caa.LAST_PROJECT_PTR, "w", encoding="utf-8") as f:
        json.dump({"path": str(path)}, f)
    app, _ = make_app(monkeypatch)
    try:
        app._auto_restore_last_project()
        app.update()
        assert app.cfg["NER_ENABLED"] is False
        assert app.v_ner.get() is False
        # Keys missing from an older project's cfg show the pipeline default.
        assert app.v_mlx.get() is caa.DEFAULTS["FEATURE_MATRIX_LANGUAGE"]
    finally:
        app.destroy()


def test_new_project_resets_toolbar_checkboxes(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        app.v_ner.set(False)
        app._toggle("NER_ENABLED", False)
        app._mark_clean()
        app.new_project()
        app.update()
        assert app.cfg["NER_ENABLED"] is True
        assert app.v_ner.get() is True
    finally:
        app.destroy()


# =========================================================================
# Open Input read errors; menu shortcut labels
# =========================================================================

def test_open_input_non_utf8_file_shows_error_and_changes_nothing(monkeypatch, tmp_path):
    bad = tmp_path / "latin.txt"
    bad.write_bytes("çalışma".encode("cp1254"))
    app, dialogs = make_app(monkeypatch)
    try:
        app._set_txt_input_text("existing text")
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(bad))
        app.open_input()
        assert "showerror" in dialogs.names()
        assert "UTF-8" in dialogs.calls[-1][1][1]
        assert app.txt_input.get("1.0", "end-1c") == "existing text"
        assert app._dirty is False
    finally:
        app.destroy()


def test_open_input_unreadable_file_shows_error(monkeypatch, tmp_path):
    app, dialogs = make_app(monkeypatch)
    try:
        monkeypatch.setattr(caa.filedialog, "askopenfilename",
                            lambda **k: str(tmp_path / "missing.txt"))
        app.open_input()
        assert "showerror" in dialogs.names()
    finally:
        app.destroy()


def test_open_input_valid_file_loads(monkeypatch, tmp_path):
    good = tmp_path / "in.txt"
    good.write_text("kitap amazing", encoding="utf-8")
    app, dialogs = make_app(monkeypatch)
    try:
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(good))
        app.open_input()
        assert app.txt_input.get("1.0", "end-1c") == "kitap amazing"
        assert "showerror" not in dialogs.names()
    finally:
        app.destroy()


def test_file_menu_does_not_advertise_unbound_shortcuts(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        labels = menu_labels(file_menu(app))
        assert labels == ["Open Input...", "Run", "Export Table...", "Exit"]
        assert not any("⌘" in lab or "Ctrl" in lab for lab in labels)
    finally:
        app.destroy()


# =========================================================================
# Reranker status indicator
# =========================================================================

def test_reranker_status_shows_unavailable_with_reason(monkeypatch):
    app, dialogs = make_app(monkeypatch)
    try:
        monkeypatch.setattr(reranking, "last_load_failure", "model cannot run with the installed scikit-learn 1.7.2")
        app._reranker_bundle = None
        app._update_reranker_status()
        assert "unavailable" in app._reranker_status_var.get()
        assert "scikit-learn 1.7.2" in app._reranker_status_reason
        assert dialogs.calls == []  # indicator only; the fallback contract forbids a dialog
    finally:
        app.destroy()


def test_reranker_status_shows_active(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        app._reranker_bundle = reranking.ReRankerBundle(
            model=None, tfidf=None, dictvec=None, threshold=0.85, metadata={})
        app._update_reranker_status()
        assert app._reranker_status_var.get() == "MIXED reranker: active"
    finally:
        app.destroy()


def test_reranker_status_updated_when_bundle_loads(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        app.annotator = object()
        monkeypatch.setattr(reranking, "load_reranker_bundle", lambda: None)
        monkeypatch.setattr(reranking, "last_load_failure", "missing file")
        assert app._reranker_status_var.get() == ""
        app._ensure_annotator_ready()
        assert "unavailable" in app._reranker_status_var.get()
        assert app._reranker_status_reason == "missing file"
    finally:
        app.destroy()


# =========================================================================
# Plain mouse click selects a grid cell
# =========================================================================

def _click_cell(sheet, r, c):
    mt = sheet.MT
    mt.update()
    assert mt.winfo_viewable(), "table must be mapped before clicking"
    sheet.see(r, c)
    mt.update()
    rp, cp = mt.row_positions, mt.col_positions
    x = int((cp[c] + cp[c + 1]) // 2 - mt.canvasx(0))
    y = int((rp[r] + rp[r + 1]) // 2 - mt.canvasy(0))
    assert 0 <= x < mt.winfo_width() and 0 <= y < mt.winfo_height(), (
        f"cell ({r}, {c}) is not visible in the {mt.winfo_width()}x{mt.winfo_height()} table")
    mt.event_generate('<ButtonPress-1>', x=x, y=y)
    mt.event_generate('<ButtonRelease-1>', x=x, y=y)
    mt.update()


def test_plain_click_selects_cell_in_main_grid(monkeypatch):
    # Regression: sheet.bind("<Button-1>") replaced tksheet's own click
    # handler, so clicking a cell never changed the selection.
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        app._active_area = "text"
        target = vis_row(app, "amazing")
        _click_cell(app.sheet, target, 2)
        cur = app.sheet.get_currently_selected()
        assert (cur.row, cur.column) == (target, 2)
        assert app._active_area == "sheet"
    finally:
        app.destroy()


def test_plain_click_selects_cell_in_full_edit_window(monkeypatch):
    app, _ = make_app(monkeypatch)
    try:
        run_once(app)
        app.open_full_edit_window()
        app.update()
        full = app._full_sheet
        app._active_sheet = None
        target = vis_row(app, "boss'um")
        _click_cell(full, target, 3)
        cur = full.get_currently_selected()
        assert (cur.row, cur.column) == (target, 3)
        assert app._active_sheet is full
    finally:
        app.destroy()
