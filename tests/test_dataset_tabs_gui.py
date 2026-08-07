# tests/test_dataset_tabs_gui.py
"""
Permanent GUI regression tests for multiple annotation datasets (dataset
tab bar, Add New Data dialog, dataset-scoped state) and the Export Table
dialog (dataset + format chooser, including CoNLL/JSONL).

Follows the same real-Tk-event conventions as test_uid_review_tool_gui.py.
Requires a real, working Tk display; the whole module is skipped otherwise
so the rest of the suite still runs (e.g. a CI runner with no X server).
"""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk  # noqa: E402

import cs_annotator_app as caa  # noqa: E402
import annotation_model  # noqa: E402


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
    """Every test in this module gets its own fake ~/.cs_annotator. Without
    this, App.__init__'s scheduled _auto_restore_last_project (or an
    explicit open_project_save/save_project_progress call in one test) could
    read or write the developer's REAL project-save directory, and a project
    saved by one test could silently leak into and pollute a later test's
    fresh App() instance via that real pointer file."""
    app_dir = tmp_path / "cs_annotator_home"
    monkeypatch.setattr(caa, "APP_DIR", str(app_dir))
    monkeypatch.setattr(caa, "LAST_PROJECT_PTR", str(app_dir / "last_project.json"))


# --- shared helpers -------------------------------------------------------

def real_click(widget, xy=None):
    if xy is None:
        w = widget.winfo_width() or 20
        h = widget.winfo_height() or 20
        xy = (w // 2, h // 2)
    x, y = xy
    widget.event_generate('<Enter>', x=x, y=y)
    widget.event_generate('<ButtonPress-1>', x=x, y=y)
    widget.event_generate('<ButtonRelease-1>', x=x, y=y)


def real_type(entry_widget, text):
    entry_widget.focus_force()
    entry_widget.update()
    for ch in text:
        entry_widget.event_generate('<space>' if ch == ' ' else f'<KeyPress-{ch}>')
    entry_widget.update()


def real_combobox_select(combo_widget, value):
    values = list(combo_widget.cget('values'))
    idx = values.index(value)
    combo_widget.focus_force()
    combo_widget.update()
    combo_widget.event_generate('<Button-1>')
    combo_widget.update()
    listbox_path = combo_widget._w + '.popdown.f.l'
    bbox = combo_widget.tk.call(listbox_path, 'bbox', idx)
    x, y, w, h = [int(v) for v in bbox]
    cx, cy = x + w // 2, y + h // 2
    combo_widget.tk.call('event', 'generate', listbox_path, '<Motion>', '-x', cx, '-y', cy)
    combo_widget.tk.call('event', 'generate', listbox_path, '<ButtonPress-1>', '-x', cx, '-y', cy)
    combo_widget.tk.call('event', 'generate', listbox_path, '<ButtonRelease-1>', '-x', cx, '-y', cy)
    combo_widget.update()


def find_buttons_by_text(root_widget):
    out = {}

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, caa.ttk.Button):
                out[c.cget('text')] = c
            walk(c)

    walk(root_widget)
    return out


def find_widgets(root_widget):
    out = []

    def walk(w):
        out.append(w)
        for c in w.winfo_children():
            walk(c)

    walk(root_widget)
    return out


def find_toplevel(app, title):
    return next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel) and w.title() == title)


def make_app(monkeypatch):
    """A real App() with dialogs neutered so validation-failure paths
    (blank name, empty text, empty export) don't block on a real modal
    messagebox during the test run."""
    app = caa.App()
    monkeypatch.setattr(caa.messagebox, "showwarning", lambda *a, **k: None)
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
    monkeypatch.setattr(caa.messagebox, "showinfo", lambda *a, **k: None)
    app.update()
    return app


def stub_pipeline(app, monkeypatch, *, fail=False):
    """Replace the (slow, model-loading) real annotation pipeline with a
    deterministic stub so dataset-tab tests exercise the tab/dataset wiring,
    not cs_pipeline.Annotator itself (which has its own extensive test
    coverage in test_cs_pipeline.py)."""
    calls = []

    def _fake(text):
        calls.append(text)
        if fail:
            raise RuntimeError("stubbed annotation failure")
        lines = [f"{tok}\tTR" for tok in text.split()]
        return "\n".join(lines)

    monkeypatch.setattr(app, "_run_annotation_pipeline", _fake)
    return calls


def tab_button_texts(app):
    frame = app._dataset_tabs_frame
    return [c.cget('text') for c in frame.winfo_children() if isinstance(c, caa.ttk.Button)]


# =========================================================================
# Dataset tabs: initial state, Add New Data, independence
# =========================================================================

def test_initial_dataset_is_data_1_with_one_tab():
    app = caa.App()
    try:
        app.update()
        assert len(app.datasets) == 1
        assert app.datasets[0]['name'] == "Data 1"
        assert app._active_dataset_index == 0
        texts = tab_button_texts(app)
        assert "Data 1" in texts
        assert "+" in texts
    finally:
        app.destroy()


def test_plus_button_opens_add_new_data_dialog(monkeypatch):
    app = make_app(monkeypatch)
    try:
        buttons = find_buttons_by_text(app._dataset_tabs_frame)
        real_click(buttons["+"])
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        assert dlg is not None
        dlg.destroy()
    finally:
        app.destroy()


def test_enter_new_text_creates_independent_second_dataset(monkeypatch):
    app = make_app(monkeypatch)
    calls = stub_pipeline(app, monkeypatch)
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        entries = [w for w in widgets if isinstance(w, caa.ttk.Entry)]
        text_boxes = [w for w in widgets if isinstance(w, caa.ScrolledText)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}

        name_entry = entries[0]
        name_entry.delete(0, 'end')
        real_type(name_entry, "Data 2")

        text_box = text_boxes[0]
        text_box.insert("1.0", "hello world")

        real_click(buttons["Create"])
        app.update()

        assert calls == ["hello world"]
        assert len(app.datasets) == 2
        assert app._active_dataset_index == 1
        assert app.datasets[1]['name'] == "Data 2"
        assert app.datasets[1]['source_text'] == "hello world"
        tokens = [r['token'] for r in app.blocks[0] if not annotation_model.is_meta_row_token(r['token'])]
        assert tokens == ['hello', 'world']
        # the first dataset must be untouched
        assert app.datasets[0]['name'] == "Data 1"
        assert app.datasets[0]['source_text'] == ""
    finally:
        app.destroy()


def test_rerun_current_text_creates_new_dataset_without_changing_first(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "orig text")
        app.run_pipeline()
        app.update()
        original_blocks_snapshot = copy.deepcopy(app.blocks)

        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        radios = [w for w in widgets if isinstance(w, caa.ttk.Radiobutton)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        rerun_radio = next(w for w in radios if w.cget('text') == "Re-run Current Text")
        real_click(rerun_radio)
        app.update()

        real_click(buttons["Create"])
        app.update()

        assert len(app.datasets) == 2
        assert app.datasets[1]['source_text'] == "orig text"
        assert app.datasets[0]['source_text'] == "orig text"
        assert app.datasets[0]['blocks'] == original_blocks_snapshot
        # independent block objects, not aliased
        assert app.datasets[0]['blocks'] is not app.datasets[1]['blocks']
    finally:
        app.destroy()


def test_cancel_add_new_data_creates_no_dataset(monkeypatch):
    app = make_app(monkeypatch)
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        buttons = {w.cget('text'): w for w in find_widgets(dlg) if isinstance(w, caa.ttk.Button)}
        real_click(buttons["Cancel"])
        app.update()
        assert len(app.datasets) == 1
    finally:
        app.destroy()


def test_annotation_failure_creates_no_dataset(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch, fail=True)
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        text_boxes = [w for w in widgets if isinstance(w, caa.ScrolledText)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        text_boxes[0].insert("1.0", "some text")

        real_click(buttons["Create"])
        app.update()

        assert len(app.datasets) == 1
        # the dialog must still be open (no broken/empty tab created, and no crash)
        assert dlg.winfo_exists()
        dlg.destroy()
    finally:
        app.destroy()


def test_blank_name_does_not_create_dataset(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        entries = [w for w in widgets if isinstance(w, caa.ttk.Entry)]
        text_boxes = [w for w in widgets if isinstance(w, caa.ScrolledText)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}

        # Clear the name field via the widget API rather than a simulated
        # mouse drag-select: dragging over an Entry's text depends on the
        # dialog already having real, mapped geometry, which is timing-
        # sensitive for a Toplevel that was just created this instant (it
        # was observed to occasionally miss and leave the prefilled default
        # name in place, which is not what this test is about -- it is
        # testing the blank-name validation path, not text-drag-selection).
        entries[0].delete(0, 'end')
        text_boxes[0].insert("1.0", "some text")
        real_click(buttons["Create"])
        app.update()

        assert len(app.datasets) == 1
        dlg.destroy()
    finally:
        app.destroy()


# =========================================================================
# Add New Data: "Open New File" mode
# =========================================================================

def _find_radio(widgets, text):
    return next(w for w in widgets if isinstance(w, caa.ttk.Radiobutton) and w.cget('text') == text)


def _find_file_display_label(widgets):
    return next(w for w in widgets
                if isinstance(w, caa.ttk.Label) and w.cget('text') == '(none selected)')


def _open_file_mode(app):
    """Open the Add New Data dialog and select "Open New File" mode. Returns
    (dlg, widgets, buttons, file_display_label)."""
    app._open_add_new_data_dialog()
    app.update()
    dlg = find_toplevel(app, "Add New Data")
    widgets = find_widgets(dlg)
    file_display_label = _find_file_display_label(widgets)
    real_click(_find_radio(widgets, "Open New File"))
    app.update()
    buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
    return dlg, widgets, buttons, file_display_label


def _write_utf8_file(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_open_new_file_option_is_visible(monkeypatch):
    app = make_app(monkeypatch)
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        radio_texts = {w.cget('text') for w in widgets if isinstance(w, caa.ttk.Radiobutton)}
        assert radio_texts == {"Open New File", "Enter New Text", "Re-run Current Text"}
        dlg.destroy()
    finally:
        app.destroy()


def test_browse_opens_file_chooser(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "sample.txt", "merhaba dunya")
    app = make_app(monkeypatch)
    calls = []
    monkeypatch.setattr(caa.filedialog, "askopenfilename",
                         lambda **k: (calls.append(k) or str(path)))
    try:
        dlg, widgets, buttons, file_display_label = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        assert len(calls) == 1
        assert file_display_label.cget('text') == "sample.txt"
        dlg.destroy()
    finally:
        app.destroy()


def test_cancelling_browse_changes_nothing(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: "")  # user cancels
    monkeypatch.setattr(caa.messagebox, "showwarning", lambda *a, **k: None)
    try:
        dlg, widgets, buttons, file_display_label = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        assert file_display_label.cget('text') == "(none selected)"

        # Confirms internal state truly never changed: Create still refuses
        # for "no file chosen", not because the chosen file was invalid.
        real_click(buttons["Create"])
        app.update()
        assert len(app.datasets) == 1
        dlg.destroy()
    finally:
        app.destroy()


def test_selecting_turkish_utf8_file_becomes_source_text_and_dataset(monkeypatch, tmp_path):
    content = "Bugün meetinge katıldım. Türkçe karakterler: çğıöşü ÇĞİÖŞÜ."
    path = _write_utf8_file(tmp_path, "turkce.txt", content)
    app = make_app(monkeypatch)
    calls = stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        dlg, widgets, buttons, file_display_label = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        # Pipeline must not run just from selecting a file.
        assert calls == []

        real_click(buttons["Create"])
        app.update()

        assert calls == [content]  # pipeline ran only after Create, on the file's content
        assert len(app.datasets) == 2
        assert app.datasets[1]['source_text'] == content
        # Only the basename is ever kept -- never the full local path
        # (privacy: an absolute path can expose the user's account name and
        # local directory structure if the project is shared).
        assert app.datasets[1]['source_filename'] == "turkce.txt"
        assert str(path) != app.datasets[1]['source_filename']
        assert str(tmp_path) not in (app.datasets[1]['source_filename'] or "")
    finally:
        app.destroy()


def test_open_new_file_does_not_alter_existing_dataset(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "sample.txt", "yeni veri")
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        app.txt_input.insert("1.0", "original active dataset text")
        app.update()
        before_blocks = copy.deepcopy(app.datasets[0]['blocks'])  # [] -- never annotated

        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()

        # The active dataset's blocks are untouched by adding a new dataset
        # (Add New Data never runs the pipeline against dataset 0); its
        # source_text correctly reflects the editor at the moment of Create
        # (the same "sync before switching away" behavior tab-switching
        # already relies on), not some earlier snapshot.
        assert app.datasets[0]['source_text'] == "original active dataset text"
        assert app.datasets[0]['blocks'] == before_blocks
        assert app._active_dataset_index == 1  # switched to the NEW dataset, not overwritten
    finally:
        app.destroy()


def test_open_new_file_create_without_selecting_file_warns(monkeypatch):
    app = make_app(monkeypatch)
    warnings = []
    monkeypatch.setattr(caa.messagebox, "showwarning", lambda *a, **k: warnings.append(a))
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Create"])
        app.update()
        assert warnings
        assert len(app.datasets) == 1
        dlg.destroy()
    finally:
        app.destroy()


def test_open_new_file_empty_file_rejected(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "empty.txt", "   \n  ")
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    warnings = []
    monkeypatch.setattr(caa.messagebox, "showwarning", lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()
        assert warnings
        assert len(app.datasets) == 1
    finally:
        app.destroy()


def test_open_new_file_unreadable_file_shows_error(monkeypatch, tmp_path):
    missing_path = tmp_path / "does_not_exist.txt"  # never actually created
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    errors = []
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(missing_path))
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()
        assert errors
        assert len(app.datasets) == 1
    finally:
        app.destroy()


def test_open_new_file_invalid_utf8_shows_error(monkeypatch, tmp_path):
    path = tmp_path / "bad_encoding.txt"
    with open(path, "wb") as f:
        f.write(b"valid start \xff\xfe invalid bytes")
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    errors = []
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()
        assert errors
        assert len(app.datasets) == 1
    finally:
        app.destroy()


def test_open_new_file_annotation_failure_creates_no_tab(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "sample.txt", "will fail")
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch, fail=True)
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()
        assert len(app.datasets) == 1
    finally:
        app.destroy()


def test_open_new_file_mode_does_not_use_leftover_text_box_content(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "sample.txt", "file content wins")
    app = make_app(monkeypatch)
    calls = stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        text_boxes = [w for w in widgets if isinstance(w, caa.ScrolledText)]
        # Type something while still in the default "Enter New Text" mode...
        text_boxes[0].insert("1.0", "leftover text box content")

        # ...then switch to "Open New File" and select a real file.
        real_click(_find_radio(widgets, "Open New File"))
        app.update()
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()

        assert calls == ["file content wins"]  # not the leftover text box content
        assert app.datasets[1]['source_text'] == "file content wins"
    finally:
        app.destroy()


def test_mode_switching_preserves_text_and_selected_file(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "sample.txt", "kept across switches")
    app = make_app(monkeypatch)
    calls = stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        app._open_add_new_data_dialog()
        app.update()
        dlg = find_toplevel(app, "Add New Data")
        widgets = find_widgets(dlg)
        text_boxes = [w for w in widgets if isinstance(w, caa.ScrolledText)]
        file_display_label = _find_file_display_label(widgets)
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}

        text_boxes[0].insert("1.0", "my typed text")

        real_click(_find_radio(widgets, "Open New File"))
        app.update()
        real_click(buttons["Browse…"])
        app.update()
        assert file_display_label.cget('text') == "sample.txt"

        # Back to Enter New Text -- the typed text must still be there.
        real_click(_find_radio(widgets, "Enter New Text"))
        app.update()
        assert text_boxes[0].get("1.0", "end-1c") == "my typed text"

        # And the selected file must still be remembered too.
        real_click(_find_radio(widgets, "Open New File"))
        app.update()
        assert file_display_label.cget('text') == "sample.txt"

        real_click(buttons["Create"])
        app.update()
        assert calls == ["kept across switches"]  # file mode was the one actually used
    finally:
        app.destroy()


def test_open_new_file_save_reopen_retains_dataset_without_external_file(monkeypatch, tmp_path):
    content = "saved from a file that will disappear"
    src_path = _write_utf8_file(tmp_path, "will_vanish.txt", content)
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(src_path))
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj_file")
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()

        assert app.save_project_progress() is True

        os.remove(src_path)  # the original file no longer exists
        assert not os.path.exists(src_path)

        app2 = caa.App()
        monkeypatch.setattr(caa.messagebox, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
        try:
            saved_path = os.path.join(caa.APP_DIR, "proj_file" + caa.PROJECT_EXT)
            monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: saved_path)
            app2.open_project_save()  # must not raise even though src_path is gone
            app2.update()
            assert len(app2.datasets) == 2
            assert app2.datasets[1]['source_text'] == content
        finally:
            app2.destroy()
    finally:
        app.destroy()


def test_open_new_file_saved_project_never_contains_full_local_path(monkeypatch, tmp_path):
    # End-to-end privacy regression: a real Open New File -> Create ->
    # Save Project Progress flow must never write the file's full local
    # path (which would expose the user's account name and directory
    # structure if the .trenproj were shared) into the saved file --
    # only the basename, and only inside "source_filename".
    src_dir = tmp_path / "private-folder"
    src_dir.mkdir()
    src_path = _write_utf8_file(src_dir, "corpus.txt", "hassas icerik")
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(src_path))
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj_privacy")
    try:
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()

        assert app.save_project_progress() is True

        saved_path = os.path.join(caa.APP_DIR, "proj_privacy" + caa.PROJECT_EXT)
        with open(saved_path, encoding="utf-8") as f:
            raw = f.read()

        assert "corpus.txt" in raw
        assert str(src_dir) not in raw
        assert "private-folder" not in raw
        assert str(src_path) not in raw

        payload = json.loads(raw)
        assert payload["datasets"][1]["source_filename"] == "corpus.txt"
        assert "source_path" not in payload["datasets"][1]
    finally:
        app.destroy()


def test_open_new_file_successful_creation_marks_dirty(monkeypatch, tmp_path):
    path = _write_utf8_file(tmp_path, "sample.txt", "dirty me")
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
    try:
        assert app._dirty is False
        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        assert app._dirty is False  # selecting a file alone must not mark dirty

        real_click(buttons["Create"])
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


@pytest.mark.parametrize("scenario", ["cancel_browse", "empty_file", "bad_utf8", "annotation_failure"])
def test_open_new_file_failure_paths_preserve_prior_dirty_state(monkeypatch, tmp_path, scenario):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "showwarning", lambda *a, **k: None)
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)

    if scenario == "cancel_browse":
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: "")
        stub_pipeline(app, monkeypatch)
    elif scenario == "empty_file":
        path = _write_utf8_file(tmp_path, "empty.txt", "")
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        stub_pipeline(app, monkeypatch)
    elif scenario == "bad_utf8":
        path = tmp_path / "bad.txt"
        with open(path, "wb") as f:
            f.write(b"\xff\xfe\x00bad")
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        stub_pipeline(app, monkeypatch)
    else:
        path = _write_utf8_file(tmp_path, "sample.txt", "will fail annotation")
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        stub_pipeline(app, monkeypatch, fail=True)

    try:
        # Start already dirty (from an unrelated edit) to prove failures
        # preserve whatever the prior dirty state was, not just "stays clean".
        app.txt_input.insert("1.0", "unrelated pre-existing edit")
        app.update()
        assert app._dirty is True

        dlg, widgets, buttons, _ = _open_file_mode(app)
        real_click(buttons["Browse…"])
        app.update()
        real_click(buttons["Create"])
        app.update()

        assert len(app.datasets) == 1
        assert app._dirty is True  # unchanged, not reset and not re-marked in a new way
    finally:
        app.destroy()


def test_editing_dataset_2_does_not_mutate_dataset_1(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()

        new_ds = app._create_dataset_from_text("Data 2", "gamma delta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()

        # mutate the now-active dataset 2 directly through the live model
        app.blocks[0][0]['label'] = 'EN'
        app._sync_active_dataset_from_live()

        assert app.datasets[1]['blocks'][0][0]['label'] == 'EN'
        assert app.datasets[0]['blocks'][0][0]['label'] == 'TR'
    finally:
        app.destroy()


def test_switching_tabs_restores_correct_blocks_and_text(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()

        new_ds = app._create_dataset_from_text("Data 2", "gamma delta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()
        assert app.txt_input.get("1.0", "end-1c") == "gamma delta"

        app._switch_dataset(0)
        app.update()
        assert app._active_dataset_index == 0
        assert app.txt_input.get("1.0", "end-1c") == "alpha beta"
        tokens = [r['token'] for r in app.blocks[0] if not annotation_model.is_meta_row_token(r['token'])]
        assert tokens == ['alpha', 'beta']

        app._switch_dataset(1)
        app.update()
        assert app.txt_input.get("1.0", "end-1c") == "gamma delta"
    finally:
        app.destroy()


def test_switching_to_same_active_tab_is_a_noop(monkeypatch):
    app = make_app(monkeypatch)
    try:
        app.txt_input.insert("1.0", "unsaved edit")
        app._switch_dataset(0)  # already active
        app.update()
        # source text in the editor must be untouched by the no-op switch
        assert app.txt_input.get("1.0", "end-1c") == "unsaved edit"
    finally:
        app.destroy()


def test_switching_dataset_closes_uid_review_tool(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        app.blocks[0][0]['label'] = 'UID'

        app.open_uid_review_tool()
        app.update()
        assert app._uid_win is not None

        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()

        assert app._uid_win is None
    finally:
        app.destroy()


def test_active_tab_style_reflects_active_dataset(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()

        frame = app._dataset_tabs_frame
        tab_buttons = [c for c in frame.winfo_children() if isinstance(c, caa.ttk.Button)
                       and c.cget('text') in ("Data 1", "Data 2")]
        styles = {b.cget('text'): b.cget('style') for b in tab_buttons}
        assert styles["Data 2"] == "DatasetTabActive.TButton"
        assert styles["Data 1"] == "DatasetTab.TButton"
    finally:
        app.destroy()


# =========================================================================
# Export Table dialog
# =========================================================================

def test_export_dialog_lists_all_datasets_and_formats(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()

        app.open_export_dialog()
        app.update()
        dlg = find_toplevel(app, "Export Table")
        combos = [w for w in find_widgets(dlg) if isinstance(w, caa.ttk.Combobox)]
        dataset_values = list(combos[0].cget('values'))
        format_values = list(combos[1].cget('values'))
        assert dataset_values == ["Data 1", "Data 2"]
        assert format_values == ["TXT", "CSV", "CoNLL", "JSONL"]
        dlg.destroy()
    finally:
        app.destroy()


def test_export_writes_selected_non_active_dataset_only(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)  # Data 2 is now active
        app.update()

        out_path = tmp_path / "export.jsonl"
        monkeypatch.setattr(caa.filedialog, "asksaveasfilename", lambda **k: str(out_path))

        app.open_export_dialog()
        app.update()
        dlg = find_toplevel(app, "Export Table")
        widgets = find_widgets(dlg)
        combos = [w for w in widgets if isinstance(w, caa.ttk.Combobox)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}

        real_combobox_select(combos[0], "Data 1")   # not the active dataset
        real_combobox_select(combos[1], "JSONL")
        real_click(buttons["Export"])
        app.update()

        assert out_path.exists()
        line = out_path.read_text(encoding="utf-8").strip().splitlines()[0]
        record = json.loads(line)
        assert record["dataset"] == "Data 1"
        assert record["tokens"][0]["token"] == "alpha"
    finally:
        app.destroy()


def test_cancelling_export_creates_no_file(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()

        out_path = tmp_path / "should_not_exist.txt"
        monkeypatch.setattr(caa.filedialog, "asksaveasfilename", lambda **k: "")

        app.open_export_dialog()
        app.update()
        dlg = find_toplevel(app, "Export Table")
        buttons = {w.cget('text'): w for w in find_widgets(dlg) if isinstance(w, caa.ttk.Button)}
        real_click(buttons["Export"])
        app.update()

        assert not out_path.exists()
        dlg.destroy()
    finally:
        app.destroy()


def test_conll_export_content(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()

        out_path = tmp_path / "export.conll"
        monkeypatch.setattr(caa.filedialog, "asksaveasfilename", lambda **k: str(out_path))

        app.open_export_dialog()
        app.update()
        dlg = find_toplevel(app, "Export Table")
        widgets = find_widgets(dlg)
        combos = [w for w in widgets if isinstance(w, caa.ttk.Combobox)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        real_combobox_select(combos[0], "Data 1")
        real_combobox_select(combos[1], "CoNLL")
        app.update()
        real_click(buttons["Export"])
        app.update()

        text = out_path.read_text(encoding="utf-8")
        assert text.startswith("# TREN CoNLL export\n")
        assert "alpha" in text and "beta" in text
    finally:
        app.destroy()


def test_export_does_not_mutate_dataset(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()
        before = copy.deepcopy(app.blocks)

        out_path = tmp_path / "export.txt"
        monkeypatch.setattr(caa.filedialog, "asksaveasfilename", lambda **k: str(out_path))
        app.open_export_dialog()
        app.update()
        dlg = find_toplevel(app, "Export Table")
        buttons = {w.cget('text'): w for w in find_widgets(dlg) if isinstance(w, caa.ttk.Button)}
        real_click(buttons["Export"])
        app.update()

        assert app.blocks == before
    finally:
        app.destroy()


# =========================================================================
# Multi-dataset project save/reopen (APP_DIR redirected into tmp_path so
# these tests never touch the user's real ~/.cs_annotator)
# =========================================================================

def test_save_and_reopen_multi_dataset_project(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app_dir = tmp_path / "cs_annotator_home"
        ptr_path = app_dir / "last_project.json"
        monkeypatch.setattr(caa, "APP_DIR", str(app_dir))
        monkeypatch.setattr(caa, "LAST_PROJECT_PTR", str(ptr_path))

        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()

        monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "multiproj")
        app.save_project_progress()

        saved_path = app_dir / "multiproj.trenproj"
        assert saved_path.exists()
        with open(saved_path, encoding="utf-8") as f:
            payload = json.load(f)
        assert len(payload["datasets"]) == 2
        assert payload["active_dataset_index"] == 1

        # reopen into a fresh app instance
        app2 = caa.App()
        monkeypatch.setattr(caa.messagebox, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
        try:
            monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(saved_path))
            app2.open_project_save()
            app2.update()
            assert len(app2.datasets) == 2
            assert app2._active_dataset_index == 1
            assert app2.datasets[0]['name'] == "Data 1"
            assert app2.datasets[0]['source_text'] == "alpha"
            assert app2.datasets[1]['name'] == "Data 2"
            assert app2.datasets[1]['source_text'] == "beta"
        finally:
            app2.destroy()
    finally:
        app.destroy()


def test_open_legacy_single_dataset_project_becomes_data_1(monkeypatch, tmp_path):
    legacy_path = tmp_path / "legacy.trenproj"
    legacy_payload = {
        "version": 1,
        "name": "legacy",
        "input_text": "eski metin",
        "cfg": {},
        "blocks": [[{"idx": 1, "token": "eski", "label": "TR", "gloss": ""}]],
        "extra_headers": [],
    }
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f)

    app = make_app(monkeypatch)
    try:
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(legacy_path))
        app.open_project_save()
        app.update()
        assert len(app.datasets) == 1
        assert app.datasets[0]['name'] == "Data 1"
        assert app.datasets[0]['source_text'] == "eski metin"
        assert app.txt_input.get("1.0", "end-1c") == "eski metin"
    finally:
        app.destroy()


def test_open_malformed_project_shows_error_and_does_not_crash(monkeypatch, tmp_path):
    bad_path = tmp_path / "bad.trenproj"
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "datasets": [{"name": "", "blocks": []}]}, f)

    app = make_app(monkeypatch)
    errors = []
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: errors.append(a))
    try:
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(bad_path))
        app.open_project_save()
        app.update()
        assert errors  # an error dialog was shown
        assert len(app.datasets) == 1  # original state untouched
        assert app.datasets[0]['name'] == "Data 1"
    finally:
        app.destroy()


# =========================================================================
# Open Project: unsaved-changes guard, and the ordering that protects
# dirty work (read + validate the selected file BEFORE ever touching the
# current project or even asking to save it)
# =========================================================================

def _write_valid_project(path, text="new content", token="new"):
    payload = {
        "version": 2,
        "datasets": [{"name": "Data 1", "source_text": text,
                      "blocks": [[{"idx": 1, "token": token, "label": "TR", "gloss": ""}]]}],
        "active_dataset_index": 0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_open_project_clean_project_no_prompt(monkeypatch, tmp_path):
    path = tmp_path / "proj.trenproj"
    _write_valid_project(path)
    app = make_app(monkeypatch)
    ask_calls = []
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: ask_calls.append(a) or None)
    try:
        assert app._dirty is False
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        assert ask_calls == []  # clean project: proceeds without ever prompting
        assert app.datasets[0]['source_text'] == "new content"
    finally:
        app.destroy()


def test_open_project_cancelling_file_chooser_changes_nothing_even_when_dirty(monkeypatch):
    app = make_app(monkeypatch)
    ask_calls = []
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: ask_calls.append(a) or None)
    try:
        app.txt_input.insert("1.0", "keep me")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: "")  # user cancels
        app.open_project_save()
        app.update()
        assert ask_calls == []  # never even asks -- nothing to protect yet
        assert app.txt_input.get("1.0", "end-1c") == "keep me"
        assert app._dirty is True
    finally:
        app.destroy()


def test_open_project_invalid_file_changes_nothing_even_when_dirty(monkeypatch, tmp_path):
    bad_path = tmp_path / "bad.trenproj"
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "datasets": []}, f)

    app = make_app(monkeypatch)
    ask_calls = []
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: ask_calls.append(a) or None)
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
    try:
        app.txt_input.insert("1.0", "keep me too")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(bad_path))
        app.open_project_save()
        app.update()
        # Validation fails before the current project is ever put at risk,
        # so the unsaved-changes guard must never even fire.
        assert ask_calls == []
        assert app.txt_input.get("1.0", "end-1c") == "keep me too"
        assert app.datasets[0]['name'] == "Data 1"
        assert app._dirty is True
    finally:
        app.destroy()


def test_open_project_unreadable_file_changes_nothing_even_when_dirty(monkeypatch, tmp_path):
    bad_path = tmp_path / "not_json.trenproj"
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("this is not valid JSON {{{")

    app = make_app(monkeypatch)
    ask_calls = []
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: ask_calls.append(a) or None)
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
    try:
        app.txt_input.insert("1.0", "keep me three")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(bad_path))
        app.open_project_save()
        app.update()
        assert ask_calls == []
        assert app.txt_input.get("1.0", "end-1c") == "keep me three"
        assert app._dirty is True
    finally:
        app.destroy()


def test_open_project_cancel_guard_aborts_open(monkeypatch, tmp_path):
    path = tmp_path / "proj.trenproj"
    _write_valid_project(path)
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: None)  # Cancel
    try:
        app.txt_input.insert("1.0", "original content")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        # Cancel: the current project must be completely untouched, and the
        # file that WAS successfully validated must not have replaced it.
        assert app.txt_input.get("1.0", "end-1c") == "original content"
        assert app.datasets[0]['name'] == "Data 1"
        assert app._dirty is True
    finally:
        app.destroy()


def test_open_project_save_cancelled_during_open_aborts_open(monkeypatch, tmp_path):
    path = tmp_path / "proj.trenproj"
    _write_valid_project(path)
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # Save
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: None)  # cancels name dialog
    try:
        app.txt_input.insert("1.0", "must not be replaced")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        assert app.txt_input.get("1.0", "end-1c") == "must not be replaced"
        assert app.datasets[0]['name'] == "Data 1"
        assert app._dirty is True
    finally:
        app.destroy()


def test_open_project_discard_proceeds_with_open(monkeypatch, tmp_path):
    path = tmp_path / "proj.trenproj"
    _write_valid_project(path, text="incoming", token="incoming")
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: False)  # Discard
    try:
        app.txt_input.insert("1.0", "thrown away")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        assert app.txt_input.get("1.0", "end-1c") == "incoming"
        assert app.datasets[0]['source_text'] == "incoming"
        assert app._dirty is False
    finally:
        app.destroy()


def test_open_project_save_success_then_proceeds_with_open(monkeypatch, tmp_path):
    path = tmp_path / "proj.trenproj"
    _write_valid_project(path, text="incoming2", token="incoming2")
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # Save
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj_before_open")
    try:
        app.txt_input.insert("1.0", "to be saved first")
        app.update()
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        saved_path = os.path.join(caa.APP_DIR, "proj_before_open" + caa.PROJECT_EXT)
        assert os.path.isfile(saved_path)  # the prior project really was saved
        assert app.txt_input.get("1.0", "end-1c") == "incoming2"  # then the open proceeded
        assert app.datasets[0]['source_text'] == "incoming2"
        assert app._dirty is False
    finally:
        app.destroy()


# =========================================================================
# Project dirty-state tracking
# =========================================================================

def test_new_app_starts_clean():
    app = caa.App()
    try:
        app.update()
        assert app._dirty is False
        assert app._has_unsaved_progress() is False
    finally:
        app.destroy()


def test_typing_in_input_editor_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    try:
        assert app._dirty is False
        app.txt_input.insert("1.0", "hello")
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_running_annotation_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.update()
        app._mark_clean()  # isolate the effect of run_pipeline itself
        app.run_pipeline()
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_manual_table_edit_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        app._mark_clean()

        app.blocks[0][0]['label'] = 'EN'
        vis_r = next(r for r, (b, ri) in app._row_index_map.items() if b == 0 and ri == 0)
        event = type('E', (), {'row': vis_r, 'column': 2})()
        app._on_sheet_end_edit(event, sheet_obj=app.sheet)
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_uid_apply_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        app.blocks[0][0]['label'] = 'UID'
        app._mark_clean()

        app.open_uid_review_tool()
        app.update()
        app._uid_label_var.set('TR')
        app._uid_gloss_var.set('')
        app._uid_apply()
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_uid_undo_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        app.blocks[0][0]['label'] = 'UID'

        app.open_uid_review_tool()
        app.update()
        app._uid_label_var.set('TR')
        app._uid_apply()
        app.update()
        app._mark_clean()

        app._uid_undo_last()
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_merge_cells_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()
        app._mark_clean()

        app.sheet.deselect()
        app.sheet.create_selection_box(0, 2, 2, 3)
        app.merge_selected_cells()
        app.update()
        dlg = find_toplevel(app, "Merge Cells")
        widgets = find_widgets(dlg)
        combos = [w for w in widgets if isinstance(w, caa.ttk.Combobox)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        real_combobox_select(combos[0], 'TR')
        real_click(buttons["Confirm"])
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_undo_merge_cells_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()

        app.sheet.deselect()
        app.sheet.create_selection_box(0, 2, 2, 3)
        app.merge_selected_cells()
        app.update()
        dlg = find_toplevel(app, "Merge Cells")
        widgets = find_widgets(dlg)
        combos = [w for w in widgets if isinstance(w, caa.ttk.Combobox)]
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        real_combobox_select(combos[0], 'TR')
        real_click(buttons["Confirm"])
        app.update()
        app._mark_clean()

        app.undo_merge_cells()
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_adding_dataset_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app._mark_clean()
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._mark_dirty()  # mirrors what the real Add New Data confirm handler does
        app._load_dataset_into_live(1)
        app.update()
        assert app._dirty is True
    finally:
        app.destroy()


def test_editing_any_dataset_marks_whole_project_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()
        app._mark_clean()

        app.blocks[0][0]['label'] = 'EN'
        vis_r = next(r for r, (b, ri) in app._row_index_map.items() if b == 0 and ri == 0)
        event = type('E', (), {'row': vis_r, 'column': 2})()
        app._on_sheet_end_edit(event, sheet_obj=app.sheet)
        app.update()
        assert app._dirty is True  # project-wide flag, not per-dataset
    finally:
        app.destroy()


def test_switching_dataset_tabs_does_not_mark_dirty(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app.datasets.append(new_ds)
        app.update()
        app._mark_clean()

        app._switch_dataset(1)
        app.update()
        assert app._dirty is False
        app._switch_dataset(0)
        app.update()
        assert app._dirty is False
    finally:
        app.destroy()


def test_toggle_config_change_marks_dirty(monkeypatch):
    app = make_app(monkeypatch)
    try:
        assert app._dirty is False
        before = app.cfg.get('NER_ENABLED', False)
        app._toggle('NER_ENABLED', not before)
        assert app._dirty is True
        assert app.cfg['NER_ENABLED'] == (not before)
    finally:
        app.destroy()


def test_toggle_config_same_value_does_not_mark_dirty(monkeypatch):
    app = make_app(monkeypatch)
    try:
        current = app.cfg.get('NER_ENABLED', False)
        app._toggle('NER_ENABLED', current)  # no actual change
        assert app._dirty is False
    finally:
        app.destroy()


def test_toggle_config_not_dirty_when_restored_during_project_load(monkeypatch, tmp_path):
    path = tmp_path / "proj.trenproj"
    payload = {
        "version": 2,
        "cfg": {"NER_ENABLED": True, "FEATURE_MATRIX_LANGUAGE": True},
        "datasets": [{"name": "Data 1", "blocks": []}],
        "active_dataset_index": 0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    app = make_app(monkeypatch)
    try:
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(path))
        app.open_project_save()
        app.update()
        # self.cfg was replaced wholesale by loading, not via _toggle -- this
        # must not mark the freshly-opened project dirty.
        assert app.cfg['NER_ENABLED'] is True
        assert app._dirty is False
    finally:
        app.destroy()


def test_config_dirty_full_cycle_clean_dirty_save_clean(monkeypatch, tmp_path):
    # The exact scenario from the audit: clean saved project -> toggle a
    # config checkbox -> dirty -> save -> clean -> switch dataset without
    # touching config -> still clean.
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj_cfg")
    monkeypatch.setattr(caa.messagebox, "askyesno", lambda *a, **k: True)  # confirm overwrite on 2nd save
    try:
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app.datasets.append(new_ds)
        app.update()
        assert app.save_project_progress() is True
        assert app._dirty is False

        before = app.cfg.get('FEATURE_MATRIX_LANGUAGE', False)
        app._toggle('FEATURE_MATRIX_LANGUAGE', not before)
        assert app._dirty is True

        assert app.save_project_progress() is True
        assert app._dirty is False

        app._switch_dataset(1)
        app.update()
        assert app._dirty is False
        app._switch_dataset(0)
        app.update()
        assert app._dirty is False
    finally:
        app.destroy()


def test_export_does_not_mark_dirty(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        app._mark_clean()

        out_path = tmp_path / "export.txt"
        monkeypatch.setattr(caa.filedialog, "asksaveasfilename", lambda **k: str(out_path))
        app.open_export_dialog()
        app.update()
        dlg = find_toplevel(app, "Export Table")
        buttons = {w.cget('text'): w for w in find_widgets(dlg) if isinstance(w, caa.ttk.Button)}
        real_click(buttons["Export"])
        app.update()

        assert app._dirty is False
    finally:
        app.destroy()


def test_successful_save_marks_clean(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        assert app._dirty is True

        monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj1")
        ok = app.save_project_progress()
        assert ok is True
        assert app._dirty is False
    finally:
        app.destroy()


def test_successful_open_marks_clean(monkeypatch, tmp_path):
    saved_path = tmp_path / "proj.trenproj"
    payload = {
        "version": 2,
        "datasets": [{"name": "Data 1", "blocks": [], "source_text": "hi"}],
        "active_dataset_index": 0,
    }
    with open(saved_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    app = make_app(monkeypatch)
    try:
        app.txt_input.insert("1.0", "unsaved stuff")
        app.update()
        assert app._dirty is True

        # Dirty, so open_project_save's unsaved-changes guard fires first;
        # Discard lets the open proceed without saving the throwaway text.
        monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: False)
        monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: str(saved_path))
        app.open_project_save()
        app.update()
        assert app._dirty is False
    finally:
        app.destroy()


def test_closing_clean_project_shows_no_prompt_and_closes(monkeypatch):
    app = make_app(monkeypatch)
    ask_calls = []
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: ask_calls.append(a))
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        assert app._dirty is False
        app._on_close_request()
        assert ask_calls == []
        assert destroy_calls["n"] == 1
    finally:
        tk.Tk.destroy(app)


def test_closing_dirty_project_shows_prompt(monkeypatch):
    app = make_app(monkeypatch)
    ask_calls = []
    monkeypatch.setattr(caa.messagebox, "askyesnocancel",
                         lambda *a, **k: ask_calls.append(a) or None)  # None == Cancel
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        app.txt_input.insert("1.0", "unsaved")
        app.update()
        app._on_close_request()
        assert len(ask_calls) == 1
        assert destroy_calls["n"] == 0  # Cancel keeps the app open
    finally:
        tk.Tk.destroy(app)


def test_new_project_warns_only_when_dirty(monkeypatch):
    app = make_app(monkeypatch)
    ask_calls = []
    # New Project goes through the shared Save/Discard/Cancel guard, so the
    # relevant messagebox call is askyesnocancel, not the old 2-way askyesno.
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: ask_calls.append(a) or False)
    try:
        assert app._dirty is False
        app.new_project()
        app.update()
        assert ask_calls == []  # clean project: no prompt at all

        app.txt_input.insert("1.0", "dirty now")
        app.update()
        assert app._dirty is True
        app.new_project()
        app.update()
        assert len(ask_calls) == 1  # dirty project: prompt shown
    finally:
        app.destroy()


def test_new_project_cancel_leaves_current_project_untouched(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: None)  # Cancel
    try:
        app.txt_input.insert("1.0", "keep me")
        app.update()
        app.new_project()
        app.update()
        assert app.datasets[0]['name'] == "Data 1"
        assert app.txt_input.get("1.0", "end-1c") == "keep me"
        assert app._dirty is True
    finally:
        app.destroy()


def test_new_project_discard_proceeds_without_saving(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: False)  # Discard
    save_calls = {"n": 0}
    try:
        orig_save = app.save_project_progress
        app.save_project_progress = lambda: (save_calls.__setitem__("n", save_calls["n"] + 1), orig_save())[1]
        app.txt_input.insert("1.0", "throwaway")
        app.update()
        app.new_project()
        app.update()
        assert save_calls["n"] == 0
        assert app.txt_input.get("1.0", "end-1c") == ""
        assert app._dirty is False
    finally:
        app.destroy()


def test_new_project_save_succeeds_then_proceeds(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # Save
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj_np")
    try:
        app.txt_input.insert("1.0", "to be saved")
        app.update()
        app.new_project()
        app.update()
        saved_path = os.path.join(caa.APP_DIR, "proj_np" + caa.PROJECT_EXT)
        assert os.path.isfile(saved_path)
        assert app.txt_input.get("1.0", "end-1c") == ""  # New Project proceeded
        assert app._dirty is False
    finally:
        app.destroy()


def test_new_project_save_cancelled_aborts_new_project(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # Save
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: None)  # cancels name dialog
    try:
        app.txt_input.insert("1.0", "must survive")
        app.update()
        app.new_project()
        app.update()
        assert app.txt_input.get("1.0", "end-1c") == "must survive"  # New Project aborted
        assert app._dirty is True
    finally:
        app.destroy()


# =========================================================================
# Close flow: Save must not silently discard work if it didn't happen
# =========================================================================

def test_close_cancel_project_name_dialog_keeps_app_open(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # "Save"
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: None)  # user cancels name dialog
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        app.txt_input.insert("1.0", "unsaved")
        app.update()
        app._on_close_request()
        assert destroy_calls["n"] == 0
        assert app._dirty is True  # nothing was saved, still dirty
    finally:
        tk.Tk.destroy(app)


def test_close_reject_overwrite_keeps_app_open(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    # A file with this name already exists.
    existing = os.path.join(caa.APP_DIR, "proj1" + caa.PROJECT_EXT)
    os.makedirs(caa.APP_DIR, exist_ok=True)
    with open(existing, "w", encoding="utf-8") as f:
        f.write("{}")

    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # "Save"
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj1")
    monkeypatch.setattr(caa.messagebox, "askyesno", lambda *a, **k: False)  # reject overwrite
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        app.txt_input.insert("1.0", "unsaved")
        app.update()
        app._on_close_request()
        assert destroy_calls["n"] == 0
        assert app._dirty is True
    finally:
        tk.Tk.destroy(app)


def test_close_save_write_failure_keeps_app_open(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # "Save"
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj1")

    import builtins
    real_open = builtins.open

    def _boom(path, *a, **k):
        if str(path).endswith(caa.PROJECT_EXT):
            raise OSError("simulated disk failure")
        return real_open(path, *a, **k)

    monkeypatch.setattr(caa, "open", _boom, raising=False)
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        app.txt_input.insert("1.0", "unsaved")
        app.update()
        app._on_close_request()
        assert destroy_calls["n"] == 0
        assert app._dirty is True
    finally:
        tk.Tk.destroy(app)


def test_close_successful_save_closes_app(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: True)  # "Save"
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj1")
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        app.txt_input.insert("1.0", "unsaved")
        app.update()
        app._on_close_request()
        assert destroy_calls["n"] == 1
    finally:
        tk.Tk.destroy(app)


def test_close_explicit_discard_closes_without_saving(monkeypatch):
    app = make_app(monkeypatch)
    monkeypatch.setattr(caa.messagebox, "askyesnocancel", lambda *a, **k: False)  # "Discard"
    save_calls = {"n": 0}
    orig_save = app.save_project_progress
    def _tracked_save():
        save_calls["n"] += 1
        return orig_save()
    app.save_project_progress = _tracked_save
    destroy_calls = {"n": 0}
    app.destroy = lambda: destroy_calls.__setitem__("n", destroy_calls["n"] + 1)
    try:
        app.txt_input.insert("1.0", "unsaved")
        app.update()
        app._on_close_request()
        assert save_calls["n"] == 0
        assert destroy_calls["n"] == 1
    finally:
        tk.Tk.destroy(app)


# =========================================================================
# Undo history is session-only: reset after project load
# =========================================================================

def test_save_reopen_restores_data_but_resets_undo_stacks(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        app.blocks[0][0]['label'] = 'UID'
        app.open_uid_review_tool()
        app.update()
        app._uid_label_var.set('TR')
        app._uid_apply()
        app.update()
        assert len(app._uid_undo_stack) == 1

        monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "proj_undo")
        assert app.save_project_progress() is True

        saved_path = os.path.join(caa.APP_DIR, "proj_undo" + caa.PROJECT_EXT)
        with open(saved_path, encoding="utf-8") as f:
            payload = json.load(f)
        assert "uid_undo_stack" not in payload["datasets"][0]
        assert "merge_cells_undo_stack" not in payload["datasets"][0]

        app2 = caa.App()
        monkeypatch.setattr(caa.messagebox, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
        try:
            monkeypatch.setattr(caa.filedialog, "askopenfilename", lambda **k: saved_path)
            app2.open_project_save()
            app2.update()
            tokens = [r['token'] for r in app2.blocks[0] if not annotation_model.is_meta_row_token(r['token'])]
            assert tokens == ['alpha']
            assert app2._uid_undo_stack == []
            assert app2._merge_cells_undo_stack == []
        finally:
            app2.destroy()
    finally:
        app.destroy()


# =========================================================================
# Active-dataset synchronization immediately after run_pipeline
# =========================================================================

def test_run_pipeline_immediately_syncs_active_dataset(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha beta")
        app.run_pipeline()
        app.update()

        ds = app.datasets[app._active_dataset_index]
        assert ds['blocks'] is app.blocks  # same live object, not stale
        assert ds['source_text'] == "alpha beta"
    finally:
        app.destroy()


def test_run_pipeline_does_not_touch_other_datasets(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        new_ds = app._create_dataset_from_text("Data 2", "beta")
        app._sync_active_dataset_from_live()
        app.datasets.append(new_ds)
        app._load_dataset_into_live(1)
        app.update()

        other_before = copy.deepcopy(app.datasets[0])

        app.txt_input.delete("1.0", "end")
        app.txt_input.insert("1.0", "gamma")
        app.run_pipeline()
        app.update()

        assert app.datasets[0] == other_before
        assert app.datasets[1]['source_text'] == "gamma"
    finally:
        app.destroy()


def test_run_pipeline_failure_leaves_previous_dataset_intact(monkeypatch):
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        before_blocks = copy.deepcopy(app.blocks)
        before_source = app.datasets[app._active_dataset_index]['source_text']

        stub_pipeline(app, monkeypatch, fail=True)
        app.txt_input.delete("1.0", "end")
        app.txt_input.insert("1.0", "this will fail")
        app.run_pipeline()
        app.update()

        # self.blocks (and therefore the active dataset) must be untouched --
        # not partially cleared -- by the failed run.
        assert app.blocks == before_blocks
        assert app.datasets[app._active_dataset_index]['source_text'] == before_source
    finally:
        app.destroy()


def test_populate_table_is_transactional_on_grid_construction_failure(monkeypatch):
    # Distinct from test_run_pipeline_failure_leaves_previous_dataset_intact
    # above: here the annotation pipeline itself SUCCEEDS and returns text,
    # but the grid-construction step inside _populate_table fails. Before
    # the transactional rewrite this would have already replaced
    # self.blocks with the freshly (but incompletely) parsed result.
    app = make_app(monkeypatch)
    stub_pipeline(app, monkeypatch)
    try:
        app.txt_input.insert("1.0", "alpha")
        app.run_pipeline()
        app.update()
        before_blocks = copy.deepcopy(app.blocks)
        before_row_index_map = dict(app._row_index_map)
        before_sep_rows = set(app._sep_rows)
        before_source = app.datasets[app._active_dataset_index]['source_text']
        before_dirty = app._dirty
        try:
            sheet_data_before = app.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            sheet_data_before = app.sheet.get_sheet_data()

        def _boom(*a, **k):
            raise RuntimeError("simulated grid-construction failure")

        monkeypatch.setattr(annotation_model, "build_grid_view", _boom)

        app.txt_input.delete("1.0", "end")
        app.txt_input.insert("1.0", "this will fail during grid construction")
        monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: None)
        app.run_pipeline()
        app.update()

        assert app.blocks == before_blocks
        assert app._row_index_map == before_row_index_map
        assert app._sep_rows == before_sep_rows
        assert app.datasets[app._active_dataset_index]['source_text'] == before_source
        assert app._dirty == before_dirty
        try:
            sheet_data_after = app.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            sheet_data_after = app.sheet.get_sheet_data()
        assert sheet_data_after == sheet_data_before
        assert app.blocks  # not emptied
    finally:
        app.destroy()
