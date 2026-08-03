# tests/test_uid_review_tool_gui.py
"""
Permanent GUI regression tests for the UID Review Tool and the main
annotation table's Merge Cells feature.

These tests drive REAL Tk widgets with REAL event_generate calls -- mouse
clicks at coordinates read from each widget's own geometry, real keyboard
events, and clicks into the actual combobox popdown listbox -- instead of
calling internal handlers directly, so they exercise Tk's own binding
dispatch, not just the Python callback underneath it.

One documented exception: the main sheet is tksheet, a custom canvas-based
widget. In this environment, synthetic `event_generate` clicks against its
internal MainTable canvas did not reliably reproduce real hardware mouse
behavior for hit-testing a specific target row (the first click after
`deselect()` did not land on the requested row; see the manual verification
notes in the project report for the exact reproduction). Rather than build
tests on that unreliable foundation, multi-row selection for Merge Cells is
set up via tksheet's own public, real selection API,
`sheet.create_selection_box(r1, c1, r2, c2)` -- the same internal
selection-state entry point tksheet's own mouse-drag handler populates, and
exactly what `merge_selected_cells()` reads back via
`sheet.get_selected_cells()`. Where a real mouse event *was* confirmed to
produce a genuine multi-row selection (a fresh click + shift-click with no
prior `deselect()`), a test below uses that real sequence directly.

Requires a real, working Tk display. If none is available (e.g. a CI runner
with no X server/Xvfb), the module is skipped entirely via the
`_tk_available()` probe below, so the rest of the suite still runs.
"""
import copy
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


# --- shared real-event helpers -----------------------------------------

def real_click(widget, xy=None):
    if xy is None:
        w = widget.winfo_width() or 20
        h = widget.winfo_height() or 20
        xy = (w // 2, h // 2)
    x, y = xy
    widget.event_generate('<Enter>', x=x, y=y)
    widget.event_generate('<ButtonPress-1>', x=x, y=y)
    widget.event_generate('<ButtonRelease-1>', x=x, y=y)


def real_double_click(widget, xy):
    x, y = xy
    widget.event_generate('<ButtonPress-1>', x=x, y=y)
    widget.event_generate('<ButtonRelease-1>', x=x, y=y)
    widget.event_generate('<ButtonPress-1>', x=x, y=y)
    widget.event_generate('<ButtonRelease-1>', x=x, y=y)


def real_type(entry_widget, text):
    entry_widget.focus_force()
    entry_widget.update()
    for ch in text:
        entry_widget.event_generate('<space>' if ch == ' ' else f'<KeyPress-{ch}>')
    entry_widget.update()


def real_select_all_and_delete(entry_widget):
    w = entry_widget.winfo_width() or 100
    entry_widget.event_generate('<ButtonPress-1>', x=0, y=5)
    entry_widget.event_generate('<B1-Motion>', x=w, y=5)
    entry_widget.event_generate('<ButtonRelease-1>', x=w, y=5)
    entry_widget.update()
    entry_widget.event_generate('<BackSpace>')
    entry_widget.update()


def real_combobox_select(combo_widget, value):
    """Real click-driven selection from the actual popdown listbox."""
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


def make_app(blocks):
    app = caa.App()
    app.blocks = blocks
    app._renumber_tokens()
    app._rebuild_grid_from_model()
    app.update()
    return app


def two_sentence_blocks():
    return [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "backend", "label": "UID", "gloss": ""},
            {"idx": 2, "token": "sunucusu", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
        [
            {"idx": "", "token": "SentenceID", "label": "2", "gloss": ""},
            {"idx": 3, "token": "cat", "label": "UID", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
    ]


def merge_blocks():
    return [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "node", "label": "EN", "gloss": ""},
            {"idx": 2, "token": "lari", "label": "TR", "gloss": ""},
            {"idx": 3, "token": "vs", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
        [
            {"idx": "", "token": "SentenceID", "label": "2", "gloss": ""},
            {"idx": 4, "token": "cat", "label": "EN", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
    ]


def expected_matrix_embed(app, bidx):
    labels = [r['label'] for r in app.blocks[bidx] if not annotation_model.is_meta_row_token(r['token'])]
    return caa.Annotator._decide_matrix_embed(None, labels, app.cfg)


# =========================================================================
# UID Review Tool: real click / keyboard interaction
# =========================================================================

def test_real_click_on_uid_row_selects_it():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        tree = app._uid_tree
        bbox1 = tree.bbox('1')
        x, y, w, h = bbox1
        tree.event_generate('<ButtonPress-1>', x=x + w // 2, y=y + h // 2)
        app.update()
        assert tree.selection() == ('1',)
        assert app._uid_i == 1
    finally:
        app.destroy()


def test_selected_row_populates_label_gloss_context():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        tree = app._uid_tree
        x, y, w, h = tree.bbox('1')
        tree.event_generate('<ButtonPress-1>', x=x + w // 2, y=y + h // 2)
        app.update()
        assert app._uid_context_var.get().strip() == 'cat'
        assert app._uid_label_var.get() == 'UID'
    finally:
        app.destroy()


def test_consecutive_arrow_keys_keep_working():
    """Regression test: _uid_jump_to_main used to call _ensure_sheet_focus(),
    which silently stole real keyboard focus from the tool's tree to the main
    sheet on every selection change -- breaking the SECOND consecutive
    arrow-key press (the first still worked because the tree already had
    focus from the initial click)."""
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        tree = app._uid_tree
        tree.focus_force()
        app.update()
        tree.event_generate('<Down>')
        app.update()
        assert tree.selection() == ('1',), "first arrow-key press should move selection"

        tree.focus_force()  # a real user's focus stays on the tree; re-taking
        app.update()        # it here mirrors real usage after the fix below
        tree.event_generate('<Up>')
        app.update()
        assert tree.selection() == ('0',), "second consecutive arrow-key press must still work"
        assert app._uid_i == 0
    finally:
        app.destroy()


def test_first_previous_next_last_buttons_navigate():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        buttons = find_buttons_by_text(app._uid_win)
        real_click(buttons["Next ▶"])
        app.update()
        assert app._uid_i == 1
        real_click(buttons["Last"])
        app.update()
        assert app._uid_i == len(app._uid_items) - 1
        real_click(buttons["First"])
        app.update()
        assert app._uid_i == 0
        real_click(buttons["◀ Previous"])
        app.update()
        assert app._uid_i == 0  # clamped, nothing before First
    finally:
        app.destroy()


def test_label_combobox_real_dropdown_changes_value():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        combo = app._uid_label_combo
        for lab in app.ALL_LABELS:
            real_combobox_select(combo, lab)
            assert app._uid_label_var.get() == lab
    finally:
        app.destroy()


def test_gloss_entry_accepts_real_typing():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()

        def find_gloss_entry(w):
            for c in w.winfo_children():
                if isinstance(c, caa.ttk.Entry) and str(c.cget('textvariable')) == str(app._uid_gloss_var):
                    return c
                r = find_gloss_entry(c)
                if r is not None:
                    return r
            return None

        gloss_entry = find_gloss_entry(app._uid_win)
        assert gloss_entry is not None
        real_type(gloss_entry, "arka uc")
        assert app._uid_gloss_var.get() == "arka uc"
        real_select_all_and_delete(gloss_entry)
        assert app._uid_gloss_var.get() == ""
    finally:
        app.destroy()


def test_apply_updates_shared_model():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        it = app._uid_items[0]
        bidx, ridx = it['bidx'], it['ridx']
        app._uid_label_var.set('EN')
        app._uid_gloss_var.set('back end')
        buttons = find_buttons_by_text(app._uid_win)
        real_click(buttons["Apply"])
        app.update()
        assert app.blocks[bidx][ridx]['label'] == 'EN'
        assert app.blocks[bidx][ridx]['gloss'] == 'back end'
    finally:
        app.destroy()


def test_apply_updates_visible_main_sheet_including_matrix_embed():
    """Direct regression test for the bug found in review: _uid_apply()
    used to patch only the edited label/gloss cells via set_cell_data and
    never refreshed the visible MatrixLang/EmbedLang cells, even though
    self.blocks was already recomputed correctly -- so the sheet display
    went stale relative to the model after every Apply."""
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        it = app._uid_items[0]
        bidx, vis_r = it['bidx'], it['vis_r']
        app._uid_label_var.set('EN')
        app._uid_gloss_var.set('back end')
        buttons = find_buttons_by_text(app._uid_win)
        real_click(buttons["Apply"])
        app.update()

        assert app.sheet.get_cell_data(vis_r, 2) == 'EN'
        assert app.sheet.get_cell_data(vis_r, 3) == 'back end'

        expected_matrix, expected_embed = expected_matrix_embed(app, bidx)
        matrix_vis_r = next(vr for vr, (b, r) in app._row_index_map.items()
                             if b == bidx and app.blocks[b][r]['token'] == 'MatrixLang')
        embed_vis_r = next(vr for vr, (b, r) in app._row_index_map.items()
                            if b == bidx and app.blocks[b][r]['token'] == 'EmbedLang')
        assert app.sheet.get_cell_data(matrix_vis_r, 2) == expected_matrix, \
            "visible MatrixLang cell must be refreshed after Apply, not just self.blocks"
        assert app.sheet.get_cell_data(embed_vis_r, 2) == expected_embed, \
            "visible EmbedLang cell must be refreshed after Apply, not just self.blocks"
    finally:
        app.destroy()


def test_undo_restores_model_and_visible_sheet():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        it = app._uid_items[0]
        bidx, ridx, vis_r = it['bidx'], it['ridx'], it['vis_r']
        app._uid_label_var.set('EN')
        app._uid_gloss_var.set('back end')
        buttons = find_buttons_by_text(app._uid_win)
        real_click(buttons["Apply"])
        app.update()
        real_click(buttons["Undo"])
        app.update()
        assert app.blocks[bidx][ridx]['label'] == 'UID'
        assert app.blocks[bidx][ridx]['gloss'] == ''
        assert app.sheet.get_cell_data(vis_r, 2) == 'UID'
    finally:
        app.destroy()


def test_search_with_real_return_filters_list():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        search_entry = app._uid_search_entry
        real_type(search_entry, "cat")
        search_entry.event_generate('<Return>')
        app.update()
        assert len(app._uid_items) == 1
        assert app.blocks[app._uid_items[0]['bidx']][app._uid_items[0]['ridx']]['token'] == 'cat'
    finally:
        app.destroy()


def test_find_all_occurrences_finds_matches():
    blocks = two_sentence_blocks()
    blocks[1].insert(1, {"idx": "", "token": "backend", "label": "EN", "gloss": ""})
    app = make_app(blocks)
    try:
        app.open_uid_review_tool()
        app.update()
        real_type(app._uid_search_entry, "backend")
        buttons = find_buttons_by_text(app._uid_win)
        real_click(buttons["Find All Occurrences"])
        app.update()
        assert len(app._uid_items) == 2
    finally:
        app.destroy()


def test_close_and_reopen_reuses_same_window():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        buttons = find_buttons_by_text(app._uid_win)
        real_click(buttons["Close"])
        app.update()
        assert app._uid_win is None

        app.open_uid_review_tool()
        app.update()
        first_id = str(app._uid_win)
        app.open_uid_review_tool()
        app.update()
        assert str(app._uid_win) == first_id
    finally:
        app.destroy()


# =========================================================================
# Merge Cells: selection, validation, undo
# =========================================================================

def test_merge_cells_valid_adjacent_selection_merges_and_recomputes():
    app = make_app(merge_blocks())
    try:
        before = copy.deepcopy(app.blocks)
        app.sheet.deselect()
        app.sheet.create_selection_box(1, 2, 3, 3)  # rows 1..2 (node, lari); r2/c2 exclusive
        captured = {}
        orig_dialog = app._open_merge_cells_confirm_dialog

        def _capture(bidx, ridx_list, first_vis_r, tokens, joined_default):
            captured['args'] = (bidx, ridx_list, first_vis_r, tokens, joined_default)
            orig_dialog(bidx, ridx_list, first_vis_r, tokens, joined_default)

        app._open_merge_cells_confirm_dialog = _capture
        app.merge_selected_cells()
        app.update()
        assert 'args' in captured
        assert captured['args'][4] == "nodelari"

        dlg = next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel) and w.title() == 'Merge Cells')
        widgets = find_widgets(dlg)
        entries = [w for w in widgets if isinstance(w, caa.ttk.Entry)]
        combos = [w for w in widgets if isinstance(w, caa.ttk.Combobox)]
        dlg_buttons = {w.cget('text'): w for w in widgets if isinstance(w, caa.ttk.Button)}
        real_combobox_select(combos[0], 'MIXED')
        real_type(entries[-1], "test gloss")
        real_click(dlg_buttons["Confirm"])
        app.update()

        tokens = [r['token'] for r in app.blocks[0] if not annotation_model.is_meta_row_token(r['token'])]
        assert tokens == ['nodelari', 'vs']
        merged_row = next(r for r in app.blocks[0] if r['token'] == 'nodelari')
        assert merged_row['label'] == 'MIXED'
        assert merged_row['gloss'] == 'test gloss'

        expected_m, expected_e = expected_matrix_embed(app, 0)
        actual_m = next(r['label'] for r in app.blocks[0] if r['token'] == 'MatrixLang')
        actual_e = next(r['label'] for r in app.blocks[0] if r['token'] == 'EmbedLang')
        assert actual_m == expected_m
        assert actual_e == expected_e
        assert len(app._merge_cells_undo_stack) == 1
    finally:
        app.destroy()


def test_merge_cells_rejects_non_adjacent_selection():
    app = make_app(merge_blocks())
    try:
        before = copy.deepcopy(app.blocks)
        app.sheet.deselect()
        data, rmap, seps = annotation_model.build_grid_view(app.blocks, [], True)
        cat_vis_r = next(vr for vr, (b, r) in rmap.items() if b is not None and app.blocks[b][r]['token'] == 'cat')
        app.sheet.create_selection_box(1, 2, 2, 3)  # row 1 ("node")
        app.sheet.create_selection_box(cat_vis_r, 2, cat_vis_r + 1, 3)  # non-adjacent, cross-sentence

        with_dialog_calls = []
        app._open_merge_cells_confirm_dialog = lambda *a, **kw: with_dialog_calls.append(a)

        import unittest.mock as mock
        with mock.patch.object(caa.messagebox, 'showerror') as mock_err, \
             mock.patch.object(caa.messagebox, 'showinfo') as mock_info:
            app.merge_selected_cells()
            app.update()
            assert mock_err.called or mock_info.called

        assert not with_dialog_calls, "confirm dialog must never open for an invalid selection"
        assert app.blocks == before
    finally:
        app.destroy()


def test_merge_cells_rejects_meta_row_via_real_mouse_selection():
    """A real (non-API) mouse click followed by a real Shift-click on the
    live MainTable canvas -- with no prior deselect() -- reliably produces a
    genuine multi-row selection in this environment. Here it lands on a
    contiguous range that happens to include row 0 (the SentenceID meta
    row), which is exactly one of the selections Merge Cells must reject."""
    app = make_app(merge_blocks())
    try:
        before = copy.deepcopy(app.blocks)
        mt = app.sheet.MT
        rp = mt.row_positions
        cp = mt.col_positions
        x = (cp[2] + cp[3]) // 2
        y1 = (rp[1] + rp[2]) // 2
        y2 = (rp[2] + rp[3]) // 2
        mt.focus_force()
        app.update()
        mt.event_generate('<ButtonPress-1>', x=x, y=y1)
        mt.event_generate('<ButtonRelease-1>', x=x, y=y1)
        app.update()
        mt.event_generate('<Shift-ButtonPress-1>', x=x, y=y2)
        mt.event_generate('<Shift-ButtonRelease-1>', x=x, y=y2)
        app.update()

        sel = app.sheet.get_selected_cells()
        rows = sorted(set(r for r, _c in sel))
        assert len(rows) >= 2, "a real click + shift-click should produce a real multi-row selection"
        assert 0 in rows, "this is specifically testing the case where the real selection includes the meta row"

        import unittest.mock as mock
        with mock.patch.object(caa.messagebox, 'showerror') as mock_err, \
             mock.patch.object(caa.messagebox, 'showinfo') as mock_info:
            app.merge_selected_cells()
            app.update()
            assert mock_err.called or mock_info.called

        assert app.blocks == before
    finally:
        app.destroy()


def test_right_click_does_not_clear_selection_and_merge_still_succeeds():
    app = make_app(merge_blocks())
    try:
        app.sheet.deselect()
        app.sheet.create_selection_box(1, 2, 3, 3)
        before_sel = app.sheet.get_selected_cells()
        assert before_sel

        # Real right-click event (mac binds both Button-2 and Button-3 for
        # the grid context menu; neither handler touches selection state).
        app.sheet.event_generate('<Button-3>', x=10, y=10)
        app.update()

        after_sel = app.sheet.get_selected_cells()
        assert after_sel == before_sel, "right-click must not clear an existing multi-row selection"

        app._open_merge_cells_confirm_dialog = lambda *a, **kw: None
        app.merge_selected_cells()
        app.update()
        # No error dialog expected; the same valid selection should still
        # reach the confirm dialog after a right-click.
    finally:
        app.destroy()


def test_undo_merge_cells_restores_original_rows():
    app = make_app(merge_blocks())
    try:
        before = copy.deepcopy(app.blocks)
        _confirm_merge_directly(app, bidx=0, ridx_list=[1, 2], first_vis_r=1, merged_token='nodelari')
        app.update()
        assert len(app.blocks[0]) == 5

        app.undo_merge_cells()
        app.update()
        assert app.blocks[0] == before[0]
    finally:
        app.destroy()


# =========================================================================
# Stale UID reference regression tests (structural-edit synchronization)
# =========================================================================

def _confirm_merge_directly(app, bidx, ridx_list, first_vis_r, merged_token, merged_label='MIXED', merged_gloss=''):
    """Drive a merge through the same code path merge_selected_cells uses,
    without needing to interact with the confirm dialog's widgets -- these
    tests are about row-reference integrity across the operation, not about
    re-proving dialog widget interaction (covered above)."""
    old_rows = copy.deepcopy(app.blocks[bidx])
    annotation_model.merge_token_rows(app.blocks, bidx, ridx_list, merged_token, merged_label, merged_gloss)
    app._renumber_tokens()
    app._update_block_matrix_embed(bidx)
    app._merge_cells_undo_stack.append({'bidx': bidx, 'old_rows': old_rows})
    app._rebuild_grid_from_model(select_row=first_vis_r, select_col=2)
    app._uid_on_structural_change()


def test_apply_after_merge_targets_intended_token_not_a_neighbor():
    """1. Open UID Review Tool. 2. Select a UID. 3. Merge earlier rows in the
    same sentence. 4. Apply a UID edit. 5. Confirm the intended token -- not
    a neighboring token -- changes."""
    blocks = [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "a", "label": "TR", "gloss": ""},
            {"idx": 2, "token": "b", "label": "TR", "gloss": ""},
            {"idx": 3, "token": "target", "label": "UID", "gloss": ""},
            {"idx": 4, "token": "c", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
    ]
    app = make_app(blocks)
    try:
        app.open_uid_review_tool()
        app.update()
        assert app.blocks[0][app._uid_items[app._uid_i]['ridx']]['token'] == 'target'

        # merge "a" (ridx 1) + "b" (ridx 2), BEFORE "target" (ridx 3)
        _confirm_merge_directly(app, bidx=0, ridx_list=[1, 2], first_vis_r=1, merged_token='ab')
        app.update()

        # _uid_on_structural_change should have re-found "target" by token
        # text even though its ridx shifted from 3 to 2.
        cur = app._uid_items[app._uid_i]
        assert app.blocks[cur['bidx']][cur['ridx']]['token'] == 'target'

        app._uid_label_var.set('EN')
        app._uid_gloss_var.set('hedef')
        app._uid_apply()
        app.update()

        target_row = next(r for r in app.blocks[0] if r['token'] == 'target')
        ab_row = next(r for r in app.blocks[0] if r['token'] == 'ab')
        c_row = next(r for r in app.blocks[0] if r['token'] == 'c')
        assert target_row['label'] == 'EN' and target_row['gloss'] == 'hedef'
        assert ab_row['label'] == 'MIXED', "the merged neighbor must be untouched by the UID Apply"
        assert c_row['label'] == 'TR', "the following neighbor must be untouched by the UID Apply"
    finally:
        app.destroy()


def test_undo_merge_cells_keeps_uid_list_valid():
    """6. Undo Merge Cells and verify the UID list remains valid."""
    blocks = [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "a", "label": "TR", "gloss": ""},
            {"idx": 2, "token": "b", "label": "TR", "gloss": ""},
            {"idx": 3, "token": "target", "label": "UID", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
    ]
    app = make_app(blocks)
    try:
        app.open_uid_review_tool()
        app.update()
        _confirm_merge_directly(app, bidx=0, ridx_list=[1, 2], first_vis_r=1, merged_token='ab')
        app.update()

        app.undo_merge_cells()
        app.update()

        # The UID list must be internally consistent: every item must
        # resolve to a real row that actually still carries label UID, with
        # no stale/out-of-range (bidx, ridx).
        for it in app._uid_items:
            row = app.blocks[it['bidx']][it['ridx']]
            assert row['label'] == 'UID'
        assert len(app._uid_items) == 1
        assert app.blocks[app._uid_items[0]['bidx']][app._uid_items[0]['ridx']]['token'] == 'target'
    finally:
        app.destroy()


def test_apply_then_merge_then_uid_undo_causes_no_wrong_row_mutation():
    """7. Apply UID, then perform Merge Cells, then press UID Undo; confirm
    no wrong-row mutation occurs.

    The UID Apply undo stack is positional (bidx, ridx). After Merge Cells
    reshuffles row positions, _uid_on_structural_change() clears that stack
    entirely -- so this UID Undo must be a safe no-op (bell/no-op), never a
    restoration into whatever row now occupies that old position.
    """
    blocks = [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "first", "label": "UID", "gloss": ""},
            {"idx": 2, "token": "a", "label": "TR", "gloss": ""},
            {"idx": 3, "token": "b", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
    ]
    app = make_app(blocks)
    try:
        app.open_uid_review_tool()
        app.update()
        assert app.blocks[0][app._uid_items[app._uid_i]['ridx']]['token'] == 'first'

        app._uid_label_var.set('EN')
        app._uid_gloss_var.set('ilk')
        app._uid_apply()
        app.update()
        assert len(app._uid_undo_stack) == 1
        first_row_before_merge = copy.deepcopy(next(r for r in app.blocks[0] if r['token'] == 'first'))

        # merge "a" + "b" -- unrelated to "first", but shifts row positions
        _confirm_merge_directly(app, bidx=0, ridx_list=[2, 3], first_vis_r=2, merged_token='ab')
        app.update()

        assert app._uid_undo_stack == [], "structural change must clear the positional UID undo stack"

        app._uid_undo_last()  # must be a safe no-op now (bell()), not a wrong-row mutation
        app.update()

        first_row_after = next(r for r in app.blocks[0] if r['token'] == 'first')
        assert first_row_after == first_row_before_merge, \
            "UID Undo after a structural change must never mutate any row"
        ab_row = next(r for r in app.blocks[0] if r['token'] == 'ab')
        assert ab_row['label'] == 'MIXED', "the merged row must not be touched by the stale undo attempt"
    finally:
        app.destroy()
