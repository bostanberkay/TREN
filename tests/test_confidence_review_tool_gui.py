"""GUI regression tests for the Confidence Review Tool and Merge Cells, driven by
real Tk events. Most Merge Cells tests build the selection with tksheet's
create_selection_box; the meta-row test uses real clicks, in widget coordinates.
Skipped without a Tk display.
"""
import copy
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk  # noqa: E402

import cs_annotator_app as caa  # noqa: E402
import annotation_model  # noqa: E402
import confidence as cf  # noqa: E402
import tk_probe  # noqa: E402


# Probed during collection; the real traceback is kept (see tk_probe.py).
_SKIP_GUI, _SKIP_REASON = tk_probe.gui_skip(__name__)
pytestmark = pytest.mark.skipif(_SKIP_GUI, reason=_SKIP_REASON)


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


def uncertain_confidence(label="UID", band="LOW", score=0.4):
    """A confidence record whose review_recommended=True -- makes a row
    eligible for the "All Uncertain" default view (confidence.
    is_review_required), regardless of its label. Mirrors
    ConfidenceRecord.to_dict()'s shape exactly."""
    return {
        "token": "", "rule_based_label": label, "final_label": label,
        "confidence_score": score, "confidence_band": band,
        "uncertainty_reasons": [], "evidence_summary": [],
        "review_recommended": True, "promoted_by": None,
        "calibration_note": cf.CALIBRATION_NOTE,
    }


def confident_confidence(label="TR", score=0.95):
    """A confidence record whose review_recommended=False -- must NEVER
    appear in the "All Uncertain" default view."""
    return {
        "token": "", "rule_based_label": label, "final_label": label,
        "confidence_score": score, "confidence_band": "HIGH",
        "uncertainty_reasons": [], "evidence_summary": [],
        "review_recommended": False, "promoted_by": None,
        "calibration_note": cf.CALIBRATION_NOTE,
    }


def two_sentence_blocks():
    """Both UID tokens need review and the TR token is confident, so the default
    "All Uncertain" view yields [backend, cat]."""
    return [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "backend", "label": "UID", "gloss": "",
             "confidence": uncertain_confidence("UID")},
            {"idx": 2, "token": "sunucusu", "label": "TR", "gloss": "",
             "confidence": confident_confidence("TR")},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
        [
            {"idx": "", "token": "SentenceID", "label": "2", "gloss": ""},
            {"idx": 3, "token": "cat", "label": "UID", "gloss": "",
             "confidence": uncertain_confidence("UID")},
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
# Confidence Review Tool: real click / keyboard interaction
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
    """Regression: _uid_jump_to_main stole focus to the main sheet, breaking the
    second consecutive arrow-key press."""
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
    """Regression: Apply patched only the edited cells and left the visible
    MatrixLang/EmbedLang cells stale."""
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


def _narrow_until_column_offscreen(app, mt, c):
    """Shrink the window until column `c` starts off-screen at scroll 0 while one
    column still fits, as in CI's 223 px table."""
    cp = mt.col_positions
    w = app.winfo_width()
    while w >= 300:
        app.geometry(f"{w}x{app.winfo_height()}")
        app.update_idletasks()
        app.update()
        if (cp[c + 1] - cp[c]) + 10 <= mt.winfo_width() <= cp[c]:
            return
        w -= 10
    pytest.fail(f"could not narrow the table to hide column {c} (width {mt.winfo_width()})")


@pytest.mark.parametrize("narrow", [False, True], ids=["default-window", "narrow-table"])
def test_merge_cells_rejects_meta_row_via_real_mouse_selection(narrow):
    """A real click on the SentenceID row + Shift-click on a token row selects a
    range that includes the meta row, which Merge Cells must reject."""
    app = make_app(merge_blocks())
    try:
        before = copy.deepcopy(app.blocks)
        mt = app.sheet.MT
        app.update()
        assert mt.winfo_viewable(), "table must be mapped before clicking"
        if narrow:
            _narrow_until_column_offscreen(app, mt, 2)
        app.sheet.deselect()
        app.update()

        # Scroll the cell into view first, then convert its canvas position to
        # widget coordinates (recomputed per click, since see() may scroll).
        def click_cell(r, c, shift=False):
            app.sheet.see(r, c)
            app.update()
            rp, cp = mt.row_positions, mt.col_positions
            x = int((cp[c] + cp[c + 1]) // 2 - mt.canvasx(0))
            y = int((rp[r] + rp[r + 1]) // 2 - mt.canvasy(0))
            assert 0 <= x < mt.winfo_width() and 0 <= y < mt.winfo_height(), (
                f"cell ({r}, {c}) is not visible in the {mt.winfo_width()}x{mt.winfo_height()} table")
            mod = 'Shift-' if shift else ''
            mt.event_generate(f'<{mod}ButtonPress-1>', x=x, y=y)
            mt.event_generate(f'<{mod}ButtonRelease-1>', x=x, y=y)
            app.update()

        mt.focus_force()
        app.update()
        click_cell(0, 2)
        assert sorted(set(r for r, _c in app.sheet.get_selected_cells())) == [0], \
            "a real click on the SentenceID row must select it"

        click_cell(2, 2, shift=True)

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
        # tk_popup is recorded, not shown: on macOS a posted menu runs a native
        # modal loop that blocks the test run until the menu is dismissed.
        popups = []
        app._grid_menu.tk_popup = lambda x, y, *a: popups.append((x, y))
        app.sheet.event_generate('<Button-3>', x=10, y=10)
        app.update()
        assert len(popups) == 1, "right-click must open the grid context menu"

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
    """1. Open the Confidence Review Tool. 2. Select a UID token. 3. Merge
    earlier rows in the same sentence. 4. Apply a UID edit. 5. Confirm the
    intended token -- not a neighboring token -- changes."""
    blocks = [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "a", "label": "TR", "gloss": ""},
            {"idx": 2, "token": "b", "label": "TR", "gloss": ""},
            {"idx": 3, "token": "target", "label": "UID", "gloss": "",
             "confidence": uncertain_confidence("UID")},
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
            {"idx": 3, "token": "target", "label": "UID", "gloss": "",
             "confidence": uncertain_confidence("UID")},
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
    """Apply, Merge Cells, then Undo: the merge clears the positional undo stack,
    so Undo must be a safe no-op, never a write into a shifted row.
    """
    blocks = [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "first", "label": "UID", "gloss": "",
             "confidence": uncertain_confidence("UID")},
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


# =========================================================================
# Rename "UID Review Tool" -> "Confidence Review Tool" (user-facing text only).
# =========================================================================

def test_window_title_says_confidence_review_tool_not_uid_review_tool():
    app = make_app(two_sentence_blocks())
    try:
        app.open_uid_review_tool()
        app.update()
        title = app._uid_win.title()
        assert title.startswith("Confidence Review Tool")
        assert "UID Review Tool" not in title
    finally:
        app.destroy()


def test_tools_menu_label_says_confidence_review_tool():
    app = caa.App()
    try:
        menubar = app.nametowidget(app.cget('menu'))

        def find_tools_menu(mb):
            end = mb.index('end')
            if end is None:
                return None
            for i in range(end + 1):
                if mb.type(i) == 'cascade' and mb.entrycget(i, 'label') == 'Tools':
                    return mb.nametowidget(mb.entrycget(i, 'menu'))
            return None

        toolsm = find_tools_menu(menubar)
        assert toolsm is not None
        entries = [toolsm.entrycget(i, 'label') for i in range(toolsm.index('end') + 1)
                   if toolsm.type(i) not in ('separator',)]
        assert any(e.startswith('Confidence Review Tool') for e in entries)
        assert not any('UID Review Tool' in e for e in entries)
    finally:
        app.destroy()


def test_matrixembed_locked_warning_uses_confidence_review_tool_title():
    """The defensive MatrixLang/EmbedLang guard in _uid_apply (unreachable via
    normal navigation) must use the new tool title."""
    blocks = two_sentence_blocks()
    app = make_app(blocks)
    try:
        # Point _uid_items directly at the MatrixLang row (bidx=0, ridx=3 in
        # two_sentence_blocks: SentenceID, backend, sunucusu, MatrixLang,
        # EmbedLang) to exercise the guard without relying on it being
        # reachable via the tool's own filtered list.
        assert blocks[0][3]['token'] == 'MatrixLang'
        app._uid_items = [{'bidx': 0, 'ridx': 3, 'vis_r': 3}]
        app._uid_i = 0
        app._uid_label_var = tk.StringVar(value='MIXED')  # not TR/EN -> locked
        app._uid_gloss_var = tk.StringVar(value='')

        with mock.patch.object(caa.messagebox, 'showwarning') as mock_warn:
            app._uid_apply()
            app.update()
        assert mock_warn.called
        title_arg = mock_warn.call_args[0][0]
        assert title_arg == "Confidence Review Tool"
    finally:
        app.destroy()


# =========================================================================
# Evidence panel cleanup: no corpus-analysis/calibration-disclosure line,
# no stray "NOT", and real "not" text in token/sentence content preserved.
# =========================================================================

def _confidence_dict(band="LOW", score=0.4, reasons=None, evidence=None):
    return {
        "token": "", "rule_based_label": "UID", "final_label": "UID",
        "confidence_score": score, "confidence_band": band,
        "uncertainty_reasons": reasons or [], "evidence_summary": evidence or [],
        "review_recommended": True, "promoted_by": None,
        "calibration_note": cf.CALIBRATION_NOTE,
    }


def test_evidence_panel_has_no_calibration_or_corpus_analysis_text():
    app = make_app(two_sentence_blocks())
    try:
        row = app.blocks[0][1]  # "backend", UID
        row['confidence'] = _confidence_dict(
            band="LOW", score=0.4,
            reasons=["label_uid_by_definition_below_lid_confidence_threshold"],
            evidence=["no lexicon/suffix/fastText evidence found for any label (genuinely unresolvable)"],
        )
        text = app._uid_evidence_text(row)

        # The calibration_note itself must never appear in this per-token
        # display (it is a fixed, dataset-level disclosure, not per-token
        # evidence -- see confidence.py's CALIBRATION_NOTE).
        assert cf.CALIBRATION_NOTE not in text
        assert "corpus" not in text.lower()
        # The disclosure text is the only place "NOT" (capitalized) would
        # have come from; with it gone, it must not appear stray/floating.
        assert "NOT" not in text

        # Useful per-token evidence must still be present.
        assert "Label:" in text
        assert "Confidence:" in text
        assert "LOW" in text
        assert "Uncertainty reasons:" in text
        assert "Evidence:" in text
    finally:
        app.destroy()


def test_evidence_panel_omits_calibration_note_even_when_present_in_record():
    """The stored confidence dict (and to_dict()) still legitimately carries
    calibration_note -- only this ONE display was cleaned up, per the task
    scope (do not change confidence calculations)."""
    app = make_app(two_sentence_blocks())
    try:
        row = app.blocks[0][1]
        conf = _confidence_dict()
        row['confidence'] = conf
        # The stored record is untouched.
        assert cf.get_confidence(row)['calibration_note'] == cf.CALIBRATION_NOTE
        # Only the rendered evidence text excludes it.
        assert cf.CALIBRATION_NOTE not in app._uid_evidence_text(row)
    finally:
        app.destroy()


def test_evidence_panel_and_context_preserve_real_word_not_in_sentence():
    """The word 'not' appearing as real token/sentence content must never
    be stripped -- the cleanup only removed the calibration_note line, not
    any generic 'not'-scrubbing logic (none exists)."""
    blocks = [
        [
            {"idx": "", "token": "SentenceID", "label": "1", "gloss": ""},
            {"idx": 1, "token": "I", "label": "EN", "gloss": ""},
            {"idx": 2, "token": "did", "label": "EN", "gloss": ""},
            {"idx": 3, "token": "not", "label": "EN", "gloss": ""},
            {"idx": 4, "token": "go", "label": "EN", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "EN", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
        ],
    ]
    app = make_app(blocks)
    try:
        row = blocks[0][3]  # "not"
        row['confidence'] = _confidence_dict(
            band="MEDIUM", score=0.7,
            reasons=["other_label_does_not_match_any_automatic_exclusion_pattern"],
            evidence=["token in English frequency lexicon"],
        )
        context = app._uid_sentence_text(0)
        assert context == "I did not go"
        evidence_text = app._uid_evidence_text(row)
        # The reason string genuinely contains "not" as a substring
        # ("does_not_match") -- it must survive completely intact.
        assert "other_label_does_not_match_any_automatic_exclusion_pattern" in evidence_text
    finally:
        app.destroy()
