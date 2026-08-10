# tests/test_tdk_checker_gui.py
"""Permanent GUI regression tests for the TDK Checker tool -- a SEPARATE
tool from the Confidence Review Tool (Tools -> TDK Checker; never replaces
or renames it).

Every test in this file uses dictionary_provider.MockDictionaryProvider or
UnavailableProvider, injected directly via `app._tdk_provider`, set BEFORE
any lookup is triggered. No test here (or anywhere in the suite) ever
constructs a real dictionary_provider.TDKProvider or makes a real network
call -- see tests/test_dictionary_provider.py for the provider's own
network-shaped tests, all of which use an injected fake opener instead of
the real network too.

Requires a real, working Tk display; skipped entirely otherwise (mirrors
tests/test_confidence_review_tool_gui.py's own convention).
"""
import copy
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk  # noqa: E402

import cs_annotator_app as caa  # noqa: E402
import annotation_model  # noqa: E402
import dictionary_provider as dp  # noqa: E402


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


# --- shared helpers ------------------------------------------------------

def real_click(widget, xy=None):
    if xy is None:
        w = widget.winfo_width() or 20
        h = widget.winfo_height() or 20
        xy = (w // 2, h // 2)
    x, y = xy
    widget.event_generate('<Enter>', x=x, y=y)
    widget.event_generate('<ButtonPress-1>', x=x, y=y)
    widget.event_generate('<ButtonRelease-1>', x=x, y=y)


def find_buttons_by_text(root_widget):
    out = {}

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, caa.ttk.Button):
                out[c.cget('text')] = c
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
    """"cloudumuza" (bidx=0) is a genuine English-stem+Turkish-suffix
    example (matches the task's own "cloud + umuz + a" example);
    "kitaplarda" (bidx=1) is a plain Turkish word with its own valid
    suffix chain, used for the second-sentence/dataset-switching tests."""
    return [
        [
            {"idx": "", "token": "SentenceID", "label": "rd_1", "gloss": ""},
            {"idx": 1, "token": "cloudumuza", "label": "UID", "gloss": ""},
            {"idx": 2, "token": "geldi", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "EN", "gloss": ""},
        ],
        [
            {"idx": "", "token": "SentenceID", "label": "rd_2", "gloss": ""},
            {"idx": 3, "token": "kitaplarda", "label": "TR", "gloss": ""},
            {"idx": "", "token": "MatrixLang", "label": "TR", "gloss": ""},
            {"idx": "", "token": "EmbedLang", "label": "-", "gloss": ""},
        ],
    ]


def mock_provider(delay=0.0):
    return dp.MockDictionaryProvider(
        responses={"cloudumuza": "NOT_FOUND", "cloud": "NOT_FOUND", "umuz": "NOT_FOUND", "a": "NOT_FOUND",
                   "kitaplarda": "FOUND", "kitap": "FOUND"},
        default_status="NOT_FOUND", delay_seconds=delay,
    )


def select_visible_row(app, vis_r, col=2):
    app.sheet.deselect()
    app.sheet.create_selection_box(vis_r, col, vis_r + 1, col + 1)
    app.update()


def wait_until(condition, timeout=3.0, app=None):
    start = time.time()
    while not condition() and time.time() - start < timeout:
        if app is not None:
            app.update()
        time.sleep(0.01)
    return condition()


def wait_for_tdk_idle(app, timeout=3.0):
    return wait_until(lambda: app._tdk_status_var.get() not in ("", "checking..."), timeout=timeout, app=app)


# =========================================================================
# Opening from the grid: token/sentence-ID/token-index propagation,
# automatic parse + lookup
# =========================================================================

def test_open_from_grid_populates_token_and_runs_parser_and_lookup():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)  # "cloudumuza"
        app.open_tdk_checker_from_grid()
        app.update()

        assert app._tdk_token_var.get() == "cloudumuza"
        # automatic parser execution, no manual Re-parse click needed
        assert app._tdk_root_var.get() == "cloud"
        assert app._tdk_segments_var.get() == "umuz + a"

        # automatic explicit lookup (triggered by the grid click itself)
        assert wait_for_tdk_idle(app)
        rows = [app._tdk_results_tree.item(i)['values'] for i in app._tdk_results_tree.get_children()]
        terms_checked = {r[0] for r in rows}
        assert terms_checked == {"cloudumuza", "cloud", "umuz", "a"}
        assert app._tdk_provider.call_count == 4
    finally:
        app.destroy()


def test_open_from_grid_propagates_sentence_id_and_token_index():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert app._tdk_sentid_var.get() == "rd_1"
        assert app._tdk_tokidx_var.get() == "1"
        assert app._tdk_context_var.get() == "cloudumuza geldi"
    finally:
        app.destroy()


def test_window_title_and_visible_fields_present():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert app._tdk_win.title() == 'TDK Checker'
        buttons = find_buttons_by_text(app._tdk_win)
        for label in ("Re-parse", "Check TDK", "Apply Correction", "Undo", "Find All Occurrences", "Close"):
            assert label in buttons
    finally:
        app.destroy()


def test_no_gloss_widgets_anywhere_in_tdk_checker_window():
    """Gloss is removed entirely from the TDK Checker (requirement: label,
    input, display, gloss-specific buttons/text) -- Gloss elsewhere in
    TREN (main table, Auto-Glossing Tool) is untouched and out of scope
    here."""
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert not hasattr(app, '_tdk_gloss_var')

        texts = []

        def walk(w):
            for c in w.winfo_children():
                for opt in ('text',):
                    try:
                        val = c.cget(opt)
                        if val:
                            texts.append(str(val))
                    except Exception:
                        pass
                walk(c)

        walk(app._tdk_win)
        assert not any('gloss' in t.lower() for t in texts)
    finally:
        app.destroy()


# =========================================================================
# Invalid-selection handling
# =========================================================================

def test_no_selection_shows_clear_message():
    app = make_app(two_sentence_blocks())
    try:
        app.sheet.deselect()
        app.update()
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert bidx is None
        assert err == ('info', "Select a token row first.")
    finally:
        app.destroy()


def test_multiple_selection_uses_first_valid_token_row():
    app = make_app(two_sentence_blocks())
    try:
        app.sheet.deselect()
        # rows 1 ("cloudumuza") and 2 ("geldi") both valid token rows
        app.sheet.create_selection_box(1, 2, 3, 3)
        app.update()
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert err is None
        assert app.blocks[bidx][ridx]['token'] == 'cloudumuza'  # first valid row, in order
    finally:
        app.destroy()


def test_multiple_selection_all_invalid_shows_single_selection_message():
    app = make_app(two_sentence_blocks())
    try:
        app.sheet.deselect()
        # rows 3 (MatrixLang) and 4 (EmbedLang) of block 0 -- both meta rows
        app.sheet.create_selection_box(3, 2, 5, 3)
        app.update()
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert bidx is None
        assert err is not None and err[0] == 'info'
    finally:
        app.destroy()


def test_matrixlang_row_disabled_with_warning():
    app = make_app(two_sentence_blocks())
    try:
        select_visible_row(app, 3)  # MatrixLang row of block 0
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert bidx is None
        assert err == ('warning', "MatrixLang rows cannot be opened in TDK Checker.")
    finally:
        app.destroy()


def test_embedlang_row_disabled_with_warning():
    app = make_app(two_sentence_blocks())
    try:
        select_visible_row(app, 4)  # EmbedLang row of block 0
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert bidx is None
        assert err == ('warning', "EmbedLang rows cannot be opened in TDK Checker.")
    finally:
        app.destroy()


def test_separator_row_disabled():
    app = make_app(two_sentence_blocks())
    try:
        # the blank separator row between block 0 and block 1
        sep_rows = sorted(app._sep_rows)
        assert sep_rows
        select_visible_row(app, sep_rows[0])
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert bidx is None
        assert err == ('warning', "Cannot open TDK Checker for a separator row.")
    finally:
        app.destroy()


def test_empty_token_shows_warning():
    blocks = two_sentence_blocks()
    blocks[0].insert(2, {"idx": "", "token": "", "label": "", "gloss": ""})
    app = make_app(blocks)
    try:
        select_visible_row(app, 2)  # the blank-token row just inserted
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert bidx is None
        assert err is not None and err[0] == 'warning'
    finally:
        app.destroy()


def test_grid_menu_entry_disabled_for_invalid_selection():
    app = make_app(two_sentence_blocks())
    try:
        select_visible_row(app, 3)  # MatrixLang
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        state = 'disabled' if err is not None else 'normal'
        app._grid_menu.entryconfig("Open in TDK Checker", state=state)
        assert app._grid_menu.entrycget("Open in TDK Checker", "state") == 'disabled'

        select_visible_row(app, 1)  # cloudumuza -- valid
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        state = 'disabled' if err is not None else 'normal'
        app._grid_menu.entryconfig("Open in TDK Checker", state=state)
        assert app._grid_menu.entrycget("Open in TDK Checker", "state") == 'normal'
    finally:
        app.destroy()


# =========================================================================
# Window reuse / recreate-if-destroyed / dataset switching
# =========================================================================

def test_reuses_existing_window_instead_of_duplicating():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        first_id = str(app._tdk_win)

        select_visible_row(app, 2)  # "geldi"
        app.open_tdk_checker_from_grid()
        app.update()
        assert str(app._tdk_win) == first_id
        assert app._tdk_token_var.get() == "geldi"  # reused window, new row loaded
    finally:
        app.destroy()


def test_recreates_window_after_manual_close():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        buttons = find_buttons_by_text(app._tdk_win)
        real_click(buttons["Close"])
        app.update()
        assert app._tdk_win is None

        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert app._tdk_win is not None
        assert app._tdk_token_var.get() == "cloudumuza"
    finally:
        app.destroy()


def test_dataset_switch_closes_tdk_window_and_clears_current_row():
    app = caa.App()
    try:
        ds1 = annotation_model.make_dataset("Data 1", "", two_sentence_blocks(), [])
        ds2 = annotation_model.make_dataset("Data 2", "", two_sentence_blocks(), [])
        app.datasets = [ds1, ds2]
        app._load_dataset_into_live(0)
        app.update()

        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert app._tdk_win is not None
        assert app._tdk_current is not None

        app._sync_active_dataset_from_live()
        app._load_dataset_into_live(1)
        app.update()

        assert app._tdk_win is None
        assert app._tdk_current is None
    finally:
        app.destroy()


def test_tdk_action_never_operates_on_previous_dataset():
    """After switching datasets, re-opening TDK Checker and applying a
    correction must only ever touch the NEWLY active dataset's blocks."""
    app = caa.App()
    try:
        ds1_blocks = two_sentence_blocks()
        ds2_blocks = two_sentence_blocks()
        ds1 = annotation_model.make_dataset("Data 1", "", ds1_blocks, [])
        ds2 = annotation_model.make_dataset("Data 2", "", ds2_blocks, [])
        app.datasets = [ds1, ds2]
        app._load_dataset_into_live(0)
        app.update()

        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()

        app._sync_active_dataset_from_live()
        app._load_dataset_into_live(1)
        app.update()

        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        app._tdk_root_var.set("cloud")
        app._tdk_segments_var.set("umuz + a")
        app._tdk_apply_correction()
        app.update()

        assert app.datasets[1]['blocks'][0][1]['tdk_segmentation']['root'] == "cloud"
        assert app.datasets[0]['blocks'][0][1].get('tdk_segmentation') is None  # dataset 1 untouched
    finally:
        app.destroy()


# =========================================================================
# Manual segmentation correction, Apply Correction, Undo,
# main-table synchronization, dirty-state
# =========================================================================

def test_manual_segmentation_correction_and_apply():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)

        assert app._dirty is False
        original_gloss = app.blocks[0][1]['gloss']
        app._tdk_root_var.set("cloud")
        app._tdk_segments_var.set("umuz + a")  # matches the task's example correction
        app._tdk_apply_correction()
        app.update()

        row = app.blocks[0][1]
        assert row['gloss'] == original_gloss  # Apply Correction never touches gloss
        assert row['tdk_segmentation'] == {'root': 'cloud', 'segments': ['umuz', 'a'],
                                            'source': 'manual', 'success': True}
        assert row['label'] == 'UID'  # never auto-changed
        assert app._dirty is True  # project dirty-state updated
    finally:
        app.destroy()


def test_apply_correction_never_touches_label_or_gloss():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        before_label = app.blocks[0][1]['label']
        before_gloss = app.blocks[0][1]['gloss']
        app._tdk_root_var.set("cloud")
        app._tdk_segments_var.set("umuz + a")
        app._tdk_apply_correction()
        app.update()
        assert app.blocks[0][1]['label'] == before_label
        assert app.blocks[0][1]['gloss'] == before_gloss
    finally:
        app.destroy()


def test_undo_restores_segmentation_only_never_gloss():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        original_gloss = app.blocks[0][1]['gloss']

        app._tdk_root_var.set("cloud")
        app._tdk_segments_var.set("umuz-a")
        app._tdk_apply_correction()
        app.update()
        assert app.blocks[0][1]['tdk_segmentation']['root'] == "cloud"

        buttons = find_buttons_by_text(app._tdk_win)
        real_click(buttons["Undo"])
        app.update()
        assert app.blocks[0][1].get('tdk_segmentation') is None
        assert app.blocks[0][1]['gloss'] == original_gloss  # undo never touches gloss
    finally:
        app.destroy()


def test_undo_with_empty_stack_is_safe_noop():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        app._tdk_undo_last()  # nothing applied yet -- must not raise
        app.update()
    finally:
        app.destroy()


def test_apply_correction_without_a_loaded_row_shows_message():
    app = make_app(two_sentence_blocks())
    try:
        app.open_tdk_checker_tool()  # standalone, no row loaded
        app.update()
        app._tdk_token_var.set("kitap")
        app._tdk_apply_correction()  # must not raise; shows an info message
        app.update()
    finally:
        app.destroy()


def test_reparse_button_reruns_parser():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        app._tdk_root_var.set("something-else")
        buttons = find_buttons_by_text(app._tdk_win)
        real_click(buttons["Re-parse"])
        app.update()
        assert app._tdk_root_var.get() == "cloud"  # re-derived from the token, not the stale manual edit
    finally:
        app.destroy()


# =========================================================================
# Find All Occurrences
# =========================================================================

def test_find_all_occurrences_lists_matches_in_active_dataset_only():
    blocks = two_sentence_blocks()
    blocks[1].append({"idx": 4, "token": "cloudumuza", "label": "UID", "gloss": ""})
    app = make_app(blocks)
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()

        app.open_tdk_find_occurrences()
        app.update()
        win = next(w for w in app.winfo_children()
                   if isinstance(w, tk.Toplevel) and 'Find All Occurrences' in w.title())
        tree = next(w for w in win.winfo_children()[0].winfo_children() if isinstance(w, caa.ttk.Treeview))
        assert len(tree.get_children()) == 2
        win.destroy()
    finally:
        app.destroy()


def test_find_all_occurrences_apply_to_selected_only_affects_active_dataset():
    ds1_blocks = two_sentence_blocks()
    ds1_blocks[1].append({"idx": 4, "token": "cloudumuza", "label": "UID", "gloss": ""})
    ds2_blocks = two_sentence_blocks()
    app = caa.App()
    try:
        ds1 = annotation_model.make_dataset("Data 1", "", ds1_blocks, [])
        ds2 = annotation_model.make_dataset("Data 2", "", ds2_blocks, [])
        app.datasets = [ds1, ds2]
        app._load_dataset_into_live(0)
        app.update()

        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        app._tdk_root_var.set("cloud")
        app._tdk_segments_var.set("umuz + a")

        app.open_tdk_find_occurrences()
        app.update()
        win = next(w for w in app.winfo_children()
                   if isinstance(w, tk.Toplevel) and 'Find All Occurrences' in w.title())
        tree = next(w for w in win.winfo_children()[0].winfo_children() if isinstance(w, caa.ttk.Treeview))
        tree.selection_set(tree.get_children())  # select all occurrences
        buttons = find_buttons_by_text(win)
        real_click(buttons["Apply to Selected"])
        app.update()
        win.destroy()

        # every occurrence in the ACTIVE dataset got the correction
        applied_count = sum(1 for blk in app.blocks for r in blk
                             if (r.get('tdk_segmentation') or {}).get('root') == 'cloud')
        assert applied_count == 2
        # never leaked into the other (inactive) dataset, and gloss is untouched
        assert all(r.get('tdk_segmentation') is None for blk in ds2['blocks'] for r in blk)
        assert all(r.get('gloss', '') == '' for blk in app.blocks for r in blk)
    finally:
        app.destroy()


def test_find_all_occurrences_no_query_shows_message():
    app = make_app(two_sentence_blocks())
    try:
        app.open_tdk_checker_tool()
        app.update()
        app._tdk_token_var.set('')
        app.open_tdk_find_occurrences()  # must not raise
        app.update()
    finally:
        app.destroy()


# =========================================================================
# Stale asynchronous response / stale-window prevention / no GUI freeze
# =========================================================================

def test_stale_response_is_dropped_when_a_newer_lookup_started():
    app = make_app(two_sentence_blocks())
    try:
        slow_then_fast = dp.MockDictionaryProvider(
            responses={"first": ("NOT_FOUND", []), "second": ("FOUND", [{"headword": "second"}])},
            delay_seconds=0.25)
        app._tdk_provider = slow_then_fast

        app.open_tdk_checker_tool()
        app.update()
        app._tdk_token_var.set("first")
        app._tdk_root_var.set("")
        app._tdk_segments_var.set("")
        app._tdk_check_all()  # starts a slow lookup for "first"
        app.update()

        # immediately supersede it with a second, faster-completing lookup
        app._tdk_token_var.set("second")
        app._tdk_check_all()
        app.update()

        assert wait_for_tdk_idle(app, timeout=3)
        time.sleep(0.4)  # let the first (slow) request finish too, if it hasn't
        for _ in range(10):
            app.update()
            time.sleep(0.02)

        rows = [app._tdk_results_tree.item(i)['values'] for i in app._tdk_results_tree.get_children()]
        terms_shown = {r[0] for r in rows}
        # only "second"'s results are visible -- "first"'s late-arriving,
        # stale response must never have overwritten them.
        assert terms_shown == {"second"}
    finally:
        app.destroy()


def test_closing_window_during_slow_lookup_does_not_crash():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider(delay=0.3)
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()  # starts an auto lookup
        app.update()

        buttons = find_buttons_by_text(app._tdk_win)
        real_click(buttons["Close"])
        app.update()
        assert app._tdk_win is None

        # let the slow background lookup finish and try to deliver its
        # result to a now-destroyed window -- must not raise/crash.
        time.sleep(0.5)
        app.update()
    finally:
        app.destroy()


def test_no_gui_freeze_during_slow_lookup():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider(delay=0.3)
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()

        max_single_update = 0.0
        start = time.time()
        while app._tdk_status_var.get() in ("", "checking...") and time.time() - start < 3:
            t0 = time.time()
            app.update()
            max_single_update = max(max_single_update, time.time() - t0)
            time.sleep(0.01)

        # every individual update() call returned promptly -- the 0.3s
        # network delay happened entirely on the background thread, never
        # blocking the Tk main loop.
        assert max_single_update < 0.15
        assert app._tdk_status_var.get() not in ("", "checking...")
    finally:
        app.destroy()


# =========================================================================
# Offline mode
# =========================================================================

def test_offline_provider_reports_unavailable_never_crashes():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = dp.UnavailableProvider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)
        assert "UNAVAILABLE" in app._tdk_status_var.get()
    finally:
        app.destroy()


def test_rest_of_app_usable_without_ever_opening_tdk_checker():
    """The mere existence of the TDK Checker feature must never require
    network access for ordinary use of the rest of the app."""
    app = make_app(two_sentence_blocks())
    try:
        assert app._tdk_provider is None  # never constructed
        app.blocks[1][1]['gloss'] = 'x'
        app._rebuild_grid_from_model()
        app.update()
        assert app._tdk_provider is None  # still never constructed
    finally:
        app.destroy()


# =========================================================================
# Parser bug regressions, exercised through the real GUI integration (the
# app's own lexicon-aware annotator, via _ensure_tdk_parser_annotator).
# =========================================================================

def test_grid_open_parses_filmin_as_film_plus_in():
    blocks = two_sentence_blocks()
    blocks[0].insert(2, {"idx": "", "token": "filmin", "label": "UID", "gloss": ""})
    app = make_app(blocks)
    try:
        select_visible_row(app, 2)  # the "filmin" row just inserted
        app.open_tdk_checker_tool()
        bidx, ridx, vis_r, err = app._resolve_tdk_grid_selection()
        assert err is None
        app._tdk_load_row(bidx, ridx, vis_r, auto_run=False)
        app.update()
        assert app._tdk_token_var.get() == "filmin"
        assert app._tdk_root_var.get() == "film"
        assert app._tdk_segments_var.get() == "in"
    finally:
        app.destroy()


def test_grid_open_parses_surdu_as_sur_plus_du():
    blocks = two_sentence_blocks()
    blocks[0].insert(2, {"idx": "", "token": "sürdü", "label": "UID", "gloss": ""})
    app = make_app(blocks)
    try:
        app.open_tdk_checker_tool()
        bidx, ridx = 0, 2
        app._tdk_load_row(bidx, ridx, None, auto_run=False)
        app.update()
        assert app._tdk_root_var.get() == "sür"
        assert app._tdk_segments_var.get() == "dü"
    finally:
        app.destroy()


# =========================================================================
# Editable root/segments + query snapshot + staleness (STALE_RESULT)
# =========================================================================

def test_check_tdk_uses_current_edited_root_and_segments_not_stale_parser_values():
    app = make_app(two_sentence_blocks())
    try:
        provider = dp.MockDictionaryProvider(
            responses={"cloudumuza": "NOT_FOUND", "edited-root": "FOUND", "umuz": "NOT_FOUND", "a": "NOT_FOUND"})
        app._tdk_provider = provider
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)

        # user manually edits the root AFTER the automatic parse/lookup
        app._tdk_root_var.set("edited-root")
        app._tdk_segments_var.set("")
        app._tdk_check_all()
        app.update()
        assert wait_for_tdk_idle(app)

        assert "edited-root" in provider.calls
        rows = [app._tdk_results_tree.item(i)['values'] for i in app._tdk_results_tree.get_children()]
        terms_shown = {r[0] for r in rows}
        assert "edited-root" in terms_shown
    finally:
        app.destroy()


def test_editing_root_after_check_marks_results_stale():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)
        assert app._tdk_results_stale is False

        app._tdk_root_var.set("something-completely-different")
        app.update()
        assert app._tdk_results_stale is True
        rows = [app._tdk_results_tree.item(i)['values'] for i in app._tdk_results_tree.get_children()]
        assert rows  # old results still shown...
        assert all(r[2] == dp.STATUS_STALE_RESULT for r in rows)  # ...but marked stale
    finally:
        app.destroy()


def test_editing_segments_after_check_marks_results_stale():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)

        app._tdk_segments_var.set("x + y")
        app.update()
        assert app._tdk_results_stale is True
    finally:
        app.destroy()


def test_re_running_check_tdk_clears_staleness():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)

        app._tdk_root_var.set("cloud")  # same value re-typed -- still a "change" event, harmless
        app._tdk_segments_var.set("umuz + a")
        app.update()

        app._tdk_check_all()
        app.update()
        assert wait_for_tdk_idle(app)
        assert app._tdk_results_stale is False
        rows = [app._tdk_results_tree.item(i)['values'] for i in app._tdk_results_tree.get_children()]
        assert all(r[2] != dp.STATUS_STALE_RESULT for r in rows)
    finally:
        app.destroy()


def test_reparse_never_overwrites_a_manual_correction_unless_clicked():
    """Typing into Root/Segments must never itself trigger a re-parse --
    only the explicit Re-parse button (or loading a fresh row) does."""
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()

        app._tdk_root_var.set("my-manual-correction")
        app.update()
        assert app._tdk_root_var.get() == "my-manual-correction"  # untouched by any auto re-parse
    finally:
        app.destroy()


# =========================================================================
# Explanation panel: Token/Root/Suffix/Analysis/Status
# =========================================================================

def test_explanation_panel_shows_token_root_suffix_analysis_status():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)  # "cloudumuza"
        app.open_tdk_checker_from_grid()
        app.update()
        text = app._tdk_explanation_text.get('1.0', 'end')
        assert "Token: cloudumuza" in text
        assert "Root: cloud" in text
        assert "Suffix: -umuz" in text
        assert "Suffix: -a" in text
        assert "Analysis:" in text
        assert "Status:" in text
    finally:
        app.destroy()


def test_explanation_panel_never_claims_a_bare_letter_is_a_suffix_without_classification():
    blocks = two_sentence_blocks()
    blocks[0].insert(2, {"idx": "", "token": "sürdü", "label": "UID", "gloss": ""})
    app = make_app(blocks)
    try:
        app.open_tdk_checker_tool()
        app._tdk_load_row(0, 2, None, auto_run=False)
        app.update()
        text = app._tdk_explanation_text.get('1.0', 'end')
        assert "Suffix: -d\n" not in text and "Suffix: -ü\n" not in text
        assert "Suffix: -dü" in text
    finally:
        app.destroy()


def test_explanation_panel_marks_itself_stale_after_manual_edit():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        app._tdk_root_var.set("totally-different")
        app.update()
        text = app._tdk_explanation_text.get('1.0', 'end')
        assert "edited since" in text.lower()
    finally:
        app.destroy()


# =========================================================================
# Rich dictionary detail panel
# =========================================================================

def test_detail_panel_shows_full_entry_fields_for_selected_result():
    app = make_app(two_sentence_blocks())
    try:
        entry = dp.DictionaryEntry(
            headword="cloud", part_of_speech="isim", origin="İngilizce",
            senses=(dp.DictionarySense(definition="bulut", usage_labels=("argo",)),),
            compounds=("cloud computing",),
        )
        provider = dp.MockDictionaryProvider(
            responses={"cloudumuza": "NOT_FOUND", "cloud": ("FOUND", [entry]),
                       "umuz": "NOT_FOUND", "a": "NOT_FOUND"})
        app._tdk_provider = provider
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)

        # select the "cloud" (root) row in the results tree
        target_iid = next(
            i for i in app._tdk_results_tree.get_children()
            if app._tdk_results_tree.item(i)['values'][0] == 'cloud')
        app._tdk_results_tree.selection_set(target_iid)
        app._tdk_results_tree.event_generate('<<TreeviewSelect>>')
        app.update()

        detail = app._tdk_detail_text.get('1.0', 'end')
        assert "Headword: cloud" in detail
        assert "Part of speech: isim" in detail
        assert "Origin: İngilizce" in detail
        assert "bulut" in detail
        assert "Usage: argo" in detail
        assert "cloud computing" in detail
        assert "Query: cloud" in detail
    finally:
        app.destroy()


def test_detail_panel_shows_not_provided_for_missing_fields():
    app = make_app(two_sentence_blocks())
    try:
        entry = dp.DictionaryEntry(headword="cloud", senses=(dp.DictionarySense(definition="bulut"),))
        provider = dp.MockDictionaryProvider(
            responses={"cloudumuza": "NOT_FOUND", "cloud": ("FOUND", [entry]),
                       "umuz": "NOT_FOUND", "a": "NOT_FOUND"})
        app._tdk_provider = provider
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)

        target_iid = next(
            i for i in app._tdk_results_tree.get_children()
            if app._tdk_results_tree.item(i)['values'][0] == 'cloud')
        app._tdk_results_tree.selection_set(target_iid)
        app._tdk_results_tree.event_generate('<<TreeviewSelect>>')
        app.update()

        detail = app._tdk_detail_text.get('1.0', 'end')
        assert f"Origin: {dp.NOT_PROVIDED}" in detail
        assert f"Pronunciation: {dp.NOT_PROVIDED}" in detail
        assert f"Part of speech: {dp.NOT_PROVIDED}" in detail
        assert f"Compounds: {dp.NOT_PROVIDED}" in detail
    finally:
        app.destroy()


def test_detail_panel_shows_placeholder_before_any_selection():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = mock_provider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        detail = app._tdk_detail_text.get('1.0', 'end').strip()
        assert detail  # non-empty placeholder text
    finally:
        app.destroy()


# =========================================================================
# NOT_FOUND never surfaces raw Turkish backend text (e.g. "Sonuç bulunamadı")
# =========================================================================

class _FixedMessageProvider(dp.DictionaryProvider):
    """Stands in for the real TDKProvider's own NOT_FOUND behavior (see
    dictionary_provider.py's TDKProvider._parse_response), which always
    uses this exact fixed English message -- never the raw Turkish
    "Sonuç bulunamadı" the upstream endpoint itself returns."""
    name = "fixed"

    def _do_lookup(self, term, normalized):
        return dp.LookupResult(query=term, normalized_query=normalized, status=dp.STATUS_NOT_FOUND,
                                source=self.name, message="no dictionary entry found")


def test_not_found_status_display_never_shows_raw_turkish_backend_text():
    app = make_app(two_sentence_blocks())
    try:
        app._tdk_provider = _FixedMessageProvider()
        select_visible_row(app, 1)
        app.open_tdk_checker_from_grid()
        app.update()
        assert wait_for_tdk_idle(app)
        rows = [app._tdk_results_tree.item(i)['values'] for i in app._tdk_results_tree.get_children()]
        assert rows
        for r in rows:
            detail = str(r[3])
            assert "bulunamad" not in detail.lower()
            assert "sonuç" not in detail.lower() and "sonuc" not in detail.lower()
            assert "no dictionary entry found" in detail
    finally:
        app.destroy()
