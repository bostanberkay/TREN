# cs_annotator_app.py

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import os
import re
import csv
import json
import sys
import copy
import queue
import threading

from cs_pipeline import Annotator, DEFAULTS
import annotation_model
import reranker_integration
import confidence
import dictionary_provider
import tdk_parser

try:
    import tksheet
except Exception:
    tksheet = None

DARK_BG = "#222222"
DARK_FG = "#e6e6e6"

ACCENT  = "#3a7bd5"

APP_DIR = os.path.join(os.path.expanduser("~"), ".cs_annotator")
LAST_PROJECT_PTR = os.path.join(APP_DIR, "last_project.json")
PROJECT_EXT = ".trenproj"


class App(tk.Tk):
    def _set_runtime_workdir(self):
        """
        Ensure relative resource paths resolve correctly.

        - In dev: set CWD to the directory containing this file.
        - In PyInstaller (frozen): set CWD to the bundle extraction dir (sys._MEIPASS).

        This prevents errors like:
            [Errno 2] No such file or directory: 'frequent_tr_words.txt'
        when downstream code uses relative opens.
        """
        try:
            if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                base = sys._MEIPASS
            else:
                base = os.path.dirname(os.path.abspath(__file__))

            # If a bundled resources folder exists, prefer it so relative opens work
            res_dir = os.path.join(base, "resources")
            if os.path.isdir(res_dir):
                os.chdir(res_dir)
            elif base and os.path.isdir(base):
                os.chdir(base)
        except Exception:
            pass
    _LEIPZIG_APPENDIX_TEXT = """Appendix: List of Standard Abbreviations
1 first person
2 second person
3 third person
A agent-like argument of canonical transitive verb
ABL ablative
ABS absolutive
ACC accusative
ADJ adjective
ADV adverb(ial)
AGR agreement
ALL allative
ANTIP antipassive
APPL applicative
ART article
AUX auxiliary
BEN benefactive
CAUS causative
CLF classifier
COM comitative
COMP complementizer
COMPL completive
COND conditional
COP copula
CVB converb
DAT dative
DECL declarative
DEF definite
DEM demonstrative
DET determiner
DIST distal
DISTR distributive
DU dual
DUR durative
ERG ergative
EXCL exclusive
F feminine
FOC focus
FUT future
GEN genitive
IMP imperative
INCL inclusive
IND indicative
INDF indefinite
INF infinitive
INS instrumental
INTR intransitive
IPFV imperfective
IRR irrealis
LOC locative
M masculine
N neuter
N- non- (e.g. NSG nonsingular, NPST nonpast)
NEG negation, negative
NMLZ nominalizer/nominalization
NOM nominative
OBJ object
OBL oblique
P patient-like argument of canonical transitive verb
PASS passive
PFV perfective
PL plural
POSS possessive
PRED predicative
PRF perfect
PRS present
PROG progressive
PROH prohibitive
PROX proximal/proximate
PST past
PTCP participle
PURP purposive
Q question particle/marker
QUOT quotative
RECP reciprocal
REFL reflexive
REL relative
RES resultative
S single argument of canonical intransitive verb
SBJ subject
SBJV subjunctive
SG singular
TOP topic
TR transitive
VOC vocative
"""
    # Auto-Glossing Tool
    def _collect_mixed_rows(self):
        """Collect MIXED rows from the current model, in visible order."""
        items = []
        if not getattr(self, 'blocks', None):
            return items

        def sent_id_for_block(bi: int):
            try:
                blk = self.blocks[bi]
            except Exception:
                return ""
            for rr in blk:
                try:
                    if str(rr.get('token', '')).strip() == 'SentenceID':
                        v = str(rr.get('label', '') or rr.get('gloss', '') or '').strip()
                        return v
                except Exception:
                    continue
            return ""

        for vis_r, bidx, ridx, row in annotation_model.iter_visible_rows(
            self.blocks, getattr(self, '_row_index_map', {}), getattr(self, '_sep_rows', set())
        ):
            tok = str(row.get('token', '') or '').strip()
            if self._is_meta_row_token(tok):
                continue

            lab = str(row.get('label', '') or '').strip()
            if lab.upper() == 'MIXED':
                items.append({
                    'vis_r': vis_r,
                    'bidx': bidx,
                    'ridx': ridx,
                    'sent_id': sent_id_for_block(bidx),
                })
        return items

    def open_auto_glossing_tool(self):
        """Open Auto-Glossing Tool window (UI skeleton)."""
        if getattr(self, '_ag_win', None) is not None and self._ag_win.winfo_exists():
            try:
                self._ag_win.deiconify()
                self._ag_win.lift()
            except Exception:
                pass
            return

        self._ag_items = []
        self._ag_i = 0

        win = tk.Toplevel(self)
        win.title('Auto-Glossing Tool')
        win.geometry('820x420')
        win.configure(bg=DARK_BG)
        win.transient(self)

        outer = ttk.Frame(win, style='Dark.TFrame')
        outer.pack(fill='both', expand=True, padx=10, pady=10)

        # progress left, sentence id right
        top = ttk.Frame(outer, style='Dark.TFrame')
        top.pack(fill='x', pady=(0, 10))

        self._ag_status_var = tk.StringVar(value='0/0')
        self._ag_sent_var = tk.StringVar(value='SentenceID: ')

        ttk.Label(top, textvariable=self._ag_status_var, style='Dark.TLabel').pack(side='left')
        ttk.Label(top, textvariable=self._ag_sent_var, style='Dark.TLabel').pack(side='right')

        # Content area
        body = ttk.Frame(outer, style='Dark.TFrame')
        body.pack(fill='both', expand=True)

        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_columnconfigure(2, weight=1)

        card = ttk.Frame(body, style='Dark.TFrame')
        card.grid(row=0, column=1, sticky='n')

        # Make the card content symmetric
        card.grid_columnconfigure(0, weight=0)
        card.grid_columnconfigure(1, weight=1)

        # Item
        ttk.Label(card, text='Item', style='Dark.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 6))
        self._ag_item_var = tk.StringVar(value='')
        ent_item = ttk.Entry(card, textvariable=self._ag_item_var, style='Dark.TEntry', width=64, state='readonly')
        ent_item.grid(row=0, column=1, sticky='we', padx=(12, 0), pady=(0, 6))

        # Gloss
        ttk.Label(card, text='Gloss', style='Dark.TLabel').grid(row=1, column=0, sticky='w', pady=(0, 6))
        self._ag_gloss_var = tk.StringVar(value='')
        ent_gls = ttk.Entry(card, textvariable=self._ag_gloss_var, style='Dark.TEntry', width=64)
        ent_gls.grid(row=1, column=1, sticky='we', padx=(12, 0), pady=(0, 6))

        # Auto-gloss button under Gloss (aligned with entries)
        btn_row = ttk.Frame(card, style='Dark.TFrame')
        btn_row.grid(row=2, column=1, sticky='w', padx=(12, 0), pady=(2, 12))
        btn_autogloss = ttk.Button(btn_row, text='Auto-Gloss', style='Dark.TButton', command=self._ag_auto_gloss_current)
        btn_autogloss.pack(side='left')

        # Separator
        ttk.Separator(card, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='we', pady=(6, 10))

        # Label row
        ttk.Label(card, text='Label', style='Dark.TLabel').grid(row=4, column=0, sticky='w', pady=(0, 6))
        self._ag_label_var = tk.StringVar(value='')
        ent_lab = ttk.Entry(card, textvariable=self._ag_label_var, style='Dark.TEntry', width=18)
        ent_lab.grid(row=4, column=1, sticky='w', padx=(12, 0), pady=(0, 6))

        # Label buttons (grid, symmetric)
        lblbtns = ttk.Frame(card, style='Dark.TFrame')
        lblbtns.grid(row=5, column=0, columnspan=2, sticky='we', pady=(4, 0))

        labels = ["TR", "EN", "MIXED", "UID", "NE", "LANG3", "OTHER"]
        for i, lab in enumerate(labels):
            r = i // 4
            c = i % 4
            b = ttk.Button(lblbtns, text=lab, style='Dark.TButton', width=10,
                           command=lambda L=lab: self._ag_set_label(L))
            b.grid(row=r, column=c, padx=4, pady=4, sticky='we')

        for c in range(4):
            lblbtns.grid_columnconfigure(c, weight=1)

        # Appendix button
        ttk.Button(card, text='Leipzig Gloss Appendix', style='Dark.TButton',
                   command=self._open_leipzig_appendix).grid(row=6, column=0, columnspan=2, sticky='we', pady=(10, 0))

        # prev left, next right
        nav = ttk.Frame(outer, style='Dark.TFrame')
        nav.pack(fill='x', pady=(12, 0))
        ttk.Button(nav, text='◀', style='Dark.TButton', width=4, command=self._ag_prev).pack(side='left')
        ttk.Button(nav, text='▶', style='Dark.TButton', width=4, command=self._ag_next).pack(side='right')

        def _on_close():
            try:
                self._ag_commit_current_to_model()
                win.destroy()
            except Exception:
                pass
            finally:
                self._ag_win = None
        # shortcuts
        try:
            # Auto-gloss
            win.bind('<Command-Return>', lambda e: (self._ag_auto_gloss_current(), 'break'))
            win.bind('<Control-Return>', lambda e: (self._ag_auto_gloss_current(), 'break'))
            # Navigation
            win.bind('<Left>', lambda e: (self._ag_prev(), 'break'))
            win.bind('<Right>', lambda e: (self._ag_next(), 'break'))
        except Exception:
            pass

        win.protocol('WM_DELETE_WINDOW', _on_close)
        self._ag_win = win
        self._ag_refresh_items()

    def _open_leipzig_appendix(self):
        # reuse window if already open
        if getattr(self, '_ag_appendix_win', None) is not None:
            try:
                if self._ag_appendix_win.winfo_exists():
                    self._ag_appendix_win.deiconify()
                    self._ag_appendix_win.lift()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title('Leipzig Gloss Appendix')
        win.geometry('720x520')
        win.configure(bg=DARK_BG)
        win.transient(self)

        frm = tk.Frame(win, bg=DARK_BG)
        frm.pack(fill='both', expand=True, padx=10, pady=10)

        txt = ScrolledText(frm, wrap='none', bg='#1b1b1b', fg=DARK_FG, insertbackground='white')
        txt.pack(fill='both', expand=True)
        try:
            txt.configure(font=('Menlo', 12))
        except Exception:
            pass

        txt.insert('1.0', self._LEIPZIG_APPENDIX_TEXT)
        txt.configure(state='disabled')

        def _on_close():
            try:
                win.destroy()
            except Exception:
                pass
            finally:
                self._ag_appendix_win = None

        win.protocol('WM_DELETE_WINDOW', _on_close)
        self._ag_appendix_win = win

    # Navigation and sync edit helpers
    def _ag_prev(self):
        if not self._ag_items:
            return
        if self._ag_i > 0:
            self._ag_commit_current_to_model()
            self._ag_i -= 1
            self._ag_load_current()

    def _ag_next(self):
        if not self._ag_items:
            return
        j = self._ag_i + 1
        while j < len(self._ag_items):
            it = self._ag_items[j]
            try:
                row = self.blocks[it['bidx']][it['ridx']]
                if str(row.get('label', '')).upper() == 'MIXED':
                    self._ag_commit_current_to_model()
                    self._ag_i = j
                    self._ag_load_current()
                    return
            except Exception:
                pass
            j += 1

    def _ag_set_label(self, lab: str):
        try:
            self._ag_label_var.set(str(lab))
        except Exception:
            pass

    def _ag_commit_current_to_model(self):
        """Commit the current tool fields into the main model + grid.
        This runs only when navigating away (or on close).
        """
        if not getattr(self, '_ag_items', None):
            return
        it = self._ag_items[self._ag_i]
        bidx, ridx, vis_r = it['bidx'], it['ridx'], it['vis_r']
        try:
            row = self.blocks[bidx][ridx]
        except Exception:
            return

        new_label = self._ag_label_var.get()
        new_gloss = self._ag_gloss_var.get()
        if row.get('label', '') != new_label or row.get('gloss', '') != new_gloss:
            self._mark_dirty()
        row['label'] = new_label
        row['gloss'] = new_gloss

        try:
            if getattr(self, 'sheet', None) is not None:
                self.sheet.set_cell_data(vis_r, 2, row['label'])
                self.sheet.set_cell_data(vis_r, 3, row['gloss'])
                if hasattr(self.sheet, 'refresh'):
                    self.sheet.refresh()
        except Exception:
            pass

    # Auto-Gloss handler for current item
    def _ag_auto_gloss_current(self):
        if not self._ag_items:
            return
        it = self._ag_items[self._ag_i]
        try:
            row = self.blocks[it['bidx']][it['ridx']]
        except Exception:
            return
        if str(row.get('label', '')).upper() != 'MIXED':
            self.bell()
            return
        gloss = self._auto_gloss_mixed_token(row.get('token', ''))
        if gloss:
            self._ag_gloss_var.set(gloss)

    def _ag_update_status(self):
        n = len(getattr(self, '_ag_items', []) or [])
        i = getattr(self, '_ag_i', 0)
        try:
            lab = ''
            if getattr(self, '_ag_items', None):
                it = self._ag_items[self._ag_i]
                row = self.blocks[it['bidx']][it['ridx']]
                lab = str(row.get('label', '')).upper()
            self._ag_status_var.set(f"{lab} {i + 1}/{n}" if n else "0/0")
        except Exception:
            pass

    def _ag_refresh_items(self):
        try:
            self._ag_items = self._collect_mixed_rows() or []
        except Exception:
            self._ag_items = []
        self._ag_i = 0

        if not self._ag_items:
            try:
                self._ag_item_var.set("")
                self._ag_label_var.set("")
                self._ag_gloss_var.set("")
                self._ag_sent_var.set("SentenceID: ")
                self._ag_update_status()
            except Exception:
                pass
            messagebox.showinfo("Auto-Glossing Tool", "No MIXED items found.")
            return

        self._ag_load_current()

    def _ag_load_current(self):
        if not getattr(self, '_ag_items', None):
            return

        it = self._ag_items[self._ag_i]
        bidx, ridx, vis_r = it['bidx'], it['ridx'], it['vis_r']

        try:
            row = self.blocks[bidx][ridx]
        except Exception:
            return

        try:
            self._ag_item_var.set(str(row.get('token', '') or ''))
            self._ag_label_var.set(str(row.get('label', '') or ''))
            self._ag_gloss_var.set(str(row.get('gloss', '') or ''))
            sid = it.get('sent_id', '') or ''
            self._ag_sent_var.set(f"SentenceID: {sid}" if sid else "SentenceID: ")
        except Exception:
            pass

        # sync selection in main grid
        try:
            if getattr(self, 'sheet', None) is not None:
                self._cancel_edit_if_any()
                self.sheet.select_cell(vis_r, 2)
                self.sheet.see(vis_r, 2)
        except Exception:
            pass

        self._ag_update_status()

    # =====================================================================
    # Confidence Review Tool
    #
    # A focused, sequential review window for uncertain/low-confidence
    # tokens across any of the 7 schema labels (not UID-only -- see
    # confidence.py), filterable by label and by confidence band. Operates
    # on the exact same self.blocks / self._row_index_map / self._sep_rows
    # model the main table uses -- it never keeps an independent copy of
    # the corpus. Deliberately minimal beyond that filtering: no bulk
    # editing, no remembered/learned-correction system. Selecting a row,
    # applying a label, and undoing all synchronize immediately with the
    # main table through the same _rebuild_grid_from_model / set_cell_data
    # mirroring the main sheet edit handlers already use, so Save and
    # Export automatically stay in sync with zero extra wiring.
    # =====================================================================

    ALL_LABELS = ("TR", "EN", "MIXED", "UID", "NE", "OTHER", "LANG3")

    # Confidence bands the review tool's checkbox-based filter can restrict
    # to (see confidence.py). An empty self._review_band_filter set means
    # "no restriction".
    REVIEW_BANDS = (confidence.BAND_HIGH, confidence.BAND_MEDIUM, confidence.BAND_LOW)

    # View-combobox presets. "All Uncertain" is the tool's default (see
    # _review_view_mode/_uid_compute_items): every token, any label, whose
    # confidence record has review_recommended=True. "UID Only" reproduces
    # the tool's original pre-confidence-layer default (label filter
    # {"UID"}, no band restriction). "Custom" means the label/band
    # checkboxes below are in direct control (set automatically whenever
    # one of them is touched directly).
    REVIEW_VIEW_ALL_UNCERTAIN = "All Uncertain"
    REVIEW_VIEW_UID_ONLY = "UID Only"
    REVIEW_VIEW_CUSTOM = "Custom"
    REVIEW_VIEWS = (REVIEW_VIEW_ALL_UNCERTAIN, REVIEW_VIEW_UID_ONLY, REVIEW_VIEW_CUSTOM)

    def _update_block_matrix_embed(self, bidx):
        """Recompute MatrixLang/EmbedLang for block `bidx` using the exact
        same deterministic rule Annotator._decide_matrix_embed already
        implements. That method is a pure function of (labels, cfg) with no
        dependency on `self` (no lexicon/fastText/Stanza access), so it is
        called unbound here -- this never instantiates a real Annotator and
        never touches the production pipeline."""
        try:
            block = self.blocks[bidx]
        except Exception:
            return
        labels = [r.get('label', '') for r in block if not self._is_meta_row_token(r.get('token', ''))]
        matrix, embed = Annotator._decide_matrix_embed(None, labels, self.cfg)
        for r in block:
            tok = str(r.get('token', '') or '').strip()
            if tok == 'MatrixLang':
                r['label'] = matrix
            elif tok == 'EmbedLang':
                r['label'] = embed

    def _uid_sentence_text(self, bidx):
        """Full-sentence context: every non-meta token in the block, in order."""
        try:
            blk = self.blocks[bidx]
        except Exception:
            return ""
        toks = [str(r.get('token', '') or '') for r in blk if not self._is_meta_row_token(r.get('token', ''))]
        return " ".join(toks)

    def _uid_sentence_preview(self, bidx, max_len=40):
        text = self._uid_sentence_text(bidx)
        return text if len(text) <= max_len else text[:max_len - 1] + "\u2026"

    def _uid_jump_to_main(self, vis_r):
        """Select/reveal the corresponding token in the main table (and the
        Full Edit Window's sheet, if open). Deliberately does NOT move
        keyboard focus to the main sheet -- doing so would steal focus away
        from the Confidence Review Tool's own tree/controls after every selection
        change (click, arrow key, Next/Prev/First/Last), breaking repeated
        arrow-key navigation in the tool after the first press."""
        if vis_r is None:
            return
        for sh in (self.sheet, self._full_sheet):
            if sh is None:
                continue
            try:
                sh.select_cell(vis_r, 2)
                sh.see(vis_r, 2)
            except Exception:
                pass

    # --- scoped undo history (Apply only; no app-wide undo/redo exists) ---

    def _uid_push_undo(self, record):
        self._uid_undo_stack.append(record)
        if len(self._uid_undo_stack) > 200:
            self._uid_undo_stack.pop(0)

    def _uid_undo_last(self):
        if not self._uid_undo_stack:
            self.bell()
            return
        record = self._uid_undo_stack.pop()
        bidx, ridx = record['bidx'], record['ridx']
        try:
            self.blocks[bidx][ridx]['label'] = record['old']['label']
            self.blocks[bidx][ridx]['gloss'] = record['old']['gloss']
            if 'confidence' in record['old']:
                self.blocks[bidx][ridx]['confidence'] = record['old']['confidence']
            self.blocks[bidx][ridx]['reviewed'] = record['old'].get('reviewed', False)
        except Exception:
            pass
        self._update_block_matrix_embed(bidx)
        self._mark_dirty()
        self._rebuild_grid_from_model()
        if getattr(self, '_uid_win', None) is not None and self._uid_win.winfo_exists():
            self._uid_refresh_items(preserve_index=True)

    def _uid_on_structural_change(self):
        """Call this after ANY edit made outside the Confidence Review Tool that
        changes row positions or row count in self.blocks (currently:
        Merge Cells and Undo Merge Cells). self._uid_items and every entry
        on self._uid_undo_stack hold positional (bidx, ridx, vis_r)
        references; a structural edit elsewhere can silently invalidate
        them, and a later UID Apply/Undo could then target the wrong row
        or a row that no longer means what it did.

        The undo stack is always cleared here rather than remapped: its
        records point at specific (bidx, ridx) positions, and after rows
        have been merged/split/renumbered there is no reliable way to know
        whether "the same position" still means "the same edit" -- restoring
        an old (label, gloss) pair into a position that now holds a
        different, merged token would be a silent wrong-row mutation. This
        is a deliberate simplicity/safety tradeoff: the user loses the
        ability to undo a UID edit that happened before a merge, but can
        never have a merge silently corrupt an unrelated row via a stale
        undo record.

        _uid_items itself is fully recomputed (never patched in place), so
        it can never reference a stale ridx/vis_r. The previously selected
        token is re-selected by matching (bidx, token text) against the
        fresh list when possible; if that exact token no longer exists as
        a UID (e.g. it was one of the merged rows), the closest remaining
        item by list position is selected instead -- the same
        preserve_index fallback already used after every Apply/Undo.
        """
        self._uid_undo_stack = []

        if getattr(self, '_uid_win', None) is None or not self._uid_win.winfo_exists():
            return

        prev_bidx, prev_token = None, None
        if self._uid_items and 0 <= self._uid_i < len(self._uid_items):
            prev_it = self._uid_items[self._uid_i]
            prev_bidx = prev_it['bidx']
            try:
                prev_token = self.blocks[prev_it['bidx']][prev_it['ridx']].get('token', '')
            except Exception:
                prev_token = None

        old_i = self._uid_i
        self._uid_items = self._uid_compute_items()

        if self._uid_items:
            new_i = None
            if prev_token is not None:
                for i, it in enumerate(self._uid_items):
                    try:
                        cur_tok = self.blocks[it['bidx']][it['ridx']].get('token', '')
                    except Exception:
                        continue
                    if it['bidx'] == prev_bidx and cur_tok == prev_token:
                        new_i = i
                        break
            if new_i is None:
                new_i = max(0, min(old_i, len(self._uid_items) - 1))
            self._uid_i = new_i
        else:
            self._uid_i = 0

        self._uid_populate_tree()
        if self._uid_items:
            self._uid_load_current(sync_tree=True)
        else:
            self._uid_clear_editor()
            self._uid_update_title()

    # --- list computation / display ---

    def _uid_compute_items(self):
        if self._uid_mode == 'occurrences' and self._uid_query:
            return annotation_model.find_occurrences(
                self.blocks, self._row_index_map, self._sep_rows, self._uid_query)

        if self._review_view_mode == 'all_uncertain':
            # Default view: every token whose confidence record has
            # review_recommended=True (confidence.is_review_required),
            # across ALL 7 labels -- never defined by label name. A row
            # with no confidence record at all is excluded (no evidence
            # either way), never assumed uncertain.
            raw = annotation_model.collect_label_rows(
                self.blocks, self._row_index_map, self._sep_rows, set(self.ALL_LABELS))
            raw = [it for it in raw
                   if confidence.is_review_required(self.blocks[it['bidx']][it['ridx']])]
        else:
            labels = self._review_label_filter or {"UID"}
            raw = annotation_model.collect_label_rows(self.blocks, self._row_index_map, self._sep_rows, labels)
            # Confidence-band filter (requirement: "optionally show all
            # LOW-confidence or selected MEDIUM-confidence tokens"). An
            # empty self._review_band_filter means "no restriction" --
            # rows with no confidence data at all (legacy projects, or
            # rows never re-annotated since this layer was added) are then
            # included rather than silently dropped, matching the tool's
            # original unfiltered-by-confidence behavior.
            if self._review_band_filter:
                raw = [it for it in raw
                       if confidence.band_of(self.blocks[it['bidx']][it['ridx']]) in self._review_band_filter]
        if self._review_hide_reviewed:
            raw = [it for it in raw
                   if not confidence.is_reviewed(self.blocks[it['bidx']][it['ridx']])]
        if self._uid_search_var is not None:
            needle = (self._uid_search_var.get() or "").strip().casefold()
            if needle:
                raw = [it for it in raw
                       if needle in str(self.blocks[it['bidx']][it['ridx']].get('token', '') or '').casefold()]
        return raw

    def _uid_refresh_items(self, preserve_index=True):
        old_i = getattr(self, '_uid_i', 0)
        self._uid_items = self._uid_compute_items()
        if preserve_index and self._uid_items:
            self._uid_i = max(0, min(old_i, len(self._uid_items) - 1))
        else:
            self._uid_i = 0

        if getattr(self, '_uid_win', None) is None or not self._uid_win.winfo_exists():
            return

        self._uid_populate_tree()
        if self._uid_items:
            self._uid_load_current(sync_tree=True)
        else:
            self._uid_clear_editor()
            self._uid_update_title()

    def _uid_populate_tree(self):
        tree = self._uid_tree
        tree.delete(*tree.get_children())
        for i, it in enumerate(self._uid_items):
            bidx, ridx = it['bidx'], it['ridx']
            row = self.blocks[bidx][ridx]
            conf = confidence.get_confidence(row)
            band = conf.get('confidence_band', '-') if conf else '-'
            score = f"{conf['confidence_score']:.2f}" if conf else '-'
            reviewed = 'Yes' if confidence.is_reviewed(row) else 'No'
            tree.insert("", "end", iid=str(i), values=(
                self._uid_sentence_preview(bidx), row.get('token', ''),
                row.get('label', ''), row.get('gloss', '') or '',
                band, score, reviewed,
            ))

    def _uid_clear_editor(self):
        self._uid_label_var.set("")
        self._uid_gloss_var.set("")
        self._uid_context_var.set("")
        self._uid_evidence_var.set("")

    def _uid_evidence_text(self, row):
        """Human-readable confidence/evidence block for the currently
        selected row (requirement: "show current label, confidence,
        reasons, and evidence"). Never fabricates data for a row with no
        confidence record (legacy project, or a manual edit -- see
        confidence.note_manual_edit) -- shows a plain "not available" note
        instead.

        Deliberately does NOT include the record's calibration_note (the
        "NOT statistically calibrated against a ... gold-labeled
        confidence corpus" disclosure) -- that is a fixed, dataset-level
        disclaimer about the scoring METHOD as a whole, not per-token
        evidence, and showing it here read as a stray corpus-analysis/
        status line rather than something about the selected token. The
        disclosure itself is untouched in confidence.py/to_dict() and still
        persisted -- only this one display was cleaned up."""
        conf = confidence.get_confidence(row)
        if not conf:
            return "Confidence: not available for this row."
        lines = [
            f"Label: {conf.get('final_label') or row.get('label') or '-'}"
            f"  |  Confidence: {conf.get('confidence_band')} ({conf.get('confidence_score')})"
            f"  |  Reviewed: {'Yes' if confidence.is_reviewed(row) else 'No'}"
            f"  |  Rule-based label: {conf.get('rule_based_label') or '-'}"
            f"  |  Promoted by: {conf.get('promoted_by') or '-'}",
        ]
        reasons = conf.get('uncertainty_reasons') or []
        if reasons:
            lines.append("Uncertainty reasons: " + "; ".join(reasons))
        evidence = conf.get('evidence_summary') or []
        if evidence:
            lines.append("Evidence: " + "; ".join(evidence))
        return "\n".join(lines)

    def _uid_update_title(self):
        n = len(self._uid_items)
        i = (self._uid_i + 1) if n else 0
        if self._uid_mode == 'occurrences':
            mode_word = "occurrence"
        elif self._review_view_mode == 'all_uncertain':
            mode_word = "uncertain"
        elif self._review_label_filter == {"UID"}:
            mode_word = "UID"
        else:
            mode_word = "item"
        self._uid_win.title(f"Confidence Review Tool \u2014 {mode_word} {i} of {n}" if n else "Confidence Review Tool \u2014 0 of 0")

    # --- navigation ---

    def _uid_on_tree_select(self, event=None):
        sel = self._uid_tree.selection()
        if not sel:
            return
        try:
            i = int(sel[0])
        except Exception:
            return
        if 0 <= i < len(self._uid_items):
            self._uid_i = i
            self._uid_load_current(sync_tree=False)

    def _uid_on_tree_double_click(self, event=None):
        # Focus the label dropdown for editing; must NOT change any value.
        self._uid_on_tree_select(event)
        try:
            self._uid_label_combo.focus_set()
        except Exception:
            pass

    def _uid_goto(self, i):
        if not self._uid_items:
            return
        self._uid_i = max(0, min(i, len(self._uid_items) - 1))
        self._uid_load_current(sync_tree=True)

    def _uid_prev(self):
        self._uid_goto(self._uid_i - 1)

    def _uid_next(self):
        self._uid_goto(self._uid_i + 1)

    def _uid_first(self):
        self._uid_goto(0)

    def _uid_last(self):
        self._uid_goto(len(self._uid_items) - 1)

    def _uid_load_current(self, sync_tree=True):
        if not self._uid_items:
            self._uid_clear_editor()
            self._uid_update_title()
            return
        it = self._uid_items[self._uid_i]
        bidx, ridx, vis_r = it['bidx'], it['ridx'], it['vis_r']
        try:
            row = self.blocks[bidx][ridx]
        except Exception:
            return

        self._uid_label_var.set(row.get('label', '') or '')
        self._uid_gloss_var.set(row.get('gloss', '') or '')
        self._uid_context_var.set(self._uid_sentence_text(bidx))
        self._uid_evidence_var.set(self._uid_evidence_text(row))

        if sync_tree:
            try:
                self._uid_tree.selection_set(str(self._uid_i))
                self._uid_tree.see(str(self._uid_i))
                self._uid_tree.focus(str(self._uid_i))
                # Navigation reached here via First/Previous/Next/Last (not
                # a direct click/keypress on the tree itself), so real
                # keyboard focus is currently on the button just clicked.
                # Return it to the tree so arrow keys keep working right
                # after using a nav button, without requiring an extra click.
                self._uid_tree.focus_set()
            except Exception:
                pass

        self._uid_jump_to_main(vis_r)
        self._uid_update_title()

    # --- editing ---

    def _uid_apply(self):
        if not self._uid_items:
            return
        it = self._uid_items[self._uid_i]
        bidx, ridx, vis_r = it['bidx'], it['ridx'], it['vis_r']
        row = self.blocks[bidx][ridx]
        tok = row.get('token', '')
        new_label = self._uid_label_var.get()
        new_gloss = self._uid_gloss_var.get()

        if annotation_model.is_matrixembed_locked(tok, new_label):
            self.bell()
            messagebox.showwarning("Confidence Review Tool", "MatrixLang/EmbedLang rows may only be set to TR or EN.")
            return

        old_label, old_gloss = row.get('label', ''), row.get('gloss', '')
        if old_label != new_label or old_gloss != new_gloss:
            self._uid_push_undo({
                'bidx': bidx, 'ridx': ridx,
                'old': {'label': old_label, 'gloss': old_gloss,
                        'confidence': row.get('confidence'), 'reviewed': row.get('reviewed', False)},
                'new': {'label': new_label, 'gloss': new_gloss},
            })
            row['label'] = new_label
            row['gloss'] = new_gloss
            # The confidence record computed for old_label now describes a
            # label this row no longer carries -- replace it with a manual-
            # edit marker rather than leaving stale/misleading evidence
            # text around (see confidence.note_manual_edit's docstring).
            confidence.note_manual_edit(row)
            self._mark_dirty()
            self._update_block_matrix_embed(bidx)
            # Rebuild instead of patching only the edited label/gloss cells:
            # MatrixLang/EmbedLang meta rows live at a DIFFERENT visible row
            # than the token just edited, and _update_block_matrix_embed only
            # touches self.blocks -- without a rebuild those meta-row cells
            # would keep showing their pre-Apply value in both the main and
            # full-edit sheets even though self.blocks is already correct.
            # _rebuild_grid_from_model is the same shared helper _uid_undo_last
            # and Merge Cells already use for this.
            self._rebuild_grid_from_model(select_row=vis_r, select_col=2)

        # Re-derive the list: a token whose label is no longer UID drops out
        # of the default list at this same index, which naturally advances
        # the selection to the next remaining UID. Leaving the label
        # unchanged and pressing Next/Apply again simply keeps it as UID.
        self._uid_refresh_items(preserve_index=True)

    # --- search / find all occurrences ---

    def _uid_on_search(self, event=None):
        self._uid_mode = 'uid'
        self._uid_query = ''
        self._uid_refresh_items(preserve_index=False)
        return 'break'

    def open_uid_find_occurrences(self):
        query = (self._uid_search_var.get() or '').strip()
        if not query and self._uid_items:
            it = self._uid_items[self._uid_i]
            query = self.blocks[it['bidx']][it['ridx']].get('token', '')
        if not query:
            messagebox.showinfo("Find All Occurrences", "Type a token in Search, or select one first.")
            return
        self._uid_mode = 'occurrences'
        self._uid_query = query
        self._uid_refresh_items(preserve_index=False)

    # --- main window construction ---

    def open_uid_review_tool(self):
        if getattr(self, '_uid_win', None) is not None and self._uid_win.winfo_exists():
            try:
                self._uid_win.deiconify()
                self._uid_win.lift()
                self._uid_win.focus_force()
            except Exception:
                pass
            return

        self._uid_mode = 'uid'
        self._uid_query = ''
        self._uid_undo_stack = []
        # Reset the review filter to its default on every fresh open
        # (mirrors the _uid_mode/_uid_query reset above): "All Uncertain"
        # -- every token, any label, whose confidence record has
        # review_recommended=True. The checkbox-based label/band filters
        # (used by the "UID Only"/"Custom" views) reset to their own
        # backward-compatible defaults (UID-only, no band restriction) so
        # switching to either of those views starts from a known state.
        self._review_view_mode = 'all_uncertain'
        self._review_label_filter = {"UID"}
        self._review_band_filter = set()
        self._review_hide_reviewed = False

        win = tk.Toplevel(self)
        win.title('Confidence Review Tool')
        win.geometry('860x640')
        win.minsize(720, 480)
        win.configure(bg=DARK_BG)
        win.transient(self)
        # Not modal.

        outer = ttk.Frame(win, style='Dark.TFrame')
        outer.pack(fill='both', expand=True, padx=10, pady=10)

        # --- search row ---
        top = ttk.Frame(outer, style='Dark.TFrame')
        top.pack(fill='x')
        ttk.Label(top, text="Search:", style='Dark.TLabel').pack(side='left')
        self._uid_search_var = tk.StringVar(value='')
        search_entry = ttk.Entry(top, textvariable=self._uid_search_var, style='Dark.TEntry', width=20)
        search_entry.pack(side='left', padx=(4, 6))
        search_entry.bind('<Return>', self._uid_on_search)
        self._uid_search_entry = search_entry
        ttk.Button(top, text="Search", style='Dark.TButton', command=self._uid_on_search).pack(side='left')
        ttk.Button(top, text="Find All Occurrences", style='Dark.TButton',
                   command=self.open_uid_find_occurrences).pack(side='left', padx=(8, 0))

        # --- review filter row (requirement: optionally review any of the
        # 7 labels, filtered to LOW and/or selected MEDIUM confidence) ---
        def _on_filter_changed():
            # Direct checkbox interaction always means the checkbox-based
            # filter is now in control -- the view combobox reflects that
            # as "Custom" rather than silently drifting out of sync with
            # what's actually being shown.
            self._review_view_mode = 'checkbox_based'
            self._review_view_var.set(self.REVIEW_VIEW_CUSTOM)
            labels = {lab for lab, v in self._review_label_check_vars.items() if v.get()}
            self._review_label_filter = labels or {"UID"}
            self._review_band_filter = {b for b, v in self._review_band_check_vars.items() if v.get()}
            self._review_hide_reviewed = bool(self._review_hide_reviewed_var.get())
            self._uid_refresh_items(preserve_index=False)

        def _on_review_view_changed(event=None):
            choice = self._review_view_var.get()
            if choice == self.REVIEW_VIEW_ALL_UNCERTAIN:
                self._review_view_mode = 'all_uncertain'
            elif choice == self.REVIEW_VIEW_UID_ONLY:
                self._review_view_mode = 'checkbox_based'
                self._review_label_filter = {"UID"}
                self._review_band_filter = set()
                for lab, v in self._review_label_check_vars.items():
                    v.set(lab == "UID")
                for v in self._review_band_check_vars.values():
                    v.set(False)
            else:  # Custom -- leave whatever the checkboxes currently say
                self._review_view_mode = 'checkbox_based'
            self._uid_refresh_items(preserve_index=False)

        view_row = ttk.Frame(outer, style='Dark.TFrame')
        view_row.pack(fill='x', pady=(2, 0))
        ttk.Label(view_row, text="View:", style='Dark.TLabel').pack(side='left')
        self._review_view_var = tk.StringVar(value=self.REVIEW_VIEW_ALL_UNCERTAIN)
        view_combo = ttk.Combobox(view_row, textvariable=self._review_view_var, values=list(self.REVIEW_VIEWS),
                                   width=14, state='readonly')
        view_combo.pack(side='left', padx=(4, 0))
        view_combo.bind('<<ComboboxSelected>>', _on_review_view_changed)
        self._review_view_combo = view_combo
        self._make_tooltip(
            view_combo,
            "All Uncertain (default): every token, any label, whose confidence score flags it for review.\n"
            "UID Only: the tool's original view -- UID-labeled tokens only.\n"
            "Custom: whatever the Labels/Confidence checkboxes below currently say.")

        filt = ttk.Frame(outer, style='Dark.TFrame')
        filt.pack(fill='x', pady=(6, 0))
        ttk.Label(filt, text="Labels:", style='Dark.TLabel').pack(side='left')
        self._review_label_check_vars = {}
        for lab in self.ALL_LABELS:
            v = tk.BooleanVar(value=(lab in self._review_label_filter))
            ttk.Checkbutton(filt, text=lab, style='Dark.TCheckbutton', variable=v,
                             command=_on_filter_changed).pack(side='left', padx=(2, 0))
            self._review_label_check_vars[lab] = v

        filt2 = ttk.Frame(outer, style='Dark.TFrame')
        filt2.pack(fill='x', pady=(4, 0))
        ttk.Label(filt2, text="Confidence:", style='Dark.TLabel').pack(side='left')
        self._review_band_check_vars = {}
        for band in self.REVIEW_BANDS:
            v = tk.BooleanVar(value=(band in self._review_band_filter))
            ttk.Checkbutton(filt2, text=band.title(), style='Dark.TCheckbutton', variable=v,
                             command=_on_filter_changed).pack(side='left', padx=(2, 0))
            self._review_band_check_vars[band] = v
        self._review_hide_reviewed_var = tk.BooleanVar(value=self._review_hide_reviewed)
        ttk.Checkbutton(filt2, text="Hide reviewed", style='Dark.TCheckbutton',
                         variable=self._review_hide_reviewed_var,
                         command=_on_filter_changed).pack(side='left', padx=(16, 0))

        # --- list ---
        cols = ("sentence", "token", "label", "gloss", "band", "score", "reviewed")
        headers = {"sentence": "Sentence", "token": "Token", "label": "Current Label", "gloss": "Gloss",
                   "band": "Confidence", "score": "Score", "reviewed": "Reviewed"}
        widths = {"sentence": 240, "token": 110, "label": 90, "gloss": 130,
                  "band": 80, "score": 55, "reviewed": 65}

        listframe = ttk.Frame(outer, style='Dark.TFrame')
        listframe.pack(fill='both', expand=True, pady=(8, 0))
        tree = ttk.Treeview(listframe, columns=cols, show='headings', style='Conc.Treeview', selectmode='browse')
        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=widths[c], anchor='w', stretch=(c == 'sentence'))
        ysb = ttk.Scrollbar(listframe, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side='left', fill='both', expand=True)
        ysb.pack(side='left', fill='y')
        self._uid_tree = tree
        tree.bind('<<TreeviewSelect>>', self._uid_on_tree_select)
        tree.bind('<Double-1>', self._uid_on_tree_double_click)

        # --- context ---
        ctx_frame = ttk.Frame(outer, style='Dark.TFrame')
        ctx_frame.pack(fill='x', pady=(8, 0))
        ttk.Label(ctx_frame, text="Context:", style='Dark.TLabel').pack(side='left', anchor='n')
        self._uid_context_var = tk.StringVar(value='')
        ttk.Label(ctx_frame, textvariable=self._uid_context_var, style='Dark.TLabel',
                  wraplength=680, justify='left').pack(side='left', padx=(6, 0), anchor='n')

        # --- confidence / evidence (requirement: show current label,
        # confidence, reasons, and evidence) ---
        ev_frame = ttk.Frame(outer, style='Dark.TFrame')
        ev_frame.pack(fill='x', pady=(6, 0))
        ttk.Label(ev_frame, text="Evidence:", style='Dark.TLabel').pack(side='left', anchor='n')
        self._uid_evidence_var = tk.StringVar(value='')
        ttk.Label(ev_frame, textvariable=self._uid_evidence_var, style='Dark.TLabel',
                  wraplength=760, justify='left').pack(side='left', padx=(6, 0), anchor='n')

        # --- edit panel ---
        edit = ttk.Frame(outer, style='Dark.TFrame')
        edit.pack(fill='x', pady=(10, 0))
        ttk.Label(edit, text="Label:", style='Dark.TLabel').pack(side='left')
        self._uid_label_var = tk.StringVar(value='')
        label_combo = ttk.Combobox(edit, textvariable=self._uid_label_var, values=list(self.ALL_LABELS),
                                    width=8, state='readonly')
        label_combo.pack(side='left', padx=(4, 12))
        self._uid_label_combo = label_combo

        ttk.Label(edit, text="Gloss:", style='Dark.TLabel').pack(side='left')
        self._uid_gloss_var = tk.StringVar(value='')
        ttk.Entry(edit, textvariable=self._uid_gloss_var, style='Dark.TEntry', width=28).pack(side='left', padx=(4, 0))

        # --- navigation + actions ---
        nav = ttk.Frame(outer, style='Dark.TFrame')
        nav.pack(fill='x', pady=(10, 0))
        ttk.Button(nav, text="First", style='Dark.TButton', command=self._uid_first).pack(side='left')
        ttk.Button(nav, text="\u25c0 Previous", style='Dark.TButton', command=self._uid_prev).pack(side='left', padx=4)
        ttk.Button(nav, text="Next \u25b6", style='Dark.TButton', command=self._uid_next).pack(side='left', padx=4)
        ttk.Button(nav, text="Last", style='Dark.TButton', command=self._uid_last).pack(side='left', padx=4)
        ttk.Button(nav, text="Apply", style='Dark.TButton', command=self._uid_apply).pack(side='left', padx=(16, 4))
        ttk.Button(nav, text="Undo", style='Dark.TButton', command=self._uid_undo_last).pack(side='left', padx=4)
        ttk.Button(nav, text="Close", style='Dark.TButton', command=lambda: _on_close()).pack(side='right')

        # --- keyboard shortcuts ---
        def _apply_shortcut(event=None):
            self._uid_apply()
            return 'break'

        def _undo_shortcut(event=None):
            self._uid_undo_last()
            return 'break'

        def _save_shortcut(event=None):
            self.save_project_progress()
            return 'break'

        win.bind('<Control-Return>', _apply_shortcut)
        win.bind('<Command-Return>', _apply_shortcut)
        win.bind('<Control-z>', _undo_shortcut)
        win.bind('<Command-z>', _undo_shortcut)
        win.bind('<Control-s>', _save_shortcut)
        win.bind('<Command-s>', _save_shortcut)
        # Escape must never close the whole project unexpectedly -- not bound
        # to closing this window at all.

        def _on_close():
            self._uid_win = None
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol('WM_DELETE_WINDOW', _on_close)
        self._uid_win = win
        self._uid_refresh_items(preserve_index=False)
        try:
            tree.focus_set()
        except Exception:
            pass

    # =====================================================================
    # TDK Checker
    #
    # A SEPARATE tool from the Confidence Review Tool (Tools -> TDK
    # Checker; never replaces or renames it). Looks up a single token, its
    # parser-proposed root/lemma, and each proposed suffix segment against
    # a pluggable dictionary_provider.DictionaryProvider -- by default the
    # real (best-effort, undocumented-endpoint) TDKProvider, constructed
    # lazily on first use (see _ensure_tdk_provider) so importing/starting
    # the app, running annotation, or opening this window never itself
    # touches the network. The ONLY network-triggering actions are the
    # "Check TDK" button and "Open in TDK Checker" from a grid selection
    # (which explicitly requested a lookup by the act of clicking it) --
    # never automatic, never per-token, never at startup.
    #
    # TDK membership is evidence of Turkish lexicalization, not a label
    # decision: this tool never reads or writes a row's `label`. Apply
    # Correction only ever updates `gloss` and a `tdk_segmentation`
    # metadata dict on the row -- exactly like the Confidence Review
    # Tool's Apply only ever touches `label`/`gloss`/`confidence`, this
    # tool's Apply only ever touches `gloss`/`tdk_segmentation`.
    # =====================================================================

    def _ensure_tdk_parser_annotator(self):
        """A lightweight stand-in Annotator for the morphological parser
        only -- deliberately NOT self.annotator (the real pipeline
        annotator, which requires loading the full fastText model and
        Stanza). tdk_parser.parse_token's ranking is lexicon-aware (it
        needs to tell a genuine root apart from a merely-attested inflected
        surface form -- see tdk_parser.py's module docstring), so it needs
        the real turkish_freq_top/_all/english_freq_words word lists, not
        an empty stand-in. tdk_parser.load_lexicon_annotator() reads only
        those two plain-text files (no fastText, no Stanza, no model
        loading) -- local and well under a second, called by
        _set_runtime_workdir()'s already-chdir'd-into-resources/ working
        directory, exactly like Annotator.__init__'s own defaults."""
        if getattr(self, '_tdk_parser_annotator', None) is None:
            self._tdk_parser_annotator = tdk_parser.load_lexicon_annotator()
        return self._tdk_parser_annotator

    def _ensure_tdk_provider(self):
        """Lazily construct the real TDK provider on first actual use.
        Tests set self._tdk_provider directly to a MockDictionaryProvider
        BEFORE triggering any lookup, so this never runs in a test and
        never touches the network there."""
        if self._tdk_provider is None:
            self._tdk_provider = dictionary_provider.TDKProvider()
        return self._tdk_provider

    # --- row resolution (main grid) -----------------------------------

    def _resolve_tdk_grid_selection(self):
        """Resolve the current main-grid selection to a single token row
        for the TDK Checker. Never assumes the visual row number equals
        the token index -- always goes through self._row_index_map /
        annotation_model.resolve_row, exactly like Merge Cells does.

        Returns (bidx, ridx, vis_r, error) where `error` is None on
        success, or (messagebox_kind, message) on failure -- kind is
        'info' (no selection at all) or 'warning' (a selection exists but
        isn't a usable token row)."""
        if self.sheet is None:
            return None, None, None, ('info', "Select a token row first.")
        try:
            cells = self.sheet.get_selected_cells()
        except Exception:
            cells = None
        if not cells:
            return None, None, None, ('info', "Select a token row first.")

        vis_rows = sorted(set(r for r, _c in cells))

        def classify(r):
            if r in self._sep_rows:
                return 'separator', None
            bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
            if bidx is None:
                return 'unresolved', None
            row = self.blocks[bidx][ridx]
            tok = str(row.get('token', '') or '').strip()
            if tok == 'MatrixLang':
                return 'matrixlang', (bidx, ridx)
            if tok == 'EmbedLang':
                return 'embedlang', (bidx, ridx)
            if annotation_model.is_meta_row_token(row.get('token', '')):
                return 'meta', (bidx, ridx)
            if not tok:
                return 'empty', (bidx, ridx)
            return 'token', (bidx, ridx)

        classified = [(r,) + classify(r) for r in vis_rows]
        token_rows = [(r, info) for (r, kind, info) in classified if kind == 'token']

        if len(vis_rows) == 1:
            r, kind, info = classified[0]
            if kind == 'token':
                bidx, ridx = info
                return bidx, ridx, r, None
            messages = {
                'separator': ('warning', "Cannot open TDK Checker for a separator row."),
                'matrixlang': ('warning', "MatrixLang rows cannot be opened in TDK Checker."),
                'embedlang': ('warning', "EmbedLang rows cannot be opened in TDK Checker."),
                'empty': ('warning', "Selected row has no token."),
                'meta': ('warning', "Select a normal token row, not a sentence/structure row."),
                'unresolved': ('warning', "The selected row could not be resolved."),
            }
            return None, None, None, messages.get(kind, ('warning', "Select a normal token row."))

        # Multiple rows selected: use the first valid token row; if none
        # of the selection is a usable token row, ask for a single
        # selection instead of guessing.
        if token_rows:
            r, (bidx, ridx) = token_rows[0]
            return bidx, ridx, r, None
        return None, None, None, ('info', "Select a single token row to open in TDK Checker.")

    def open_tdk_checker_from_grid(self):
        bidx, ridx, vis_r, err = self._resolve_tdk_grid_selection()
        if err is not None:
            kind, msg = err
            if kind == 'info':
                messagebox.showinfo("TDK Checker", msg)
            else:
                messagebox.showwarning("TDK Checker", msg)
            return
        self._open_tdk_checker_window(bidx=bidx, ridx=ridx, vis_r=vis_r, auto_run=True)

    def open_tdk_checker_tool(self):
        """Tools -> TDK Checker: opens (or reuses) the window with no row
        preloaded -- the user types/pastes a term and clicks Check TDK
        themselves. Never auto-runs a lookup (nothing was "explicitly
        clicked" on a specific token yet)."""
        self._open_tdk_checker_window(bidx=None, ridx=None, vis_r=None, auto_run=False)

    # --- window lifecycle ------------------------------------------------

    def _open_tdk_checker_window(self, bidx=None, ridx=None, vis_r=None, auto_run=False):
        if getattr(self, '_tdk_win', None) is None or not self._tdk_win.winfo_exists():
            self._tdk_build_window()
        else:
            try:
                self._tdk_win.deiconify()
                self._tdk_win.lift()
                self._tdk_win.focus_force()
            except Exception:
                pass

        if bidx is not None:
            self._tdk_load_row(bidx, ridx, vis_r, auto_run=auto_run)

    def _tdk_sentence_id_for_block(self, bidx):
        try:
            blk = self.blocks[bidx]
        except Exception:
            return ""
        for r in blk:
            if str(r.get('token', '') or '').strip() == 'SentenceID':
                return str(r.get('label', '') or r.get('gloss', '') or '').strip()
        return ""

    def _tdk_sentence_text(self, bidx):
        try:
            blk = self.blocks[bidx]
        except Exception:
            return ""
        toks = [str(r.get('token', '') or '') for r in blk if not self._is_meta_row_token(r.get('token', ''))]
        return " ".join(toks)

    def _tdk_build_window(self):
        win = tk.Toplevel(self)
        win.title('TDK Checker')
        win.geometry('900x840')
        win.minsize(760, 700)
        win.configure(bg=DARK_BG)
        win.transient(self)

        outer = ttk.Frame(win, style='Dark.TFrame')
        outer.pack(fill='both', expand=True, padx=10, pady=10)

        # --- token + context row ---
        top = ttk.Frame(outer, style='Dark.TFrame')
        top.pack(fill='x')
        ttk.Label(top, text="Token:", style='Dark.TLabel').pack(side='left')
        self._tdk_token_var = tk.StringVar(value='')
        token_entry = ttk.Entry(top, textvariable=self._tdk_token_var, style='Dark.TEntry', width=24)
        token_entry.pack(side='left', padx=(4, 12))
        self._tdk_token_entry = token_entry
        ttk.Label(top, text="Sentence ID:", style='Dark.TLabel').pack(side='left')
        self._tdk_sentid_var = tk.StringVar(value='')
        ttk.Label(top, textvariable=self._tdk_sentid_var, style='Dark.TLabel').pack(side='left', padx=(4, 12))
        ttk.Label(top, text="Token Index:", style='Dark.TLabel').pack(side='left')
        self._tdk_tokidx_var = tk.StringVar(value='')
        ttk.Label(top, textvariable=self._tdk_tokidx_var, style='Dark.TLabel').pack(side='left', padx=(4, 0))

        ctx_frame = ttk.Frame(outer, style='Dark.TFrame')
        ctx_frame.pack(fill='x', pady=(6, 0))
        ttk.Label(ctx_frame, text="Sentence:", style='Dark.TLabel').pack(side='left', anchor='n')
        self._tdk_context_var = tk.StringVar(value='')
        ttk.Label(ctx_frame, textvariable=self._tdk_context_var, style='Dark.TLabel',
                  wraplength=820, justify='left').pack(side='left', padx=(6, 0), anchor='n')

        # --- parser row: root/segments are freely editable; edits after a
        # Check TDK run mark the displayed results/explanation stale (see
        # _tdk_on_root_or_segments_edited) but NEVER trigger a lookup or a
        # re-parse by themselves -- only Re-parse or Check TDK do that. ---
        parse_frame = ttk.Frame(outer, style='Dark.TFrame')
        parse_frame.pack(fill='x', pady=(10, 0))
        ttk.Label(parse_frame, text="Root/Lemma:", style='Dark.TLabel').pack(side='left')
        self._tdk_root_var = tk.StringVar(value='')
        ttk.Entry(parse_frame, textvariable=self._tdk_root_var, style='Dark.TEntry', width=16).pack(
            side='left', padx=(4, 12))
        ttk.Label(parse_frame, text="Segments:", style='Dark.TLabel').pack(side='left')
        self._tdk_segments_var = tk.StringVar(value='')
        ttk.Entry(parse_frame, textvariable=self._tdk_segments_var, style='Dark.TEntry', width=24).pack(
            side='left', padx=(4, 12))
        ttk.Button(parse_frame, text="Re-parse", style='Dark.TButton', command=self._tdk_reparse).pack(side='left')
        self._tdk_root_var.trace_add('write', self._tdk_on_root_or_segments_edited)
        self._tdk_segments_var.trace_add('write', self._tdk_on_root_or_segments_edited)

        parse_status_frame = ttk.Frame(outer, style='Dark.TFrame')
        parse_status_frame.pack(fill='x', pady=(4, 0))
        ttk.Label(parse_status_frame, text="Parser:", style='Dark.TLabel').pack(side='left')
        self._tdk_parser_status_var = tk.StringVar(value='')
        ttk.Label(parse_status_frame, textvariable=self._tdk_parser_status_var, style='Dark.TLabel',
                  wraplength=820, justify='left').pack(side='left', padx=(6, 0))

        # --- explanation panel: exactly what the parser found and why,
        # segment by segment (task requirement: "explain how suffixes were
        # found"). Read-only, refreshed by _tdk_render_explanation(). ---
        expl_frame = ttk.Frame(outer, style='Dark.TFrame')
        expl_frame.pack(fill='x', pady=(8, 0))
        ttk.Label(expl_frame, text="Explanation:", style='Dark.TLabel').pack(anchor='w')
        expl_text = tk.Text(expl_frame, height=6, width=100, bg=DARK_BG, fg=DARK_FG,
                             insertbackground=DARK_FG, relief='flat', wrap='word')
        expl_text.configure(state='disabled')
        expl_text.pack(fill='x', pady=(2, 0))
        self._tdk_explanation_text = expl_text

        # --- TDK lookup row ---
        lookup_frame = ttk.Frame(outer, style='Dark.TFrame')
        lookup_frame.pack(fill='x', pady=(10, 0))
        ttk.Button(lookup_frame, text="Check TDK", style='Dark.TButton', command=self._tdk_check_all).pack(
            side='left')
        ttk.Label(lookup_frame, text="Status:", style='Dark.TLabel').pack(side='left', padx=(12, 0))
        self._tdk_status_var = tk.StringVar(value='')
        ttk.Label(lookup_frame, textvariable=self._tdk_status_var, style='Dark.TLabel').pack(
            side='left', padx=(4, 0))
        self._make_tooltip(
            lookup_frame,
            "TDK Checker sends only the selected token/root/segment to sozluk.gov.tr, never the "
            "surrounding sentence, and only when you click Check TDK or Open in TDK Checker. TDK "
            "membership is evidence of Turkish lexicalization -- it never changes a token's label "
            "automatically. Editing Root/Lemma or Segments after a check marks the results below as "
            "stale (a previous-query answer) until you click Check TDK again.")

        # --- results tree: full token / root / each segment ---
        results_frame = ttk.Frame(outer, style='Dark.TFrame')
        results_frame.pack(fill='both', expand=True, pady=(8, 0))
        cols = ("term", "kind", "status", "detail")
        headers = {"term": "Term", "kind": "Type", "status": "TDK Status", "detail": "Detail"}
        widths = {"term": 140, "kind": 80, "status": 110, "detail": 380}
        tree = ttk.Treeview(results_frame, columns=cols, show='headings', style='Conc.Treeview',
                             selectmode='browse', height=5)
        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=widths[c], anchor='w', stretch=(c == 'detail'))
        ysb = ttk.Scrollbar(results_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side='left', fill='both', expand=True)
        ysb.pack(side='left', fill='y')
        self._tdk_results_tree = tree
        tree.bind('<<TreeviewSelect>>', self._tdk_on_result_select)

        # --- dictionary detail panel: the FULL TDK entry (not just
        # FOUND/NOT_FOUND) for whichever result row is selected above --
        # headword, POS, every sense's definition/usage labels/examples,
        # origin, pronunciation, compounds, idioms, proverbs, source,
        # query. Missing fields always show "Not provided", never a
        # guessed value. Read-only. ---
        detail_frame = ttk.Frame(outer, style='Dark.TFrame')
        detail_frame.pack(fill='both', expand=True, pady=(6, 0))
        ttk.Label(detail_frame, text="Dictionary Detail:", style='Dark.TLabel').pack(anchor='w')
        detail_text = tk.Text(detail_frame, height=10, width=100, bg=DARK_BG, fg=DARK_FG,
                               insertbackground=DARK_FG, relief='flat', wrap='word')
        detail_text.configure(state='disabled')
        detail_text.pack(fill='both', expand=True, pady=(2, 0))
        self._tdk_detail_text = detail_text

        # --- navigation + actions ---
        nav = ttk.Frame(outer, style='Dark.TFrame')
        nav.pack(fill='x', pady=(10, 0))
        ttk.Button(nav, text="Apply Correction", style='Dark.TButton', command=self._tdk_apply_correction).pack(
            side='left')
        ttk.Button(nav, text="Undo", style='Dark.TButton', command=self._tdk_undo_last).pack(side='left', padx=4)
        ttk.Button(nav, text="Find All Occurrences", style='Dark.TButton',
                   command=self.open_tdk_find_occurrences).pack(side='left', padx=4)
        ttk.Button(nav, text="Close", style='Dark.TButton', command=lambda: _on_close()).pack(side='right')

        def _apply_shortcut(event=None):
            self._tdk_apply_correction()
            return 'break'

        def _undo_shortcut(event=None):
            self._tdk_undo_last()
            return 'break'

        win.bind('<Control-Return>', _apply_shortcut)
        win.bind('<Command-Return>', _apply_shortcut)
        win.bind('<Control-z>', _undo_shortcut)
        win.bind('<Command-z>', _undo_shortcut)

        def _on_close():
            self._tdk_win = None
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol('WM_DELETE_WINDOW', _on_close)
        self._tdk_win = win
        self._tdk_clear_results()
        self._tdk_render_explanation()

    # --- loading a row / parsing ------------------------------------------

    def _tdk_load_row(self, bidx, ridx, vis_r, auto_run=False):
        """Populate the checker from a specific main-table row. Preserves
        active dataset / sentence ID / token index / row identity via
        self._tdk_current, checked before Apply Correction ever writes
        anything back."""
        try:
            row = self.blocks[bidx][ridx]
        except Exception:
            messagebox.showerror("TDK Checker", "The selected row could not be read.")
            return

        dataset_id = None
        try:
            dataset_id = self.datasets[self._active_dataset_index].get('id')
        except Exception:
            pass

        self._tdk_current = {'bidx': bidx, 'ridx': ridx, 'vis_r': vis_r, 'dataset_id': dataset_id}
        token = str(row.get('token', '') or '')
        self._tdk_token_var.set(token)
        sid = self._tdk_sentence_id_for_block(bidx)
        self._tdk_sentid_var.set(sid)
        self._tdk_tokidx_var.set(str(row.get('idx', '') or ''))
        self._tdk_context_var.set(self._tdk_sentence_text(bidx))

        self._tdk_last_query_snapshot = None
        self._tdk_last_results = []
        self._tdk_results_stale = False
        self._tdk_clear_results()

        existing_seg = row.get('tdk_segmentation')
        self._tdk_reparse(prefer_existing=existing_seg)

        if auto_run:
            self._tdk_check_all()

    def _tdk_reparse(self, prefer_existing=None):
        """Run the morphological parser on the current token field. If
        `prefer_existing` (a previously-applied tdk_segmentation dict) is
        given, that manual correction is shown instead of re-deriving a
        fresh automatic split -- re-opening a token someone already
        corrected must not silently discard their correction. Only ever
        called from _tdk_load_row (an initial load, honoring
        prefer_existing) or the explicit Re-parse button -- typing into
        the Root/Lemma or Segments fields never calls this on its own."""
        token = self._tdk_token_var.get().strip()
        if not token:
            self._tdk_root_var.set('')
            self._tdk_segments_var.set('')
            self._tdk_parser_status_var.set('(no token)')
            self._tdk_parse_result = None
            self._tdk_render_explanation()
            return

        if prefer_existing:
            root = prefer_existing.get('root', '') or ''
            segs = prefer_existing.get('segments', []) or []
            self._tdk_root_var.set(root)
            self._tdk_segments_var.set(' + '.join(segs))
            self._tdk_parser_status_var.set("previously corrected segmentation (manual)")
            self._tdk_parse_result = tdk_parser.ParseResult(
                token=token, root=root, segments=tuple(segs), success=True, source='manual',
                category=tdk_parser.CATEGORY_MANUAL, part_of_speech='unknown',
                reason='previously applied correction')
            self._tdk_render_explanation()
            return

        annotator = self._ensure_tdk_parser_annotator()
        if annotator is None:
            self._tdk_parser_status_var.set("parser unavailable")
            self._tdk_root_var.set(token)
            self._tdk_segments_var.set('')
            self._tdk_parse_result = tdk_parser._whole_token_fallback(token, "annotator not available")
            self._tdk_render_explanation()
            return

        result = tdk_parser.parse_token(token, annotator)
        self._tdk_parse_result = result
        self._tdk_root_var.set(result.root)
        self._tdk_segments_var.set(' + '.join(result.segments))
        self._tdk_parser_status_var.set(result.reason)
        self._tdk_render_explanation()

    _TDK_CATEGORY_DISPLAY = {
        'full_turkish_lexical_item': 'full Turkish lexical item',
        'english_root_turkish_suffix': 'English root + Turkish suffix',
        'ambiguous_candidate': 'ambiguous candidate',
        'invalid_parser_proposal': 'invalid parser proposal',
        'manual_correction': 'manual correction',
    }

    def _tdk_render_explanation(self):
        """Renders the 'how were these suffixes found' panel from the
        current self._tdk_parse_result -- Token / Root / Suffix / Analysis
        (per segment) / Status, mirroring the task's own example format.
        Never shows a single character as a suffix's "analysis" unless
        tdk_parser itself classified it as a genuinely valid suffix."""
        widget = getattr(self, '_tdk_explanation_text', None)
        if widget is None:
            return
        result = self._tdk_parse_result
        lines = []
        if result is None:
            lines = ["(no token parsed yet)"]
        else:
            current_root = self._tdk_root_var.get().strip()
            current_segtext = self._tdk_segments_var.get().strip()
            parsed_segtext = ' + '.join(result.segments)
            if current_root != (result.root or '') or current_segtext != parsed_segtext:
                lines.append("(Root/Segments edited since this parse -- click Re-parse to refresh "
                              "this explanation; Check TDK will use your edited values regardless.)")
            lines.append(f"Token: {result.token}")
            lines.append(f"Root: {result.root or dictionary_provider.NOT_PROVIDED}")
            if result.segments:
                explanations = result.segment_explanations or ()
                for i, seg in enumerate(result.segments):
                    expl = explanations[i] if i < len(explanations) else None
                    lines.append(f"Suffix: -{seg}")
                    if expl is not None:
                        lines.append(f"Analysis: {expl.rule}" + ("" if expl.valid else " (NOT a recognized suffix)"))
                    else:
                        lines.append("Analysis: manually specified segment (not auto-classified)")
            else:
                lines.append("Suffix: (none -- no suffix split proposed)")
            lines.append(f"Status: {self._TDK_CATEGORY_DISPLAY.get(result.category, result.category)}")
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', "\n".join(lines))
        widget.configure(state='disabled')

    def _tdk_on_root_or_segments_edited(self, *_args):
        """Bound to the Root/Lemma and Segments StringVar traces. Never
        triggers a lookup or a re-parse -- only marks whatever is currently
        displayed (TDK results, explanation) as referring to a previous
        query, so the user can never mistake a stale answer for one that
        reflects their latest edit."""
        self._tdk_recompute_staleness()
        self._tdk_render_explanation()

    def _tdk_recompute_staleness(self):
        """Compares the CURRENT token/root/segments against the query
        snapshot Check TDK was last run against (see _tdk_check_all).
        If they differ, every currently-displayed TDK result is stale --
        it answers a query that no longer matches what's in the fields."""
        if self._tdk_last_query_snapshot is None:
            return
        self._tdk_results_stale = (self._tdk_terms_to_check() != self._tdk_last_query_snapshot)
        if self._tdk_last_results:
            self._tdk_render_results_tree()

    # --- TDK lookup (async, off the Tk main thread) -----------------------

    def _tdk_terms_to_check(self):
        token = self._tdk_token_var.get().strip()
        root = self._tdk_root_var.get().strip()
        seg_text = self._tdk_segments_var.get().strip()
        segments = [s.strip() for s in re.split(r"[+\-]", seg_text) if s.strip()] if seg_text else []
        return {'full': token, 'root': root, 'segments': segments}

    def _tdk_check_all(self):
        """Triggered explicitly by the 'Check TDK' button, or automatically
        exactly once right after 'Open in TDK Checker' from the grid
        (never anywhere else, never at startup, never per-token during
        normal annotation). Looks up the full token, the CURRENT root/
        lemma, and each CURRENT proposed segment -- always whatever is in
        the fields right now, never a stale parser value. Always off the
        Tk main thread, so the GUI never freezes even if the network is
        slow or unreachable."""
        terms = self._tdk_terms_to_check()
        if not terms['full']:
            messagebox.showwarning("TDK Checker", "Enter a token first.")
            return

        self._tdk_lookup_generation += 1
        generation = self._tdk_lookup_generation
        # New query snapshot: everything after this point is compared
        # against exactly these values, never the parser's original
        # proposal, to decide whether a displayed result is stale.
        self._tdk_last_query_snapshot = dict(terms)
        self._tdk_results_stale = False
        self._tdk_status_var.set("checking...")
        provider = self._ensure_tdk_provider()

        def worker():
            # Runs on a background thread: touches ONLY `provider` (a
            # plain, thread-safe-by-design synchronous object with no Tk
            # dependency) and the thread-safe result queue below -- never
            # any Tk widget, never self.after() directly. Calling Tk APIs
            # from a non-main thread is not reliably safe (in particular,
            # after() requires the interpreter to be inside a running
            # mainloop); routing every result through queue.Queue and
            # draining it exclusively from a main-thread-scheduled
            # after()-poller (_tdk_poll_results) avoids that entirely.
            results = []
            try:
                r_full = provider.lookup(terms['full'])
                results.append(('full', terms['full'], r_full))
                if terms['root'] and terms['root'].lower() != terms['full'].lower():
                    r_root = provider.lookup(terms['root'])
                    results.append(('root', terms['root'], r_root))
                for seg in terms['segments']:
                    r_seg = provider.lookup(seg)
                    results.append(('segment', seg, r_seg))
            except Exception as e:
                results.append(('full', terms['full'],
                                 dictionary_provider.LookupResult(
                                     query=terms['full'], normalized_query='',
                                     status=dictionary_provider.STATUS_UNAVAILABLE,
                                     source='tdk', message=f"unexpected error: {e}")))
            self._tdk_result_queue.put((generation, results))

        threading.Thread(target=worker, daemon=True).start()
        self._tdk_ensure_poller_running()

    def _tdk_ensure_poller_running(self):
        """Starts the main-thread result-queue poller if it isn't already
        scheduled. Always called from the main thread (never from a
        worker) -- the poller re-schedules itself via after() as long as
        the TDK window stays open, and stops on its own once it's closed,
        so there is never more than one poll loop running."""
        if self._tdk_poll_scheduled:
            return
        self._tdk_poll_scheduled = True
        self.after(50, self._tdk_poll_results)

    def _tdk_poll_results(self):
        """Runs exclusively as an after() callback on the Tk main thread.
        Drains every result a background worker has queued so far, then
        re-schedules itself -- unless the TDK window has been closed, in
        which case polling simply stops (a fresh poller is started the
        next time a lookup is triggered)."""
        try:
            while True:
                generation, results = self._tdk_result_queue.get_nowait()
                self._tdk_on_lookup_results(generation, results)
        except queue.Empty:
            pass

        if getattr(self, '_tdk_win', None) is not None and self._tdk_win.winfo_exists():
            self.after(50, self._tdk_poll_results)
        else:
            self._tdk_poll_scheduled = False

    def _tdk_on_lookup_results(self, generation, results):
        """Runs on the Tk main thread (scheduled via after()). Drops the
        result entirely if a newer lookup has since been started -- the
        user changed the token/re-clicked before this one finished (the
        request-ID / generation protection required for async safety).
        Even for the latest generation, if root/segments were edited AFTER
        this exact lookup was kicked off but before it returned, the
        result is marked STALE_RESULT rather than shown as current --
        _tdk_recompute_staleness re-checks that against the CURRENT
        fields, not just against the generation counter."""
        if generation != self._tdk_lookup_generation:
            return  # stale response -- ignore
        if getattr(self, '_tdk_win', None) is None or not self._tdk_win.winfo_exists():
            return
        self._tdk_populate_results(results)
        self._tdk_recompute_staleness()
        if self._tdk_results_stale:
            self._tdk_status_var.set(f"{dictionary_provider.STATUS_STALE_RESULT} -- "
                                      "query changed since this check; press Check TDK again")
        else:
            overall = results[0][2].status if results else dictionary_provider.STATUS_UNAVAILABLE
            self._tdk_status_var.set(f"{overall} (source: {results[0][2].source})" if results else "UNAVAILABLE")

    def _tdk_clear_results(self):
        self._tdk_last_results = []
        if getattr(self, '_tdk_results_tree', None) is not None:
            try:
                self._tdk_results_tree.delete(*self._tdk_results_tree.get_children())
            except Exception:
                pass
        if getattr(self, '_tdk_status_var', None) is not None:
            self._tdk_status_var.set('')
        self._tdk_render_detail(None, None)

    def _tdk_populate_results(self, results):
        self._tdk_last_results = list(results)
        self._tdk_render_results_tree()

    def _tdk_render_results_tree(self):
        """Single source of truth for the results Treeview -- rebuilds it
        from self._tdk_last_results every time, using self._tdk_results_stale
        to decide whether every row should currently read STALE_RESULT.
        Row iid is the index into self._tdk_last_results, so selecting a
        row can recover its full LookupResult for the detail panel."""
        tree = getattr(self, '_tdk_results_tree', None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        for i, (kind, term, result) in enumerate(self._tdk_last_results):
            status = dictionary_provider.STATUS_STALE_RESULT if self._tdk_results_stale else result.status
            detail = self._tdk_short_detail(result)
            if self._tdk_results_stale:
                detail = f"(stale -- answers a previous query) {detail}"
            tree.insert("", "end", iid=str(i), values=(term, kind, status, detail))

    @staticmethod
    def _tdk_short_detail(result):
        detail = result.message
        if result.status == dictionary_provider.STATUS_FOUND and result.entries:
            heads = ", ".join(
                dictionary_provider.format_field_for_display(getattr(e, 'headword', None))
                for e in result.entries)
            detail = heads or detail
        if result.from_cache:
            detail = f"{detail} (cached)".strip()
        return detail or dictionary_provider.NOT_PROVIDED

    def _tdk_on_result_select(self, event=None):
        tree = getattr(self, '_tdk_results_tree', None)
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            self._tdk_render_detail(None, None)
            return
        try:
            idx = int(sel[0])
            kind, term, result = self._tdk_last_results[idx]
        except (ValueError, IndexError):
            self._tdk_render_detail(None, None)
            return
        self._tdk_render_detail(result, term)

    def _tdk_render_detail(self, result, term):
        """Full dictionary detail for one queried term -- headword, part
        of speech, every sense's definition/usage labels/examples, origin,
        pronunciation, compounds, idioms, proverbs, source, and the query
        itself. Every missing field shows dictionary_provider.NOT_PROVIDED,
        never a guessed value. Never dumps raw HTML/JSON -- only the
        structured fields dictionary_provider.py already extracted."""
        widget = getattr(self, '_tdk_detail_text', None)
        if widget is None:
            return
        NP = dictionary_provider.NOT_PROVIDED
        lines = []
        if result is None:
            lines = ["(select a row above to see its full dictionary detail)"]
        else:
            lines.append(f"Query: {result.query}")
            lines.append(f"Source: {result.source}")
            lines.append(f"Status: {result.status}")
            if result.entries:
                for i, e in enumerate(result.entries, 1):
                    lines.append("")
                    lines.append(f"Entry {i}")
                    lines.append(f"  Headword: {dictionary_provider.format_field_for_display(getattr(e, 'headword', None))}")
                    lines.append(f"  Part of speech: {dictionary_provider.format_field_for_display(getattr(e, 'part_of_speech', None))}")
                    lines.append(f"  Origin: {dictionary_provider.format_field_for_display(getattr(e, 'origin', None))}")
                    lines.append(f"  Pronunciation: {dictionary_provider.format_field_for_display(getattr(e, 'pronunciation', None))}")
                    senses = getattr(e, 'senses', ())
                    if senses:
                        for j, s in enumerate(senses, 1):
                            lines.append(f"  Sense {j}: {s.definition}")
                            if s.part_of_speech:
                                lines.append(f"    Part of speech: {s.part_of_speech}")
                            if s.usage_labels:
                                lines.append(f"    Usage: {', '.join(s.usage_labels)}")
                            if s.examples:
                                lines.append(f"    Examples: {'; '.join(s.examples)}")
                    else:
                        lines.append(f"  Definitions: {NP}")
                    compounds = getattr(e, 'compounds', ())
                    idioms = getattr(e, 'idioms', ())
                    proverbs = getattr(e, 'proverbs', ())
                    lines.append(f"  Compounds: {', '.join(compounds) if compounds else NP}")
                    lines.append(f"  Idioms: {', '.join(idioms) if idioms else NP}")
                    lines.append(f"  Proverbs: {', '.join(proverbs) if proverbs else NP}")
            else:
                lines.append(f"Detail: {result.message or NP}")
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', "\n".join(lines))
        widget.configure(state='disabled')

    # --- Apply Correction / Undo -------------------------------------------

    def _tdk_push_undo(self, record):
        self._tdk_undo_stack.append(record)
        if len(self._tdk_undo_stack) > 200:
            self._tdk_undo_stack.pop(0)

    def _tdk_apply_correction(self):
        """Updates ONLY the active dataset's row's tdk_segmentation
        metadata -- never `gloss` (Gloss is handled entirely by the main
        table / Auto-Glossing Tool, not this tool), never the label, never
        MatrixLang/EmbedLang (neither depends on segmentation, so no
        recomputation is needed), never a row in another dataset."""
        if self._tdk_current is None:
            messagebox.showinfo(
                "TDK Checker",
                "Apply Correction requires a token opened from the main table "
                "(use \"Open in TDK Checker\" on a selected row).")
            return
        bidx, ridx = self._tdk_current['bidx'], self._tdk_current['ridx']
        try:
            current_dataset_id = self.datasets[self._active_dataset_index].get('id')
        except Exception:
            current_dataset_id = None
        if current_dataset_id != self._tdk_current.get('dataset_id'):
            messagebox.showerror("TDK Checker", "The active dataset has changed; cannot apply this correction.")
            return
        if bidx >= len(self.blocks) or ridx >= len(self.blocks[bidx]):
            messagebox.showerror("TDK Checker", "The original row no longer exists.")
            return

        row = self.blocks[bidx][ridx]
        token = self._tdk_token_var.get().strip()
        if str(row.get('token', '') or '') != token:
            messagebox.showerror(
                "TDK Checker", "The token in this row no longer matches what TDK Checker is showing; "
                                "re-open it from the main table before applying.")
            return

        root = self._tdk_root_var.get().strip()
        seg_text = self._tdk_segments_var.get().strip()
        parsed = tdk_parser.segments_from_text(token, root, seg_text)

        old = {'tdk_segmentation': copy.deepcopy(row.get('tdk_segmentation'))}
        new_segmentation = {'root': parsed.root, 'segments': list(parsed.segments),
                             'source': 'manual', 'success': parsed.success}

        if old['tdk_segmentation'] == new_segmentation:
            return  # nothing changed -- no-op, no dirty flag, no undo entry

        self._tdk_push_undo({'bidx': bidx, 'ridx': ridx, 'old': old,
                              'new': {'tdk_segmentation': new_segmentation}})
        row['tdk_segmentation'] = new_segmentation
        self._mark_dirty()
        self._rebuild_grid_from_model(select_row=self._tdk_current.get('vis_r'), select_col=3)

        if not parsed.success:
            messagebox.showwarning(
                "TDK Checker",
                "Applied, but the root + segments you entered do not reconstruct the token exactly "
                "-- double-check the segmentation.")

    def _tdk_undo_last(self):
        if not self._tdk_undo_stack:
            self.bell()
            return
        record = self._tdk_undo_stack.pop()
        bidx, ridx = record['bidx'], record['ridx']
        try:
            row = self.blocks[bidx][ridx]
            if record['old']['tdk_segmentation'] is not None:
                row['tdk_segmentation'] = record['old']['tdk_segmentation']
            elif 'tdk_segmentation' in row:
                del row['tdk_segmentation']
        except Exception:
            pass
        self._mark_dirty()
        self._rebuild_grid_from_model()
        if self._tdk_current is not None and (bidx, ridx) == (self._tdk_current['bidx'], self._tdk_current['ridx']):
            seg = record['old']['tdk_segmentation']
            if seg:
                self._tdk_root_var.set(seg.get('root', ''))
                self._tdk_segments_var.set(' + '.join(seg.get('segments', []) or []))

    # --- Find All Occurrences ----------------------------------------------

    def open_tdk_find_occurrences(self):
        """Finds every occurrence of the current token WITHIN THE ACTIVE
        DATASET ONLY (annotation_model.find_occurrences already never
        looks outside self.blocks) and lets the user apply the current
        root/segments/gloss correction to whichever occurrences they
        select -- never silently to all of them, never to another
        dataset."""
        query = self._tdk_token_var.get().strip()
        if not query:
            messagebox.showinfo("Find All Occurrences", "Enter or select a token first.")
            return
        items = annotation_model.find_occurrences(self.blocks, self._row_index_map, self._sep_rows, query)
        if not items:
            messagebox.showinfo("Find All Occurrences", f"No occurrences of '{query}' found in this dataset.")
            return

        win = tk.Toplevel(self)
        win.title('TDK Checker — Find All Occurrences')
        win.configure(bg=DARK_BG)
        win.transient(self)
        win.geometry('520x360')

        frm = ttk.Frame(win, style='Dark.TFrame')
        frm.pack(fill='both', expand=True, padx=10, pady=10)

        cols = ("sentence_id", "token_index", "token")
        headers = {"sentence_id": "Sentence ID", "token_index": "Token Index", "token": "Token"}
        tree = ttk.Treeview(frm, columns=cols, show='headings', style='Conc.Treeview', selectmode='extended')
        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=140, anchor='w')
        tree.pack(fill='both', expand=True)

        for i, it in enumerate(items):
            row = self.blocks[it['bidx']][it['ridx']]
            sid = self._tdk_sentence_id_for_block(it['bidx'])
            tree.insert("", "end", iid=str(i), values=(sid, row.get('idx', ''), row.get('token', '')))

        def _apply_to_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Find All Occurrences", "Select at least one occurrence first.")
                return
            root = self._tdk_root_var.get().strip()
            seg_text = self._tdk_segments_var.get().strip()
            applied = 0
            for iid in sel:
                idx = int(iid)
                it = items[idx]
                bidx, ridx = it['bidx'], it['ridx']
                try:
                    row = self.blocks[bidx][ridx]
                except Exception:
                    continue
                tok = str(row.get('token', '') or '')
                parsed = tdk_parser.segments_from_text(tok, root, seg_text)
                old = {'tdk_segmentation': copy.deepcopy(row.get('tdk_segmentation'))}
                new_segmentation = {'root': parsed.root, 'segments': list(parsed.segments),
                                     'source': 'manual', 'success': parsed.success}
                self._tdk_push_undo({'bidx': bidx, 'ridx': ridx, 'old': old,
                                      'new': {'tdk_segmentation': new_segmentation}})
                row['tdk_segmentation'] = new_segmentation
                applied += 1
            if applied:
                self._mark_dirty()
                self._rebuild_grid_from_model()
            messagebox.showinfo("Find All Occurrences", f"Applied to {applied} occurrence(s) in this dataset.")

        btns = ttk.Frame(frm, style='Dark.TFrame')
        btns.pack(fill='x', pady=(8, 0))
        ttk.Button(btns, text="Apply to Selected", style='Dark.TButton', command=_apply_to_selected).pack(
            side='left')
        ttk.Button(btns, text="Close", style='Dark.TButton', command=win.destroy).pack(side='right')

    def __init__(self):
        super().__init__()

        # Make relative resource file opens work in both dev and packaged app
        self._set_runtime_workdir()

        # Ensure menu callbacks exist even if refactors happen
        if not hasattr(self, 'open_auto_glossing_tool') and hasattr(self, '_open_auto_glossing_tool_impl'):
            self.open_auto_glossing_tool = self._open_auto_glossing_tool_impl

        self.title("TREN Annotator")
        self.geometry("1200x720")
        self.configure(bg=DARK_BG)
        self._init_style()

        self.annotator = None
        self._reranker_bundle = None
        self._reranker_load_attempted = False
        # Pre-reranker (rule-based) text from the most recent
        # _run_annotation_pipeline() call -- kept solely so the confidence
        # layer (confidence.py) can recover each token's rule-based label
        # without a second, expensive Annotator.annotate() call. Never
        # shown in the UI and never persisted.
        self._last_rule_based_output = ""
        self.current_text = ""
        self.current_output = ""
        self.current_path = None

        self._core_headers = ["Token", "Item", "Label", "Gloss"]
        self.cfg = DEFAULTS.copy()

        # Project dirty-state tracking (see _mark_dirty/_mark_clean/
        # _has_unsaved_progress). A real flag, not "does data exist" --
        # otherwise closing right after a successful save would still
        # prompt to save again.
        self._dirty = False
        self._suppress_dirty = False

        # Multiple annotation datasets. self.blocks/_extra_headers always
        # mirror the *active* dataset (self.datasets[self._active_dataset_index])
        # -- see _sync_active_dataset_from_live/_load_dataset_into_live. While
        # a dataset is active, self.blocks IS that dataset's "blocks" list
        # (same object), which is what lets every existing self.blocks-mutating
        # method keep working unchanged.
        self.datasets = [annotation_model.make_dataset("Data 1", "", [], [])]
        self._active_dataset_index = 0
        self.blocks = self.datasets[0]["blocks"]
        self._extra_headers = self.datasets[0]["extra_headers"]
        self._row_index_map = {}
        self._sep_rows = set()
        self._dataset_tabs_frame = None

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._bind_keys()


        self._last_pos = None
        self._relabel_busy = False

        self._active_area = "text"  # "text" or "sheet"

        # search dialog state
        self._search_win = None
        self._search_var = tk.StringVar(value="")
        self._search_target = "text"
        self._search_matches = []  # list of match locations
        self._search_index = -1
        self._search_after_id = None

        self._conc_win = None
        self._sentence_win = None
        self._freq_win = None
        self._ag_win = None

        # Confidence Review Tool state: focused sequential review of any
        # label, filterable by label and/or confidence band (see
        # _review_label_filter/_review_band_filter below), with per-row
        # reviewed-status tracking (confidence.py) -- no remembered/learned
        # corrections.
        self._uid_win = None
        self._uid_items = []
        self._uid_i = 0
        self._uid_mode = 'uid'   # 'uid' | 'occurrences' -- internal only, no visible control
        self._uid_query = ''    # last Find-All-Occurrences query
        self._uid_search_var = None
        self._uid_evidence_var = None
        # Confidence-review filter state (confidence.py).
        #
        # _review_view_mode is the primary switch:
        #   'all_uncertain'  -- default. Every token (any of the 7 labels)
        #                       whose confidence record has
        #                       review_recommended=True (confidence.
        #                       is_review_required) -- never label-based.
        #   'checkbox_based' -- the label/band checkboxes below decide the
        #                       list, exactly as before this default
        #                       changed (this is what "UID Only" and
        #                       "Custom" in the view combobox both use;
        #                       they differ only in which checkboxes are
        #                       set, not in the filtering mechanism).
        # _review_label_filter/_review_band_filter's own defaults
        # ({"UID"}/no restriction) reproduce the tool's original,
        # pre-confidence-layer behavior and are what "UID Only" selects.
        self._review_view_mode = 'all_uncertain'
        self._review_view_var = None
        self._review_label_filter = {"UID"}
        self._review_band_filter = set()
        self._review_hide_reviewed = False
        self._review_label_check_vars = {}
        self._review_band_check_vars = {}
        self._review_hide_reviewed_var = None
        # Scoped undo history for Confidence Review Tool Apply actions only (no
        # app-wide undo/redo mechanism exists to hook into).
        self._uid_undo_stack = []
        # Scoped undo (single most-recent transaction) for the main table's
        # Merge Cells command.
        self._merge_cells_undo_stack = []

        # TDK Checker state -- a separate tool from the Confidence Review
        # Tool (never replaces/renames it). _tdk_provider is created lazily
        # (see _ensure_tdk_provider) and ONLY on the first explicit
        # "Check TDK"/"Open in TDK Checker" action -- never at startup,
        # never during normal annotation. Tests may set self._tdk_provider
        # directly to a MockDictionaryProvider before triggering a lookup.
        self._tdk_win = None
        self._tdk_provider = None
        self._tdk_parser_annotator = None
        # Background lookup threads never call any Tk API directly (not
        # even `after()` -- that is only safe from the main thread/an
        # existing after() callback). They only ever put results on this
        # thread-safe queue; a poller scheduled exclusively via after()
        # from the main thread drains it. See _tdk_check_all/_tdk_poll_results.
        self._tdk_result_queue = queue.Queue()
        self._tdk_poll_scheduled = False
        self._tdk_current = None  # {'bidx','ridx','vis_r'} of the row currently loaded, or None if opened standalone
        self._tdk_parse_result = None
        self._tdk_lookup_generation = 0  # bumped per lookup request; stale async responses are dropped
        # The exact {'full','root','segments'} query snapshot Check TDK was
        # last run against, and the results it returned -- compared against
        # the CURRENT field values any time root/segments change, so a
        # displayed result can be marked STALE_RESULT the moment it no
        # longer corresponds to what the user is now editing (see
        # _tdk_mark_stale_if_needed / _tdk_check_all).
        self._tdk_last_query_snapshot = None
        self._tdk_last_results = []
        self._tdk_results_stale = False
        self._tdk_undo_stack = []
        self._tdk_mode = 'single'  # 'single' | 'occurrences'
        self._tdk_items = []
        self._tdk_i = 0
        self._tdk_query = ''

        # concordance (KWIC) state
        self._conc_query_var = tk.StringVar(value="")
        self._conc_ctx_var = tk.IntVar(value=30)
        self._conc_ci_var = tk.BooleanVar(value=True)
        self._conc_regex_var = tk.BooleanVar(value=False)
        self._conc_tree = None
        self._conc_count_var = tk.StringVar(value="0 matches")
        self._conc_hit_spans = {}

        # grid clipboard (for copy/cut/paste)
        self._grid_clipboard = None

        # full edit window
        self._full_win = None
        self._full_sheet = None
        self._active_sheet = None  # main sheet or full sheet, whichever has focus

        # auto-restore last project (if exists)
        try:
            self.after(50, self._auto_restore_last_project)
        except Exception:
            pass

        # confirm on close
        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close_request)
        except Exception:
            pass

    # Auto-gloss helpers
    def _split_mixed_token(self, token: str):
        """Split token into (stem_hint, suffix_hint). If no delimiter, allow delimiterless parsing."""
        s = "" if token is None else str(token).strip()
        if not s:
            return "", ""
        if "-" in s:
            stem, suf = s.rsplit("-", 1)
            return stem, suf
        if "'" in s:
            stem, suf = s.rsplit("'", 1)
            return stem, suf
        return "", s

    def _suffix_registry(self):
        """Inclusive suffix registry. Some suffixes return alternative glosses."""
        V_A = "ae"
        V_I = "ıiuü"

        def rx(p):
            return re.compile(p)

        return [
            # Inflectional
            ("ABL", 3, [rx(rf"^[dt][{V_A}]n$")], "ABL"),
            ("LOC", 2, [rx(rf"^[dt][{V_A}]$")], "LOC"),
            ("GEN", 3, [rx(rf"^n?[{V_I}]n$")], "GEN"),
            ("DAT", 2, [rx(rf"^y?[{V_A}]$")], "DAT"),
            ("ACC", 2, [rx(rf"^y?[{V_I}]$")], "ACC"),
            ("INS_COM", 3, [rx(rf"^y?l[{V_A}]$")], ["INS", "COM"]),
            ("INS_COM_SPEECH", 3, [rx(r"^n[ae]n$")], ["INS", "COM"]),
            ("PL", 3, [rx(rf"^l[{V_A}]r$")], "PL"),

            # Possessive (heuristic)
            ("POSS.1SG", 2, [rx(rf"^[{V_I}]m$")], "POSS.1SG"),
            ("POSS.2SG", 2, [rx(rf"^[{V_I}]n$")], "POSS.2SG"),
            ("POSS.3SG", 2, [rx(rf"^s?[{V_I}]$")], "POSS.3SG"),
            ("POSS.3PL", 4, [rx(rf"^l[{V_A}]r[{V_I}]$")], "POSS.3PL"),
            ("POSS.PLX", 4, [rx(rf"^[{V_I}].z$")], ["POSS.1PL", "POSS.2PL"]),

            # Derivational (inclusive)
            ("AGT", 3, [rx(rf"^[cç][{V_I}]$")], "AGT"),
            ("NMLZ", 3, [rx(rf"^l[{V_I}]k$")], "NMLZ"),
            ("PRIV", 3, [rx(rf"^s[{V_I}]z$")], "PRIV"),
            ("ATTR", 2, [rx(rf"^l[{V_I}]$")], "ATTR"),
            ("SIM_ADV", 2, [rx(rf"^c[{V_A}]$")], ["SIM", "ADV"]),
            ("SIM", 3, [rx(rf"^ms[{V_I}]$")], "SIM"),

            # verbal nominalizers/participles
            ("NMLZ_V", 2, [rx(rf"^m[{V_A}]$")], "NMLZ"),
            ("PTCP", 2, [rx(rf"^[{V_A}]n$")], "PTCP"),
            ("NMLZ_DIK", 4, [rx(rf"^[dt][{V_I}]k$")], "NMLZ"),
        ]

    def _auto_gloss_candidates(self, token: str):
        """Return candidate gloss strings: stem-GLOSS-GLOSS... (inclusive)."""
        stem_hint, suf_hint = self._split_mixed_token(token)
        s = "" if token is None else str(token).strip()
        if not s:
            return []

        remaining = "" if suf_hint is None else str(suf_hint).strip().casefold()
        if not remaining:
            return []

        registry = self._suffix_registry()
        chain = []
        guard = 0

        while remaining and guard < 12:
            guard += 1
            matched = False

            for L in (5, 4, 3, 2, 1):
                if len(remaining) < L:
                    continue
                cand = remaining[-L:]

                for _name, _maxlen, patterns, glosses in registry:
                    if L > _maxlen:
                        continue
                    ok = False
                    for pat in patterns:
                        try:
                            if pat.match(cand):
                                ok = True
                                break
                        except Exception:
                            continue
                    if not ok:
                        continue

                    remaining = remaining[:-L]
                    if isinstance(glosses, list):
                        chain.append([str(x) for x in glosses])
                    else:
                        chain.append([str(glosses)])
                    matched = True
                    break

                if matched:
                    break

            if not matched:
                break

        if not chain:
            return []

        stem = (stem_hint or "").strip()
        if not stem:
            consumed = len(str(suf_hint).strip()) - len(remaining)
            base = str(suf_hint).strip()
            stem = base[:max(0, len(base) - consumed)].strip() or str(token).strip()

        chain = list(reversed(chain))

        cands = [""]
        for alts in chain:
            new_cands = []
            for pref in cands:
                for g in alts:
                    new_cands.append((pref + "-" + g) if pref else g)
            cands = new_cands

        out = []
        for gseq in cands:
            if gseq:
                out.append(stem + "-" + gseq)

        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    def _auto_gloss_mixed_token(self, token: str):
        cands = self._auto_gloss_candidates(token)
        return cands[0] if cands else ""

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('Dark.TButton', background="#2e2e2e", foreground=DARK_FG,
                        padding=6, focusthickness=3, focuscolor=ACCENT)
        style.map('Dark.TButton', background=[('active', '#3a3a3a')])
        style.configure('Dark.TCheckbutton', background=DARK_BG, foreground=DARK_FG)
        style.map('Dark.TCheckbutton', background=[('active', DARK_BG)])

        # dataset tab bar
        style.configure('DatasetTab.TButton', background="#2a2a2a", foreground="#a0a0a0",
                         padding=(10, 3))
        style.map('DatasetTab.TButton', background=[('active', '#3a3a3a')])
        style.configure('DatasetTabActive.TButton', background=ACCENT, foreground="white",
                         padding=(10, 3))
        style.map('DatasetTabActive.TButton', background=[('active', ACCENT)])

        # generic dark ttk widgets
        style.configure('Dark.TFrame', background=DARK_BG)
        style.configure('Dark.TLabel', background=DARK_BG, foreground=DARK_FG)
        style.configure('Dark.TEntry', fieldbackground="#1b1b1b", background=DARK_BG,
                        foreground=DARK_FG, insertcolor="white")
        style.configure('Dark.TSpinbox', fieldbackground="#1b1b1b", background=DARK_BG,
                        foreground=DARK_FG, insertcolor="white")

        # concordance treeview
        style.configure('Conc.Treeview',
                        background="#1b1b1b",
                        fieldbackground="#1b1b1b",
                        foreground=DARK_FG,
                        bordercolor="#404040",
                        lightcolor="#404040",
                        darkcolor="#404040")
        style.map('Conc.Treeview',
                  background=[('selected', '#2b2b2b')],
                  foreground=[('selected', 'white')])

        style.configure('Conc.Treeview.Heading',
                        background="#171717",
                        foreground=DARK_FG,
                        bordercolor="#404040")

    def _build_menu(self):
        menubar = tk.Menu(self, tearoff=False)
        filem = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        filem.add_command(label="Open Input (⌘O / Ctrl+O)", command=self.open_input)
        filem.add_command(label="Run (⌘R / Ctrl+R)", command=self.run_pipeline)
        filem.add_command(label="Export Table... (⌘S / Ctrl+S)", command=self.save_output)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filem)

        projm = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        projm.add_command(label="New Project", command=self.new_project)
        projm.add_separator()
        projm.add_command(label="Open Project Save...", command=self.open_project_save)
        projm.add_command(label="Save Project Progress...", command=self.save_project_progress)
        menubar.add_cascade(label="Project", menu=projm)

        annm = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        annm.add_command(label="Add New Column...", command=self._add_new_column_dialog)
        annm.add_separator()
        annm.add_command(label="Cut Cell(s)", command=self.cut_selected_cells)
        annm.add_command(label="Copy Cell(s)", command=self.copy_selected_cells)
        annm.add_command(label="Paste Cell(s)", command=self.paste_selected_cells)
        annm.add_command(label="Clear Selected Cell(s)", command=self.clear_selected_cells)
        annm.add_separator()
        annm.add_command(label="Insert Row Before", command=self.insert_row_before)
        annm.add_command(label="Remove Row", command=self.remove_selected_row)
        annm.add_separator()
        annm.add_command(label="Merge Cells", command=self.merge_selected_cells)
        annm.add_command(label="Undo Merge Cells", command=self.undo_merge_cells)
        menubar.add_cascade(label="Annotation", menu=annm)

        editm = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        editm.add_command(label="View Full Edit Window...", command=self.open_full_edit_window)
        menubar.add_cascade(label="Edit Window", menu=editm)

        toolsm = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        toolsm.add_command(label="Auto-Glossing Tool...", command=self.open_auto_glossing_tool)
        toolsm.add_command(label="Confidence Review Tool...", command=self.open_uid_review_tool)
        # A separate tool from the Confidence Review Tool -- never replaces
        # or renames it. See open_tdk_checker_tool/open_tdk_checker_from_grid.
        toolsm.add_command(label="TDK Checker...", command=self.open_tdk_checker_tool)
        toolsm.add_separator()
        toolsm.add_command(label="Concordance (KWIC)...", command=self.open_concordance)
        toolsm.add_command(label="Show Sentence (Context)", command=self.show_sentence_context)
        toolsm.add_command(label="Word Frequency List...", command=self.open_word_frequency)
        menubar.add_cascade(label="Tools", menu=toolsm)

        self.config(menu=menubar)


    # Window close handling
    # Project dirty-state tracking
    #
    # self._dirty is a real flag, not a "does the project contain data"
    # test -- checking for data would keep prompting to save even
    # immediately after a successful save with nothing further changed.
    # Every mutation call site (manual table edits, Confidence Review Tool Apply/
    # Undo, Merge Cells/Undo, running annotation, adding a dataset, typing
    # in the input editor) calls _mark_dirty(); switching dataset tabs and
    # exporting deliberately never do.
    def _mark_dirty(self):
        self._dirty = True

    def _mark_clean(self):
        self._dirty = False

    def _on_text_modified(self, event=None):
        """<<Modified>> fires on any change to txt_input, including the
        programmatic delete/insert used to display a different dataset's
        text -- _set_txt_input_text sets _suppress_dirty around those so
        this handler treats them as "not a user edit"."""
        if getattr(self, '_suppress_dirty', False):
            try:
                self.txt_input.edit_modified(False)
            except Exception:
                pass
            return
        try:
            if self.txt_input.edit_modified():
                self._mark_dirty()
                self.txt_input.edit_modified(False)
        except Exception:
            pass

    def _set_txt_input_text(self, text):
        """Replace the input editor's text without marking the project
        dirty. Use this for anything that is *restoring* existing state
        (switching dataset tabs, opening/auto-restoring a project, New
        Project) rather than a user typing/pasting new content."""
        self._suppress_dirty = True
        try:
            self.txt_input.delete("1.0", "end")
            self.txt_input.insert("1.0", text or "")
        finally:
            try:
                self.txt_input.edit_modified(False)
            except Exception:
                pass
            self._suppress_dirty = False

    def _has_unsaved_progress(self):
        return getattr(self, '_dirty', False)

    def _confirm_proceed_over_unsaved_changes(self, title, message):
        """Shared unsaved-changes guard for New Project / Open Project /
        Close. Returns True if the caller may proceed with its destructive
        operation, False if it must abort and leave the current project
        completely untouched.

        - Clean project: proceeds immediately, no prompt.
        - Dirty project: asks Save (Yes) / Discard (No) / Cancel.
          - Save: proceeds only if save_project_progress() actually
            succeeded (returns True) -- a cancelled name/overwrite dialog
            or a write failure aborts the pending operation instead of
            silently discarding the work.
          - Discard: proceeds without saving.
          - Cancel: aborts; nothing changes.
        """
        if not self._has_unsaved_progress():
            return True
        res = messagebox.askyesnocancel(title, message)
        if res is None:
            return False
        if res is True:
            return self.save_project_progress()
        return True

    def _on_close_request(self):
        if not self._confirm_proceed_over_unsaved_changes(
            "Close", "Save project progress before closing?"
        ):
            return
        try:
            self.destroy()
        except Exception:
            pass

    # Multiple annotation datasets
    #
    # self.blocks / self._extra_headers are always the *live* view of
    # self.datasets[self._active_dataset_index] (see the __init__ comment).
    # Every existing self.blocks-mutating method (grid edits, merge cells,
    # Confidence Review Tool, auto-glossing, ...) keeps working unchanged because it
    # is still mutating the same list object the active dataset dict points
    # to. Switching datasets is the only place that list identity changes.
    def _sync_active_dataset_from_live(self):
        """Write the live self.blocks/_extra_headers/source-text/undo-stacks
        back into the active dataset dict. Call before anything that reads
        self.datasets directly (switching tabs, saving the project, export)."""
        if not getattr(self, 'datasets', None):
            return
        ds = self.datasets[self._active_dataset_index]
        ds['blocks'] = self.blocks
        ds['extra_headers'] = self._extra_headers
        try:
            ds['source_text'] = self.txt_input.get("1.0", "end-1c")
        except Exception:
            pass
        ds['uid_undo_stack'] = list(getattr(self, '_uid_undo_stack', []) or [])
        ds['merge_cells_undo_stack'] = list(getattr(self, '_merge_cells_undo_stack', []) or [])

    def _load_dataset_into_live(self, index):
        """Make dataset `index` active: point self.blocks/_extra_headers at
        it, restore its source text into the input editor, and rebuild the
        grid from it. Never runs the annotation pipeline -- switching tabs
        must not re-annotate or reload NLP models."""
        ds = self.datasets[index]
        self._active_dataset_index = index
        self.blocks = ds['blocks']
        self._extra_headers = ds['extra_headers']
        self._uid_undo_stack = list(ds.get('uid_undo_stack', []) or [])
        self._merge_cells_undo_stack = list(ds.get('merge_cells_undo_stack', []) or [])
        self._row_index_map = {}
        self._sep_rows = set()

        try:
            self._set_txt_input_text(ds.get('source_text', ''))
        except Exception:
            pass

        self._close_dataset_scoped_windows()
        self._renumber_tokens()
        self._rebuild_grid_from_model()
        self._update_dataset_tabs_ui()

    def _switch_dataset(self, index):
        """Switch the active dataset tab. A no-op if `index` is already
        active -- re-clicking the current tab must never rebuild the grid,
        clear scoped undo history, or prompt to save (switching tabs is
        never a save point)."""
        if not getattr(self, 'datasets', None) or index == self._active_dataset_index:
            return
        if not (0 <= index < len(self.datasets)):
            return
        self._sync_active_dataset_from_live()
        self._load_dataset_into_live(index)

    def _close_dataset_scoped_windows(self):
        """Close every auxiliary window that reads/writes self.blocks by row
        index (Confidence Review Tool, TDK Checker, Auto-Glossing Tool,
        Concordance, Word Frequency, Full Edit Window, Show Sentence). None
        of them are dataset-aware, so leaving one open across a dataset
        switch would let it silently read or edit the wrong dataset via a
        stale row index -- the TDK action "must never operate on a
        previous dataset", so this is exactly as strict for it as for
        every other row-index-based tool."""
        for attr in ('_uid_win', '_ag_win', '_conc_win', '_freq_win', '_full_win', '_tdk_win'):
            win = getattr(self, attr, None)
            if win is not None:
                try:
                    if win.winfo_exists():
                        win.destroy()
                except Exception:
                    pass
                setattr(self, attr, None)

        self._tdk_current = None
        self._tdk_parse_result = None
        self._tdk_last_query_snapshot = None
        self._tdk_last_results = []
        self._tdk_results_stale = False
        self._tdk_undo_stack = []
        self._tdk_items = []
        self._tdk_i = 0
        self._tdk_mode = 'single'
        self._tdk_query = ''

        sent = getattr(self, '_sentence_win', None)
        if sent is not None:
            win = sent[0] if isinstance(sent, tuple) else sent
            try:
                if win.winfo_exists():
                    win.destroy()
            except Exception:
                pass
            self._sentence_win = None

        self._full_sheet = None
        self._uid_items = []
        self._uid_i = 0
        self._uid_mode = 'uid'
        self._uid_query = ''

    def _build_dataset_tabs(self, parent):
        """Compact dataset tab bar, packed immediately above the annotation
        table. `parent` is the frame the table itself is packed into, so the
        tab bar is added before it."""
        frame = tk.Frame(parent, bg=DARK_BG)
        frame.pack(fill="x", pady=(0, 4))
        self._dataset_tabs_frame = frame
        self._update_dataset_tabs_ui()

    def _update_dataset_tabs_ui(self):
        """Rebuild the tab bar's buttons from self.datasets. Cheap enough to
        call after every dataset add/switch/rename -- it's a handful of
        buttons, not a data-heavy widget."""
        frame = getattr(self, '_dataset_tabs_frame', None)
        if frame is None:
            return
        for child in list(frame.winfo_children()):
            child.destroy()

        for i, ds in enumerate(getattr(self, 'datasets', []) or []):
            is_active = (i == self._active_dataset_index)
            style = 'DatasetTabActive.TButton' if is_active else 'DatasetTab.TButton'
            btn = ttk.Button(
                frame, text=str(ds.get('name', 'Data')), style=style,
                command=lambda idx=i: self._switch_dataset(idx),
            )
            btn.pack(side="left", padx=(0, 2))

        plus_btn = ttk.Button(frame, text="+", width=3, style='Dark.TButton',
                               command=self._open_add_new_data_dialog)
        plus_btn.pack(side="left", padx=(4, 0))
        self._make_tooltip(plus_btn, "Add New Data")

    def _make_tooltip(self, widget, text):
        """Minimal hover tooltip; no external dependency. Also used as this
        control's accessibility text (the '+' button itself stays a bare
        glyph so the tab bar stays compact)."""
        state = {'win': None}

        def _show(_event=None):
            if state['win'] is not None:
                return
            try:
                x = widget.winfo_rootx() + 10
                y = widget.winfo_rooty() + widget.winfo_height() + 6
            except Exception:
                return
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            try:
                tw.wm_geometry(f"+{x}+{y}")
            except Exception:
                pass
            tk.Label(tw, text=text, bg="#333333", fg=DARK_FG, relief="solid",
                     borderwidth=1, padx=6, pady=2).pack()
            state['win'] = tw

        def _hide(_event=None):
            tw = state['win']
            if tw is not None:
                try:
                    tw.destroy()
                except Exception:
                    pass
                state['win'] = None

        widget.bind("<Enter>", _show)
        widget.bind("<Leave>", _hide)

    def _ensure_annotator_ready(self):
        """Lazily create the Annotator and load the reranker bundle, exactly
        once, no matter how many datasets get annotated in this session."""
        if self.annotator is None:
            self.annotator = Annotator()
        if not self._reranker_load_attempted:
            self._reranker_bundle = reranker_integration.load_reranker_bundle()
            self._reranker_load_attempted = True

    def _run_annotation_pipeline(self, text):
        """Run the full production pipeline (annotate -> reranker ->
        matrix/embed consistency) on `text` and return the TXT-style output.
        Shared by run_pipeline (active dataset) and Add New Data (a
        new/other dataset) so both go through the exact same code path."""
        self._ensure_annotator_ready()
        out = self.annotator.annotate(text, self.cfg)
        # Stashed for the confidence layer (confidence.py) -- see
        # _attach_confidence, which diffs this against the final output to
        # recover each token's pre-reranker rule-based label. Purely an
        # internal side channel: never shown in the UI, never persisted,
        # and reading/writing it has no effect on the annotation output
        # returned below.
        self._last_rule_based_output = out
        out = reranker_integration.apply_reranker(out, self.annotator, self.cfg, self._reranker_bundle)
        out = self._ensure_matrix_embed_consistency(out)
        return out

    def _attach_confidence(self, blocks):
        """Best-effort: compute and attach the confidence layer's per-token
        records (confidence.py) to `blocks` in place. Never allowed to
        interrupt annotation -- a failure here is caught and silently
        ignored, leaving `blocks` with no/partial "confidence" keys rather
        than raising. Uses self._last_rule_based_output (set by the
        _run_annotation_pipeline call that must immediately precede this)
        to recover each token's pre-reranker rule-based label."""
        try:
            rule_blocks = annotation_model.parse_annotated_text_to_blocks(self._last_rule_based_output, [])
            confidence.attach_confidence_to_blocks(
                blocks, rule_blocks, self.annotator, self.cfg, bundle=self._reranker_bundle)
        except Exception as e:
            print(f"[confidence] failed to attach confidence data, continuing without it: {e}", file=sys.stderr)

    def _read_utf8_text_file(self, path):
        """Read `path` as strict UTF-8. Raises UnicodeDecodeError/OSError on
        failure -- callers must catch those and show a clear message, never
        fall back silently to another encoding or another input source."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _create_dataset_from_text(self, name, source_text, source_filename=None):
        """Run the production pipeline on `source_text` and return a new,
        fully independent dataset dict -- never touches self.blocks or any
        existing dataset. Raises on annotation failure; the caller must not
        create a tab/dataset in that case (no broken or empty tab).
        `source_filename` is optional, display-only metadata (a basename
        only -- see annotation_model.make_dataset, which enforces that even
        if a full path is passed here) -- source_text is always what
        actually gets annotated and saved."""
        out = self._run_annotation_pipeline(source_text)
        blocks = annotation_model.parse_annotated_text_to_blocks(out, [])
        annotation_model.renumber_tokens(blocks)
        self._attach_confidence(blocks)
        return annotation_model.make_dataset(name, source_text, blocks, [], source_filename=source_filename)

    def _open_add_new_data_dialog(self):
        default_name = annotation_model.next_default_dataset_name(
            [d.get('name', '') for d in self.datasets]
        )

        win = tk.Toplevel(self)
        win.title("Add New Data")
        win.configure(bg=DARK_BG)
        win.transient(self)
        win.geometry("480x440")
        win.minsize(420, 400)

        frm = ttk.Frame(win, style='Dark.TFrame')
        frm.pack(fill='both', expand=True, padx=14, pady=14)

        ttk.Label(frm, text="Name:", style='Dark.TLabel').pack(anchor='w')
        name_var = tk.StringVar(value=default_name)
        ttk.Entry(frm, textvariable=name_var, style='Dark.TEntry', width=32).pack(anchor='w', pady=(2, 10))

        mode_var = tk.StringVar(value='enter')

        ttk.Radiobutton(frm, text="Open New File", value='file', variable=mode_var,
                         command=lambda: _set_mode('file')).pack(anchor='w')

        file_row = ttk.Frame(frm, style='Dark.TFrame')
        file_row.pack(fill='x', padx=(18, 0), pady=(2, 8))
        ttk.Label(file_row, text="File:", style='Dark.TLabel').pack(side='left')
        # Kept only for the lifetime of this dialog, solely to actually open
        # the file if/when Create is pressed -- never stored on the dataset
        # or written to .trenproj. Only its basename (file_display_var,
        # below) ever leaves this closure, via source_filename in _confirm.
        file_path_var = tk.StringVar(value='')
        file_display_var = tk.StringVar(value='(none selected)')
        file_label = ttk.Label(file_row, textvariable=file_display_var, style='Dark.TLabel',
                                foreground="#a0a0a0", width=26)
        file_label.pack(side='left', padx=(4, 6))

        def _browse():
            path = filedialog.askopenfilename(
                title="Open New File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if not path:
                return  # cancelling the file chooser changes nothing
            file_path_var.set(path)
            file_display_var.set(os.path.basename(path))
            # Only steal the Name field while it still holds its own
            # untouched "Data N" default -- never overwrite a name the user
            # typed, and never re-stomp a name already set from a previous
            # file pick.
            if name_var.get().strip() == default_name:
                stem = os.path.splitext(os.path.basename(path))[0].strip()
                if stem:
                    name_var.set(stem)

        browse_btn = ttk.Button(file_row, text="Browse…", style='Dark.TButton', command=_browse)
        browse_btn.pack(side='left')

        ttk.Radiobutton(frm, text="Enter New Text", value='enter', variable=mode_var,
                         command=lambda: _set_mode('enter')).pack(anchor='w')
        ttk.Radiobutton(frm, text="Re-run Current Text", value='rerun', variable=mode_var,
                         command=lambda: _set_mode('rerun')).pack(anchor='w', pady=(0, 8))

        ttk.Label(frm, text="Text (used only for “Enter New Text”):",
                  style='Dark.TLabel').pack(anchor='w')
        text_box = ScrolledText(frm, wrap="word", height=8, bg="#1b1b1b", fg=DARK_FG,
                                 insertbackground="white")
        text_box.pack(fill='both', expand=True, pady=(2, 10))

        def _set_mode(mode):
            # Text and the selected file path are always preserved across
            # mode switches while the dialog stays open -- only which
            # controls are enabled changes; nothing is ever cleared here.
            if mode == 'enter':
                text_box.configure(state='normal', bg="#1b1b1b", fg=DARK_FG)
            else:
                text_box.configure(state='disabled', bg="#161616", fg="#666666")
            browse_btn.configure(state=('normal' if mode == 'file' else 'disabled'))

        _set_mode('enter')

        status_var = tk.StringVar(value='')
        ttk.Label(frm, textvariable=status_var, style='Dark.TLabel', foreground="#ffcc66").pack(anchor='w')

        def _confirm():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Add New Data", "Enter a name for the new dataset.")
                return

            mode = mode_var.get()
            source_filename = None

            if mode == 'file':
                path = file_path_var.get()
                if not path:
                    messagebox.showwarning("Add New Data", "Choose a file to open.")
                    return
                try:
                    new_text = self._read_utf8_text_file(path)
                except UnicodeDecodeError as e:
                    messagebox.showerror(
                        "Add New Data", f"The selected file is not valid UTF-8:\n{e}")
                    return
                except OSError as e:
                    messagebox.showerror("Add New Data", f"Could not read the selected file:\n{e}")
                    return
                if not new_text.strip():
                    messagebox.showwarning("Add New Data", "The selected file is empty.")
                    return
                # Only the basename ever leaves this dialog -- `path` (the
                # full local path) is used here to read the file and then
                # discarded; it is never stored on the dataset or persisted.
                source_filename = os.path.basename(path)
            elif mode == 'enter':
                new_text = text_box.get("1.0", "end-1c")
                if not new_text.strip():
                    messagebox.showwarning("Add New Data", "Enter text to annotate.")
                    return
            else:
                new_text = self.txt_input.get("1.0", "end-1c")
                if not new_text.strip():
                    messagebox.showwarning(
                        "Add New Data", "The active dataset has no source text to re-run.")
                    return

            status_var.set("Running annotation pipeline…")
            win.update_idletasks()
            try:
                new_ds = self._create_dataset_from_text(name, new_text, source_filename=source_filename)
            except Exception as e:
                # Annotation failed -- create no tab/dataset at all.
                messagebox.showerror("Add New Data", f"Annotation failed:\n{e}")
                status_var.set("")
                return

            self._sync_active_dataset_from_live()
            self.datasets.append(new_ds)
            self._mark_dirty()
            self._load_dataset_into_live(len(self.datasets) - 1)
            win.destroy()

        btns = ttk.Frame(frm, style='Dark.TFrame')
        btns.pack(fill='x')
        ttk.Button(btns, text="Create", style='Dark.TButton', command=_confirm).pack(side='left')
        ttk.Button(btns, text="Cancel", style='Dark.TButton', command=win.destroy).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        win.grab_set()

    # Project Save/Load
    def new_project(self):
        if not self._confirm_proceed_over_unsaved_changes(
            "New Project", "Save changes to the current project before starting a new one?"
        ):
            return

        self.cfg = DEFAULTS.copy()
        self.datasets = [annotation_model.make_dataset("Data 1", "", [], [])]
        self._close_dataset_scoped_windows()
        self._load_dataset_into_live(0)
        self._mark_clean()

        try:
            os.makedirs(APP_DIR, exist_ok=True)
            if os.path.isfile(LAST_PROJECT_PTR):
                os.remove(LAST_PROJECT_PTR)
        except Exception:
            pass

        self._grid_clipboard = None
        self._search_matches = []
        self._search_index = -1
        self._last_pos = None

    def save_project_progress(self):
        """Save the current project to a named `.trenproj` file.

        Returns True if the file was actually written, False if the user
        cancelled the name/overwrite prompt or the write itself failed --
        callers (in particular _on_close_request) must treat False as "no
        save happened" and must not discard unsaved work on that basis."""
        name = simpledialog.askstring("Save Project", "Name your project save:")
        if not name:
            return False

        try:
            os.makedirs(APP_DIR, exist_ok=True)
        except Exception:
            pass

        path = os.path.join(APP_DIR, name + PROJECT_EXT)

        # confirm overwrite
        if os.path.isfile(path):
            if messagebox.askyesno("Overwrite?", f"A save named '{name}' already exists. Overwrite?") is False:
                return False

        try:
            text_cursor = self.txt_input.index("insert")
        except Exception:
            text_cursor = None

        try:
            sheet_sel = self.sheet.get_currently_selected() if self.sheet is not None else None
            grid_pos = (sheet_sel.row, sheet_sel.column) if sheet_sel is not None else None
        except Exception:
            grid_pos = None

        self._sync_active_dataset_from_live()

        payload = {
            "name": name,
            "cfg": self.cfg,
            "text_cursor": text_cursor,
            "grid_pos": grid_pos,
        }
        # datasets_to_payload stamps "version": CURRENT_PROJECT_SCHEMA_VERSION.
        payload.update(annotation_model.datasets_to_payload(self.datasets, self._active_dataset_index))

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return False

        try:
            with open(LAST_PROJECT_PTR, "w", encoding="utf-8") as f:
                json.dump({"path": path}, f)
        except Exception:
            pass

        self._mark_clean()
        messagebox.showinfo("Project saved", f"Saved project:\n{path}")
        return True

    def open_project_save(self):
        path = filedialog.askopenfilename(
            title="Open Project Save",
            initialdir=APP_DIR if os.path.isdir(APP_DIR) else None,
            filetypes=[("Project Saves", f"*{PROJECT_EXT}"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            messagebox.showerror("Open error", str(e))
            return

        try:
            datasets, active_index = annotation_model.datasets_from_payload(payload)
        except ValueError as e:
            messagebox.showerror("Open error", f"Malformed project file:\n{e}")
            return

        # The file is now fully read and validated -- only now does opening
        # it put the current project at risk, so the unsaved-changes guard
        # runs here rather than before the file chooser (cancelling the
        # chooser, or an unreadable/malformed file, must never even ask).
        if not self._confirm_proceed_over_unsaved_changes(
            "Open Project", "Save changes to the current project before opening another one?"
        ):
            return

        self.cfg = payload.get("cfg", DEFAULTS.copy())
        self.datasets = datasets
        self._close_dataset_scoped_windows()
        self._load_dataset_into_live(active_index)
        self._mark_clean()

        try:
            if payload.get("text_cursor"):
                self.txt_input.mark_set("insert", payload.get("text_cursor"))
        except Exception:
            pass

        try:
            gp = payload.get("grid_pos")
            if gp and self.sheet is not None:
                r, c = gp
                self.sheet.select_cell(r, c)
                self.sheet.see(r, c)
        except Exception:
            pass

        try:
            os.makedirs(APP_DIR, exist_ok=True)
            with open(LAST_PROJECT_PTR, "w", encoding="utf-8") as f:
                json.dump({"path": path}, f)
        except Exception:
            pass

        messagebox.showinfo("Project loaded", f"Loaded project:\n{path}")

    def _auto_restore_last_project(self):
        try:
            if not os.path.isfile(LAST_PROJECT_PTR):
                return
            with open(LAST_PROJECT_PTR, "r", encoding="utf-8") as f:
                ptr = json.load(f)
            path = ptr.get("path")
            if not path or not os.path.isfile(path):
                return
        except Exception:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        try:
            datasets, active_index = annotation_model.datasets_from_payload(payload)
        except ValueError:
            # Auto-restore is silent-best-effort; a malformed pointer target
            # just means the app opens empty instead of crashing on startup.
            return

        try:
            self.cfg = payload.get("cfg", DEFAULTS.copy())
            self.datasets = datasets
            self._close_dataset_scoped_windows()
            self._load_dataset_into_live(active_index)
            self._mark_clean()
        except Exception:
            return

        try:
            if payload.get("text_cursor"):
                self.txt_input.mark_set("insert", payload.get("text_cursor"))
        except Exception:
            pass

        try:
            gp = payload.get("grid_pos")
            if gp and self.sheet is not None:
                r, c = gp
                self.sheet.select_cell(r, c)
                self.sheet.see(r, c)
        except Exception:
            pass

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=DARK_BG)
        bar.pack(fill="x", padx=8, pady=6)

        def mkcheck(text, varname):
            v = tk.BooleanVar(value=self.cfg.get(varname, False))
            cb = ttk.Checkbutton(bar, text=text, style='Dark.TCheckbutton',
                                 variable=v, command=lambda:self._toggle(varname, v.get()))
            cb.pack(side="left", padx=8)
            return v

        self.v_lang = mkcheck("Language per item", "FEATURE_LANGUAGE_PER_ITEM")
        self.v_mlx  = mkcheck("MatrixLang", "FEATURE_MATRIX_LANGUAGE")
        self.v_emb  = mkcheck("EmbedLang", "FEATURE_EMBEDDED_LANGUAGE")
        self.v_ner  = mkcheck("NER", "NER_ENABLED")

        btn_run = ttk.Button(bar, text="Run", command=self.run_pipeline, style='Dark.TButton')
        btn_run.pack(side="right", padx=4)

        btn_open = ttk.Button(bar, text="Open", command=self.open_input, style='Dark.TButton')
        btn_open.pack(side="right", padx=4)

        btn_save = ttk.Button(bar, text="Export", command=self.save_output, style='Dark.TButton')
        btn_save.pack(side="right", padx=4)

    def _build_body(self):
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=DARK_BG)
        left.pack(side="left", fill="both", expand=True, padx=(8,4), pady=(0,8))

        right = tk.Frame(main, bg=DARK_BG, width=260)
        right.pack(side="right", fill="both", expand=False, padx=(4,8), pady=(0,8))

        # INPUT editor
        self.txt_input = ScrolledText(left, wrap="word", bg="#1b1b1b", fg=DARK_FG, insertbackground="white")
        self.txt_input.pack(fill="both", expand=True)
        self.txt_input.configure(font=("Menlo", 13))
        try:
            self.txt_input.tag_configure("KWIC_HIT", background="#3a3a3a")
        except Exception:
            pass

        # Track active area
        self.txt_input.bind("<FocusIn>", lambda e: setattr(self, "_active_area", "text"))
        self.txt_input.bind("<Button-1>", lambda e: setattr(self, "_active_area", "text"))

        # Dirty-state tracking for organic edits (typing/pasting). Text.
        # edit_modified() fires <<Modified>> on any change; programmatic
        # text replacement (dataset switch, project open/new) goes through
        # _set_txt_input_text, which suppresses this so it never marks the
        # project dirty on its own.
        self.txt_input.bind("<<Modified>>", self._on_text_modified)

        # Dataset tab bar (immediately above the annotation table)
        self._build_dataset_tabs(right)

        # OUTPUT grid
        tbl_frame = tk.Frame(right, bg=DARK_BG)
        tbl_frame.pack(fill="both", expand=True)

        self.sheet = None
        if tksheet is None:
            warn = tk.Label(
                tbl_frame,
                text=(
                    "tksheet not installed.\n"
                    "Grid editor requires: pip install tksheet\n\n"
                    "You can still open and run, but grid editing is disabled."
                ),
                bg=DARK_BG, fg="#ffcc66", justify="left"
            )
            warn.pack(fill="both", expand=True, padx=12, pady=12)
        else:
            self.sheet = tksheet.Sheet(
                tbl_frame,
                headers=self._all_headers(),
                data=[[]],
                show_x_scrollbar=True,
                show_y_scrollbar=True,
                show_top_left=True,
                show_row_index=True,
                show_header=True,
                show_table=True,
            )

            self.sheet.enable_bindings((
                "single_select",
                "drag_select",
                "ctrl_select",
                "shift_select",
                "cell_select",
                "edit_cell",
                "double_click_edit",
                "arrowkeys",
            ))
            try:
                self.sheet.set_options(
                    table_bg="#1b1b1b", table_fg=DARK_FG, table_grid_fg="#404040",
                    index_bg="#171717", index_fg=DARK_FG, index_grid_fg="#404040",
                    header_bg="#171717", header_fg=DARK_FG, header_grid_fg="#404040",
                    top_left_bg="#171717", top_left_fg=DARK_FG,
                    selected_rows_bg="#2b2b2b", selected_rows_fg="white",
                    selected_columns_bg="#2b2b2b", selected_columns_fg="white",
                    selected_cells_bg="#2b2b2b", selected_cells_fg="white",
                )
            except Exception:
                pass
            self.sheet.grid(row=0, column=0, sticky="nsew")
            tbl_frame.rowconfigure(0, weight=1)
            tbl_frame.columnconfigure(0, weight=1)

            # sheet event
            self.sheet.extra_bindings([
                ("end_edit_cell", self._on_sheet_end_edit),
                ("arrowkeys", self._on_sheet_arrow),
                ("cell_select", self._on_sheet_cell_select),
            ])
            try:
                self.sheet.bind("<FocusIn>", lambda e: setattr(self, "_active_area", "sheet"))
                self.sheet.bind("<Button-1>", lambda e: setattr(self, "_active_area", "sheet"))
            except Exception:
                pass

            # Context menu for grid actions
            self._grid_menu = tk.Menu(self, tearoff=False, bg=DARK_BG, fg=DARK_FG)
            self._grid_menu.add_command(label="Cut Cell(s)", command=self.cut_selected_cells)
            self._grid_menu.add_command(label="Copy Cell(s)", command=self.copy_selected_cells)
            self._grid_menu.add_command(label="Paste Cell(s)", command=self.paste_selected_cells)
            self._grid_menu.add_separator()
            self._grid_menu.add_command(label="Clear Selected Cell(s)", command=self.clear_selected_cells)
            self._grid_menu.add_separator()
            self._grid_menu.add_command(label="Insert Row Before", command=self.insert_row_before)
            self._grid_menu.add_command(label="Remove Row", command=self.remove_selected_row)
            self._grid_menu.add_separator()
            self._grid_menu.add_command(label="Merge Cells", command=self.merge_selected_cells)
            self._grid_menu.add_command(label="Undo Merge Cells", command=self.undo_merge_cells)
            self._grid_menu.add_separator()
            self._grid_menu.add_command(label="Open in TDK Checker", command=self.open_tdk_checker_from_grid)

            def _popup_grid_menu(e):
                try:
                    self._active_area = "sheet"
                    # Reflect whatever is CURRENTLY selected (right-click
                    # itself never changes selection, matching Merge
                    # Cells' own convention) -- greys the action out for an
                    # invalid selection rather than only failing after the
                    # click.
                    try:
                        _bidx, _ridx, _vis_r, _err = self._resolve_tdk_grid_selection()
                        self._grid_menu.entryconfig(
                            "Open in TDK Checker", state=('disabled' if _err is not None else 'normal'))
                    except Exception:
                        pass
                    self._grid_menu.tk_popup(e.x_root, e.y_root)
                finally:
                    try:
                        self._grid_menu.grab_release()
                    except Exception:
                        pass


            try:
                # Right-click events across platforms
                self.sheet.bind("<Button-3>", _popup_grid_menu)
                self.sheet.bind("<Button-2>", _popup_grid_menu)  # some mac setups
            except Exception:
                pass

        # Relabel panel
        pnl = tk.Frame(right, bg=DARK_BG)
        pnl.pack(fill="x", pady=(6,0))

        tk.Label(pnl, text="Relabel (selected cell):", bg=DARK_BG, fg=DARK_FG).pack(anchor="w", padx=6, pady=(6,2))
        for lbl in ("TR","EN","MIXED","UID","NE"):
            ttk.Button(pnl, text=lbl, command=lambda L=lbl:self.paste_to_label(L),
                       style='Dark.TButton', width=16).pack(padx=6, pady=2, anchor="w")
        # Insert LANG3
        ttk.Button(pnl, text="LANG3", command=lambda: self.paste_to_label("LANG3"),
                   style='Dark.TButton', width=16).pack(padx=6, pady=2, anchor="w")
        ttk.Button(pnl, text="OTHER", command=lambda: self.paste_to_label("OTHER"),
                   style='Dark.TButton', width=16).pack(padx=6, pady=2, anchor="w")

        tk.Label(pnl, text="Shortcuts:", bg=DARK_BG, fg="#a0a0a0").pack(anchor="w", padx=6, pady=(8,2))
        tk.Label(pnl, text="Enter: edit  •  Esc: cancel edit  •  ↑/↓/←/→ move  •  ⌘ (Command): multi-select", bg=DARK_BG, fg="#808080").pack(anchor="w", padx=6)

    def _bind_keys(self):
        """Keyboard bindings.

        We intentionally avoid global Command/Ctrl shortcuts.
        Arrow key movement is handled by tksheet (and our arrow handler).
        """
        # Enter starts editing the currently selected cell in the grid
        self.bind_all("<Return>", self._on_enter_edit)
        self.bind_all("<KP_Enter>", self._on_enter_edit)

        # Esc cancels editing (if any)
        self.bind_all("<Escape>", self._on_escape_cancel)

        # Find
        self.bind_all("<Control-f>", self._open_search_dialog)
        self.bind_all("<Control-F>", self._open_search_dialog)
        self.bind_all("<Command-f>", self._open_search_dialog)
        self.bind_all("<Command-F>", self._open_search_dialog)


    def _on_enter_edit(self, event=None):
        if self.sheet is None:
            return "break"
        # start editing the currently selected cell
        self._ensure_sheet_focus()
        try:
            sel = self.sheet.get_currently_selected()
            r = getattr(sel, "row", None) if sel is not None else None
            c = getattr(sel, "column", None) if sel is not None else None
            if r is None or c is None:
                cells = self.sheet.get_selected_cells()
                if cells:
                    r, c = list(cells)[0]
        except Exception:
            r = c = None

        if r is None or c is None or r in self._sep_rows:
            # fall back to first real row, prefer Label column
            r, c = self._ensure_valid_selection(prefer_col=2)
        if r is None or c is None:
            return "break"

        # idx column is not editable
        if c == 0:
            c = 1
            try:
                self.sheet.select_cell(r, c)
                self.sheet.see(r, c)
            except Exception:
                pass

        try:
            self.sheet.edit_cell(r, c)
        except Exception:
            pass
        return "break"

    def _on_escape_cancel(self, event=None):
        self._cancel_edit_if_any()
        return "break"


    def _open_search_dialog(self, event=None):
        # decide target based on last active area
        self._search_target = "sheet" if getattr(self, "_active_area", "text") == "sheet" else "text"

        if self._search_win is None or not self._search_win.winfo_exists():
            self._create_search_window()
        else:
            try:
                self._search_win.deiconify()
                self._search_win.lift()
            except Exception:
                pass
        # keep keystrokes in the Find window (prevents focus jumping to the grid)
        try:
            self._search_win.grab_set()
        except Exception:
            pass

        # Update title and focus entry
        try:
            ttl = "Find in Grid" if self._search_target == "sheet" else "Find in Text"
            self._search_win.title(ttl)
        except Exception:
            pass

        # refresh matches for current query
        self._recompute_search_matches()
        try:
            self._search_entry.focus_set()
            self._search_entry.selection_range(0, "end")
        except Exception:
            pass
        return "break"

    def _create_search_window(self):
        win = tk.Toplevel(self)
        win.title("Find")
        win.configure(bg=DARK_BG)
        win.resizable(False, False)
        win.transient(self)
        # modal-like grab so typing stays in Find window
        try:
            win.grab_set()
        except Exception:
            pass

        frm = tk.Frame(win, bg=DARK_BG)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frm, text="Find:", bg=DARK_BG, fg=DARK_FG).grid(row=0, column=0, sticky="w")

        ent = tk.Entry(frm, textvariable=self._search_var, width=28)
        ent.grid(row=0, column=1, columnspan=3, sticky="we", padx=(6, 0))

        cnt = tk.Label(frm, text="0 matches", bg=DARK_BG, fg="#a0a0a0")
        cnt.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        btn_prev = ttk.Button(frm, text="Prev", style='Dark.TButton', width=8, command=self._search_prev)
        btn_next = ttk.Button(frm, text="Next", style='Dark.TButton', width=8, command=self._search_next)
        btn_close = ttk.Button(frm, text="Close", style='Dark.TButton', width=8, command=self._close_search_dialog)

        btn_prev.grid(row=2, column=0, pady=(8, 0), sticky="w")
        btn_next.grid(row=2, column=1, pady=(8, 0), sticky="w")
        btn_close.grid(row=2, column=3, pady=(8, 0), sticky="e")

        frm.columnconfigure(1, weight=1)

        # events
        ent.bind("<KeyRelease>", self._on_search_key)
        ent.bind("<Return>", lambda e: self._search_next())
        ent.bind("<Shift-Return>", lambda e: self._search_prev())

        win.protocol("WM_DELETE_WINDOW", self._close_search_dialog)

        self._search_win = win
        self._search_entry = ent
        self._search_count_lbl = cnt

        # tags for text highlight
        try:
            self.txt_input.tag_configure("search_current", background="#3a3a00", foreground="white")
        except Exception:
            pass

    def _close_search_dialog(self):
        # clear text highlight
        try:
            self.txt_input.tag_remove("search_current", "1.0", "end")
        except Exception:
            pass
        if self._search_win is not None and self._search_win.winfo_exists():
            try:
                self._search_win.grab_release()
            except Exception:
                pass
            try:
                self._search_win.withdraw()
            except Exception:
                pass

    def _on_search_key(self, event=None):
        # debounce frequent key events
        try:
            if getattr(self, "_search_after_id", None) is not None:
                self.after_cancel(self._search_after_id)
        except Exception:
            pass
        self._search_after_id = self.after(120, self._recompute_search_matches)

    def _recompute_search_matches(self):
        q = (self._search_var.get() or "").strip()
        self._search_matches = []
        self._search_index = -1

        # clear previous highlight
        try:
            self.txt_input.tag_remove("search_current", "1.0", "end")
        except Exception:
            pass

        if not q:
            self._update_search_count_label()
            return

        if self._search_target == "text":
            self._search_matches = self._find_in_text(q)
        else:
            self._search_matches = self._find_in_sheet(q)

        if self._search_matches:
            self._search_index = 0
            self._apply_current_match()

        self._update_search_count_label()

    def _update_search_count_label(self):
        n = len(self._search_matches)
        if n == 0:
            s = "0 matches"
        else:
            s = f"{self._search_index + 1}/{n}"
        try:
            self._search_count_lbl.configure(text=s)
        except Exception:
            pass

    def _search_next(self):
        if not self._search_matches:
            self._recompute_search_matches()
            return
        self._search_index = (self._search_index + 1) % len(self._search_matches)
        self._apply_current_match()
        self._update_search_count_label()

    def _search_prev(self):
        if not self._search_matches:
            self._recompute_search_matches()
            return
        self._search_index = (self._search_index - 1) % len(self._search_matches)
        self._apply_current_match()
        self._update_search_count_label()

    def _apply_current_match(self):
        if not self._search_matches or self._search_index < 0:
            return
        m = self._search_matches[self._search_index]
        if self._search_target == "text":
            start, end = m
            try:
                self.txt_input.tag_remove("search_current", "1.0", "end")
                self.txt_input.tag_add("search_current", start, end)
                self.txt_input.mark_set("insert", end)
                self.txt_input.see(start)
                # self._ensure_text_focus()
            except Exception:
                pass
        else:
            r, c = m
            try:
                # avoid tksheet crashes if an edit widget is active
                self._cancel_edit_if_any()
                # self._ensure_sheet_focus()
                self.sheet.select_cell(r, c)
                self.sheet.see(r, c)
            except Exception:
                pass
        # Keep typing in the Find entry (do not steal focus to text/grid)
        try:
            if getattr(self, "_search_win", None) is not None and self._search_win.winfo_exists():
                if str(self._search_win.state()) != "withdrawn":
                    self._search_entry.focus_set()
        except Exception:
            pass

    def _ensure_text_focus(self):
        try:
            self.txt_input.focus_set()
        except Exception:
            pass

    def _find_in_text(self, q: str):
        # case-insensitive search
        matches = []
        try:
            start = "1.0"
            ql = q.lower()
            while True:
                idx = self.txt_input.search(ql, start, stopindex="end", nocase=True)
                if not idx:
                    break
                end = f"{idx}+{len(q)}c"
                matches.append((idx, end))
                start = end
        except Exception:
            pass
        return matches


    def _find_in_sheet(self, q: str):
        """Search in the grid using the Python model (self.blocks) to avoid tksheet crashes."""
        if not getattr(self, 'blocks', None):
            return []
        ql = (q or "").lower().strip()
        if not ql:
            return []

        out = []
        # Iterate
        for vis_r, bidx, ridx, row in annotation_model.iter_visible_rows(
            self.blocks, self._row_index_map, self._sep_rows
        ):
            vals = [
                row.get('idx', ''),
                row.get('token', ''),
                row.get('label', ''),
                row.get('gloss', ''),
            ]
            for h in self._extra_headers:
                vals.append(row.get(h, ''))
            for c, v in enumerate(vals):
                s = "" if v is None else str(v)
                if ql in s.lower():
                    out.append((vis_r, c))
        return out



    def open_full_edit_window(self):
        """Open a larger, synchronized grid editor window."""
        if tksheet is None:
            messagebox.showwarning("Missing dependency", "Full edit window requires tksheet.")
            return

        # If already open, just raise it
        if self._full_win is not None and self._full_win.winfo_exists():
            try:
                self._full_win.deiconify()
                self._full_win.lift()
            except Exception:
                pass
            return

        win = tk.Toplevel(self)
        win.title("Full Edit Window")
        win.geometry("1200x800")
        win.configure(bg=DARK_BG)
        win.transient(self)

        frm = tk.Frame(win, bg=DARK_BG)
        frm.pack(fill="both", expand=True, padx=8, pady=8)

        sh = tksheet.Sheet(
            frm,
            headers=self._all_headers(),
            data=[[]],
            show_x_scrollbar=True,
            show_y_scrollbar=True,
            show_top_left=True,
            show_row_index=True,
            show_header=True,
            show_table=True,
        )
        sh.enable_bindings((
            "single_select",
            "drag_select",
            "ctrl_select",
            "shift_select",
            "cell_select",
            "edit_cell",
            "double_click_edit",
            "arrowkeys",
        ))
        try:
            sh.set_options(
                table_bg="#1b1b1b", table_fg=DARK_FG, table_grid_fg="#404040",
                index_bg="#171717", index_fg=DARK_FG, index_grid_fg="#404040",
                header_bg="#171717", header_fg=DARK_FG, header_grid_fg="#404040",
                top_left_bg="#171717", top_left_fg=DARK_FG,
                selected_rows_bg="#2b2b2b", selected_rows_fg="white",
                selected_columns_bg="#2b2b2b", selected_columns_fg="white",
                selected_cells_bg="#2b2b2b", selected_cells_fg="white",
            )
        except Exception:
            pass

        sh.pack(fill="both", expand=True)

        # use the same handlers as the main sheet
        sh.extra_bindings([
            ("end_edit_cell", lambda ev: self._on_sheet_end_edit(ev, sheet_obj=sh)),
            ("arrowkeys", self._on_sheet_arrow),
            ("cell_select", self._on_sheet_cell_select),
        ])

        # Track active sheet
        try:
            sh.bind("<FocusIn>", lambda e: setattr(self, "_active_sheet", sh))
            sh.bind("<Button-1>", lambda e: setattr(self, "_active_sheet", sh))
        except Exception:
            pass

        def _on_close():
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", _on_close)

        self._full_win = win
        self._full_sheet = sh
        self._active_sheet = sh

        # Populate from current model
        try:
            self._rebuild_grid_from_model()
        except Exception:
            pass

    # Frequency helpers
    def _freq_normalize_token(self, tok: str):
        return annotation_model.freq_normalize_token(tok)

    def _compute_word_frequencies(self, allowed_labels=None):
        return annotation_model.compute_word_frequencies(self.blocks, allowed_labels)
    # Word Frequency List
    def open_word_frequency(self):
        """Open Word Frequency List window (counts from annotated blocks)."""
        if getattr(self, "_freq_win", None) is not None and self._freq_win.winfo_exists():
            try:
                self._freq_win.deiconify()
                self._freq_win.lift()
            except Exception:
                pass
            return

        win = tk.Toplevel(self)
        win.title("Word Frequency List")
        win.geometry("900x600")
        win.configure(bg=DARK_BG)
        win.transient(self)

        outer = ttk.Frame(win, style='Dark.TFrame')
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # header
        ttk.Label(outer, text="•Choose which labels to include and click Refresh to update the list.", style='Dark.TLabel').pack(anchor="w")
        ttk.Label(outer, text="•Double-click a token (or press Enter) to send it to Concordance.", style='Dark.TLabel').pack(anchor="w", pady=(2, 0))


        # label filters
        flt = ttk.Frame(outer, style='Dark.TFrame')
        flt.pack(anchor="w", pady=(6, 6))

        self._freq_label_vars = {}
        for lab in ("TR", "EN", "MIXED", "UID", "NE", "OTHER", "LANG3"):
            v = tk.BooleanVar(value=True)
            self._freq_label_vars[lab] = v
            ttk.Checkbutton(flt, text=lab, variable=v, style='Dark.TCheckbutton').pack(side="left", padx=(0, 6))

        btns = ttk.Frame(outer, style='Dark.TFrame')
        btns.pack(anchor="w", pady=(0, 6))
        ttk.Button(btns, text="Refresh", style='Dark.TButton', command=lambda: None).pack(side="left")

        # table
        cols = ("token", "freq", "labels")
        tree = ttk.Treeview(outer, columns=cols, show="headings", style='Conc.Treeview')
        tree.heading("token", text="Token")
        tree.heading("freq", text="Frequency")
        tree.heading("labels", text="By label")

        tree.column("token", width=260, anchor="w")
        tree.column("freq", width=100, anchor="center")
        tree.column("labels", width=420, anchor="w")

        ysb = ttk.Scrollbar(outer, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)

        tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        # Bindings to send selected token to Concordance
        def _send_selected_to_concordance(event=None):
            sel = tree.selection()
            if not sel:
                return
            try:
                tok = tree.item(sel[0], 'values')[0]
            except Exception:
                return
            if not tok:
                return
            # open concordance and send token
            try:
                self.open_concordance()
                self._conc_query_var.set(tok)
                # defaults
                self._conc_ci_var.set(True)
                self._conc_regex_var.set(False)
                self._conc_run_search()
            except Exception:
                pass

        tree.bind('<Double-1>', _send_selected_to_concordance)
        tree.bind('<Return>', _send_selected_to_concordance)

        # Ensure Treeview gets keyboard focus when clicked
        tree.bind('<Button-1>', lambda e: tree.focus_set())

        # info label
        info = ttk.Label(outer, text="", style='Dark.TLabel')

        def _run_freq():
            for iid in tree.get_children():
                tree.delete(iid)
            allowed = {k for k, v in self._freq_label_vars.items() if v.get()}
            freq, by_label, total = self._compute_word_frequencies(allowed_labels=allowed)
            items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
            for tok, cnt in items:
                lbls = by_label.get(tok, {})
                lbl_str = ", ".join(f"{k}:{v}" for k, v in sorted(lbls.items()))
                tree.insert("", "end", values=(tok, cnt, lbl_str))
            info.configure(text=f"Total tokens counted: {total}")

        # wire Refresh button
        for child in btns.winfo_children():
            if isinstance(child, ttk.Button) and child.cget("text") == "Refresh":
                child.configure(command=_run_freq)

        _run_freq()

        info.pack(anchor="w", pady=(6, 0))

        def _export_csv():
            path = filedialog.asksaveasfilename(
                title="Export Word Frequency",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")]
            )
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Token", "Frequency", "LabelBreakdown"])
                    for iid in tree.get_children():
                        w.writerow(tree.item(iid, "values"))
            except Exception as e:
                messagebox.showerror("Export error", str(e))

        ttk.Button(outer, text="Export CSV...", style='Dark.TButton', command=_export_csv).pack(anchor="w", pady=(6, 0))

        def _on_close():
            try:
                win.destroy()
            except Exception:
                pass
            finally:
                self._freq_win = None

        win.protocol("WM_DELETE_WINDOW", _on_close)
        self._freq_win = win

    # Show Sentence
    def show_sentence_context(self):
        """Show the full sentence containing the currently selected grid token."""
        # Ensure grid exists and has a selection
        sheet = self._active_sheet if getattr(self, '_active_sheet', None) is not None else self.sheet
        if sheet is None:
            self.bell()
            return

        r = None
        c = None

        # 1) Try "currently selected" (may be None after menu focus)
        try:
            sel = sheet.get_currently_selected()
            # tksheet may return an object with .row/.column, or a (row, col) tuple
            if sel is not None:
                if hasattr(sel, "row") and hasattr(sel, "column"):
                    r = getattr(sel, "row", None)
                    c = getattr(sel, "column", None)
                elif isinstance(sel, (tuple, list)) and len(sel) >= 2:
                    r, c = sel[0], sel[1]
        except Exception:
            pass

        # 2) Fallback to any selected cells
        if r is None:
            try:
                cells = sheet.get_selected_cells()
                if cells:
                    rr, cc = list(cells)[0]
                    r, c = rr, cc
            except Exception:
                pass

        # 3) If still no row, try selected rows
        if r is None:
            try:
                rows = sheet.get_selected_rows(get_cells_as_rows=True)
                if rows:
                    r = list(rows)[0]
            except Exception:
                pass

        if r is None or r in self._sep_rows:
            messagebox.showwarning("Show Sentence", "Please select a valid token row.")
            return

        # Map visible row to model row
        bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
        if bidx is None:
            messagebox.showwarning("Show Sentence", "Invalid selection.")
            return

        token = str(self.blocks[bidx][ridx].get('token', '') or '').strip()
        if not token:
            messagebox.showwarning("Show Sentence", "Selected row has no token.")
            return

        # Prefer locating by token order
        idxv = self.blocks[bidx][ridx].get('idx', '')
        try:
            token_ord = int(str(idxv).strip()) if str(idxv).strip() else None
        except Exception:
            token_ord = None

        # Get full input text
        try:
            text = self.txt_input.get("1.0", "end-1c")
        except Exception:
            text = ""
        if not text:
            messagebox.showwarning("Show Sentence", "Input text is empty.")
            return

        # Locate the correct occurrence in the input text.

        def _compile_token_pat(tok: str):
            """Build a safe regex for matching a token in running text."""
            t = tok.strip()
            # use word boundaries
            try:
                if re.match(r"^[\w\u00C0-\u024F\u0300-\u036F']+$", t):
                    return re.compile(r"\b" + re.escape(t) + r"\b")
            except Exception:
                pass
            return re.compile(re.escape(t))

        def _nth_token_span_in_text(n: int):
            """Return (start,end) char span in `text` for the n-th token in model order (1-based)."""
            if n is None or n < 1:
                return None
            cur = 0
            k = 0
            # iterate tokens in model order
            for blk in getattr(self, 'blocks', []) or []:
                for rr in blk:
                    tv = str(rr.get('token', '') or '').strip()
                    if self._is_meta_row_token(tv):
                        continue
                    k += 1
                    patx = _compile_token_pat(tv)
                    m2 = patx.search(text, cur)
                    if m2 is None:
                        # if token cannot be found from current cursor, do a global fallback search
                        m2 = patx.search(text)
                    if m2 is None:
                        continue
                    cur = m2.end()
                    if k == n:
                        return (m2.start(), m2.end())
            return None

        span = None
        if token_ord is not None:
            try:
                span = _nth_token_span_in_text(token_ord)
            except Exception:
                span = None

        if span is None:
            # fallback: first occurrence of this token
            try:
                pat = _compile_token_pat(token)
                m = pat.search(text)
                if m is None:
                    raise ValueError("not found")
                span = (m.start(), m.end())
            except Exception:
                messagebox.showwarning("Show Sentence", f"Token '{token}' not found in input text.")
                return

        start_i, end_i = span

        # Determine sentence boundaries (. ? ! or line breaks)
        left = start_i
        while left > 0 and text[left-1] not in '.?!\n':
            left -= 1
        right = end_i
        L = len(text)
        while right < L and text[right] not in '.?!\n':
            right += 1
        if right < L:
            right += 1

        sentence = text[left:right].strip()
        if not sentence:
            messagebox.showwarning("Show Sentence", "Could not extract sentence.")
            return

        # Show sentence in a reusable read-only window (dark themed)
        if self._sentence_win is not None and self._sentence_win.winfo_exists():
            win, txt = self._sentence_win
            try:
                win.deiconify()
                win.lift()
            except Exception:
                pass
            txt.configure(state="normal")
            txt.delete("1.0", "end")
        else:
            win = tk.Toplevel(self)
            win.title("Sentence Context")
            win.configure(bg=DARK_BG)
            win.geometry("800x200")
            win.transient(self)

            frm = tk.Frame(win, bg=DARK_BG)
            frm.pack(fill="both", expand=True, padx=10, pady=10)

            txt = ScrolledText(frm, wrap="word", bg="#1b1b1b", fg=DARK_FG, insertbackground="white")
            txt.pack(fill="both", expand=True)
            txt.configure(font=("Menlo", 13))

            def _on_close():
                try:
                    win.destroy()
                except Exception:
                    pass
                finally:
                    self._sentence_win = None

            win.protocol("WM_DELETE_WINDOW", _on_close)
            self._sentence_win = (win, txt)

        txt.insert("1.0", sentence)
        txt.configure(state="disabled")

        # try to highlight token
        try:
            txt.configure(state="normal")
            tpat = re.compile(re.escape(token))
            for mm in tpat.finditer(sentence):
                s = f"1.0+{mm.start()}c"
                e = f"1.0+{mm.end()}c"
                txt.tag_add("HIT", s, e)
            txt.tag_configure("HIT", background="#3a3a3a")
            txt.configure(state="disabled")
        except Exception:
            pass

    # Concordance
    def open_concordance(self):
        """Open KWIC concordance window (searches inside the input text panel)."""
        if getattr(self, "_conc_win", None) is not None and self._conc_win.winfo_exists():
            try:
                self._conc_win.deiconify()
                self._conc_win.lift()
            except Exception:
                pass
            return

        win = tk.Toplevel(self)
        win.title("Concordance (KWIC)")
        win.geometry("1000x600")
        win.configure(bg=DARK_BG)
        win.transient(self)

        top = ttk.Frame(win, style='Dark.TFrame')
        top.pack(side="top", fill="x", padx=10, pady=10)

        ttk.Label(top, text="Query:", style='Dark.TLabel').grid(row=0, column=0, sticky="w")
        ent = ttk.Entry(top, textvariable=self._conc_query_var, width=40, style='Dark.TEntry')
        ent.grid(row=0, column=1, sticky="we", padx=(6, 10))

        ttk.Label(top, text="Context (chars):", style='Dark.TLabel').grid(row=0, column=2, sticky="w")
        spn = ttk.Spinbox(top, from_=5, to=200, textvariable=self._conc_ctx_var, width=6, style='Dark.TSpinbox')
        spn.grid(row=0, column=3, sticky="w", padx=(6, 10))

        chk_ci = ttk.Checkbutton(top, text="Case-insensitive", variable=self._conc_ci_var, style='Dark.TCheckbutton')
        chk_ci.grid(row=0, column=4, sticky="w", padx=(0, 10))

        chk_rx = ttk.Checkbutton(top, text="Regex", variable=self._conc_regex_var, style='Dark.TCheckbutton')
        chk_rx.grid(row=0, column=5, sticky="w", padx=(0, 10))

        ttk.Button(top, text="Search", command=self._conc_run_search).grid(row=0, column=6, sticky="e")

        top.columnconfigure(1, weight=1)

        mid = ttk.Frame(win, style='Dark.TFrame')
        mid.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("left", "kwic", "right")
        tree = ttk.Treeview(mid, columns=cols, show="headings", selectmode="browse", style='Conc.Treeview')
        tree.heading("left", text="Left")
        tree.heading("kwic", text="KWIC")
        tree.heading("right", text="Right")
        tree.column("left", width=420, anchor="e")
        tree.column("kwic", width=140, anchor="center")
        tree.column("right", width=420, anchor="w")

        ysb = ttk.Scrollbar(mid, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")

        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        bottom = ttk.Frame(win, style='Dark.TFrame')
        bottom.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        ttk.Label(bottom, textvariable=self._conc_count_var, style='Dark.TLabel').pack(side="left")
        ttk.Button(bottom, text="Prev", command=lambda: self._conc_nav(-1)).pack(side="right", padx=(6, 0))
        ttk.Button(bottom, text="Next", command=lambda: self._conc_nav(1)).pack(side="right")
        ttk.Button(bottom, text="Clear", command=self._conc_clear).pack(side="right", padx=(0, 6))

        def _on_close():
            try:
                win.destroy()
            except Exception:
                pass
            finally:
                self._conc_win = None
                self._conc_tree = None
                self._conc_hit_spans = {}

        win.protocol("WM_DELETE_WINDOW", _on_close)

        self._conc_win = win
        self._conc_tree = tree

        try:
            ent.focus_set()
            ent.bind("<Return>", lambda e: self._conc_run_search())
            tree.bind("<Double-1>", self._conc_jump_to_hit)
            tree.bind("<Return>", self._conc_jump_to_hit)
            tree.bind("<<TreeviewSelect>>", self._conc_jump_to_hit)
            tree.bind("<Up>", lambda e: (self._conc_nav(-1), "break"))
            tree.bind("<Down>", lambda e: (self._conc_nav(1), "break"))
        except Exception:
            pass

    def _conc_clear(self):
        if self._conc_tree is None:
            return
        try:
            for iid in self._conc_tree.get_children():
                self._conc_tree.delete(iid)
        except Exception:
            pass
        self._conc_count_var.set("0 matches")
        if not hasattr(self, "_conc_hit_spans"):
            self._conc_hit_spans = {}
        self._conc_hit_spans = {}
        try:
            self.txt_input.tag_remove("KWIC_HIT", "1.0", "end")
        except Exception:
            pass

    def _conc_run_search(self):
        if self._conc_tree is None:
            return

        query = (self._conc_query_var.get() or "").strip()
        if not query:
            messagebox.showwarning("Concordance", "Please enter a query.")
            return

        try:
            ctx = int(self._conc_ctx_var.get())
        except Exception:
            ctx = 30
        ctx = max(1, min(ctx, 500))

        try:
            text = self.txt_input.get("1.0", "end-1c")
        except Exception:
            text = ""

        if not text:
            self._conc_clear()
            return

        flags = re.IGNORECASE if self._conc_ci_var.get() else 0
        try:
            pat = re.compile(query if self._conc_regex_var.get() else re.escape(query), flags)
        except Exception as e:
            messagebox.showerror("Regex error", str(e))
            return

        self._conc_clear()
        self._conc_hit_spans = {}

        def _norm(s):
            return " ".join(s.replace("\n", " ").replace("\t", " ").split())

        max_hits = 5000
        n = 0
        for m in pat.finditer(text):
            start, end = m.span()
            left = _norm(text[max(0, start-ctx):start])
            kwic = _norm(m.group(0))
            right = _norm(text[end:min(len(text), end+ctx)])

            iid = f"h{start}_{end}_{n}"
            try:
                self._conc_tree.insert("", "end", iid=iid, values=(left, kwic, right))
                self._conc_hit_spans[iid] = (start, end)
            except Exception:
                pass
            n += 1
            if n >= max_hits:
                break

        self._conc_count_var.set(f"{n} matches" + (" (truncated)" if n >= max_hits else ""))

        try:
            kids = list(self._conc_tree.get_children())
            if kids:
                self._conc_tree.selection_set(kids[0])
                self._conc_tree.focus(kids[0])
                self._conc_tree.see(kids[0])
                self._conc_jump_to_hit()
                self._conc_tree.focus_set()
        except Exception:
            pass

    def _conc_nav(self, delta):
        if self._conc_tree is None:
            return
        kids = list(self._conc_tree.get_children())
        if not kids:
            return
        sel = self._conc_tree.selection()
        cur = kids.index(sel[0]) if sel and sel[0] in kids else 0
        nxt = max(0, min(len(kids)-1, cur + delta))
        try:
            self._conc_tree.selection_set(kids[nxt])
            self._conc_tree.focus(kids[nxt])
            self._conc_tree.see(kids[nxt])
        except Exception:
            pass
        self._conc_jump_to_hit()

    def _conc_jump_to_hit(self, event=None):
        if self._conc_tree is None:
            return
        sel = self._conc_tree.selection()
        if not sel:
            return
        span = self._conc_hit_spans.get(sel[0])
        if not span:
            return
        start, end = span
        try:
            s_idx = f"1.0+{start}c"
            e_idx = f"1.0+{end}c"
            self.txt_input.tag_remove("KWIC_HIT", "1.0", "end")
            self.txt_input.tag_add("KWIC_HIT", s_idx, e_idx)
            self.txt_input.see(s_idx)
            self.txt_input.mark_set("insert", s_idx)
            self.txt_input.focus_set()
        except Exception:
            pass
    # Shortcut handlers
    def _shortcut_open(self, event=None):
        self.open_input()
        return "break"

    def _shortcut_run(self, event=None):
        self.run_pipeline()
        return "break"

    def _shortcut_save(self, event=None):
        self.save_output()
        return "break"

    def _shortcut_relabel(self, event, val):
        self.paste_to_label(val)
        return "break"

    def _on_shortcut_label(self, value):
        self.paste_to_label(value)
        return "break"

    def paste_to_label(self, new_value):
        if self.sheet is None:
            return
        if getattr(self, "_relabel_busy", False):
            return
        self._relabel_busy = True
        self._ensure_sheet_focus()
        self._cancel_edit_if_any()
        r, _ = self._ensure_valid_selection(prefer_col=2)
        if r is None:
            self.bell()
            return
        c = 2  # force Label column
        bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
        if bidx is None:
            self.bell()
            return
        tok = self.blocks[bidx][ridx].get('token')
        if annotation_model.is_matrixembed_locked(tok, new_value):
            self.bell()
            return
        self.blocks[bidx][ridx]['label'] = new_value
        self._mark_dirty()
        try:
            self.sheet.set_cell_data(r, c, new_value)
            # keep selection on label cell
            self.sheet.select_cell(r, c)
            self.sheet.see(r, c)
            if hasattr(self.sheet, "refresh"):
                self.sheet.refresh()
            self._ensure_sheet_focus()
            self.update_idletasks()
        except Exception:
            pass
        # clear busy flag after Tk has processed UI updates
        self.after_idle(lambda: setattr(self, "_relabel_busy", False))
        self._last_pos = (r, c)

    def _is_macos(self):
        try:
            return os.uname().sysname == "Darwin"
        except Exception:
            return False

    def _toggle(self, key, val):
        new_val = bool(val)
        if self.cfg.get(key) != new_val:
            self._mark_dirty()
        self.cfg[key] = new_val

    # Dynamic columns
    def _all_headers(self):
        return list(self._core_headers) + list(self._extra_headers)

    def _add_new_column(self, col_name: str):
        name = (col_name or "").strip()
        if not name:
            return False, "Column name cannot be empty."
        if "\t" in name or "\n" in name or "\r" in name:
            return False, "Column name cannot contain tabs or newlines."
        if name in set(self._all_headers()):
            return False, "Column already exists."

        self._extra_headers.append(name)
        self._mark_dirty()

        # ensure model rows have the new key
        for blk in getattr(self, 'blocks', []) or []:
            for r in blk:
                r[name] = r.get(name, "")

        # update UI
        if self.sheet is not None:
            try:
                self.sheet.headers(self._all_headers())
            except Exception:
                pass

            try:
                data = self.sheet.get_sheet_data(return_copy=True)
            except TypeError:
                data = self.sheet.get_sheet_data()

            new_data = []
            for rr in (data or []):
                row = list(rr) if rr is not None else []
                row.append("")
                new_data.append(row)

            try:
                self.sheet.set_sheet_data(new_data)
                if hasattr(self.sheet, "refresh"):
                    self.sheet.refresh()
            except Exception:
                pass

        return True, ""

    def _add_new_column_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add New Column")
        win.configure(bg=DARK_BG)
        win.resizable(False, False)
        win.transient(self)

        frm = tk.Frame(win, bg=DARK_BG)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frm, text="Name column:", bg=DARK_BG, fg=DARK_FG).grid(row=0, column=0, sticky="w")
        v = tk.StringVar(value="")
        ent = tk.Entry(frm, textvariable=v, width=26)
        ent.grid(row=0, column=1, sticky="we", padx=(8, 0))

        msg = tk.Label(frm, text="", bg=DARK_BG, fg="#ffcc66")
        msg.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def do_add():
            ok, err = self._add_new_column(v.get())
            if ok:
                try:
                    win.destroy()
                except Exception:
                    pass
            else:
                msg.configure(text=err)

        btn_add = ttk.Button(frm, text="Add", style='Dark.TButton', command=do_add, width=10)
        btn_cancel = ttk.Button(frm, text="Cancel", style='Dark.TButton', command=lambda: win.destroy(), width=10)
        btn_add.grid(row=2, column=0, pady=(10, 0), sticky="w")
        btn_cancel.grid(row=2, column=1, pady=(10, 0), sticky="e")

        frm.columnconfigure(1, weight=1)
        ent.focus_set()
        ent.bind("<Return>", lambda e: do_add())
        ent.bind("<Escape>", lambda e: win.destroy())

    def open_input(self):
        path = filedialog.askopenfilename(title="Open text file",
                                          filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path: return
        with open(path, "r", encoding="utf-8") as f:
            self.current_text = f.read()
        self.current_path = path
        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", self.current_text)
        self._mark_dirty()

    def run_pipeline(self):
        txt = self.txt_input.get("1.0", "end-1c")
        if not txt.strip():
            messagebox.showwarning("Empty", "No input text.")
            return
        try:
            out = self._run_annotation_pipeline(txt)
            self._populate_table(out)
            self._attach_confidence(self.blocks)
            self.current_output = out
            # _populate_table replaces self.blocks with a new list object;
            # sync it (and the current input text) into the active dataset
            # immediately rather than waiting for a later tab switch/save/
            # export to notice. A failed pipeline raises before this point,
            # so self.blocks (and the active dataset) are left untouched --
            # never partially cleared.
            self._sync_active_dataset_from_live()
            self._mark_dirty()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _get_sheet_rows_for_export(self):
        """Return the current grid rows exactly as shown in the UI.
        Includes blank separator rows (used as block boundaries).

        Each row is returned as strings with the current number of columns.
        """
        if self.sheet is None:
            return None
        try:
            data = self.sheet.get_sheet_data(return_copy=True)
        except TypeError:
            # older tksheet versions
            data = self.sheet.get_sheet_data()
        rows = []
        for r in (data or []):
            if r is None:
                rows.append(["" for _ in self._all_headers()])
                continue
            rr = list(r)
            ncol = len(self._all_headers())
            while len(rr) < ncol:
                rr.append("")
            rr = rr[:ncol]
            rows.append(["" if v is None else str(v) for v in rr])
        return rows

    def _get_sheet_headers_for_export(self):
        """Return the current grid headers as shown in the UI."""
        if self.sheet is None:
            return self._all_headers()
        try:
            hdrs = self.sheet.headers()
            if hdrs and isinstance(hdrs, (list, tuple)):
                h = [str(x) for x in hdrs]
                ncol = len(self._all_headers())
                while len(h) < ncol:
                    h.append("")
                return h[:ncol]
        except Exception:
            pass
        return self._all_headers()

    def _sheet_rows_to_txt(self, rows):
        return annotation_model.sheet_rows_to_txt(rows, self._all_headers())

    # Export Table
    #
    # save_output is the entry point the menu/toolbar/⌘S shortcut all still
    # call; it now always opens the dataset+format chooser instead of going
    # straight to a file dialog, per the "existing Export Table command
    # should use this chooser" requirement -- even for a single-dataset
    # project, so the workflow stays consistent as datasets are added.
    _EXPORT_FORMATS = ("TXT", "CSV", "CoNLL", "JSONL")
    _EXPORT_EXTENSIONS = {"TXT": ".txt", "CSV": ".csv", "CoNLL": ".conll", "JSONL": ".jsonl"}

    def save_output(self):
        self.open_export_dialog()

    def open_export_dialog(self):
        self._sync_active_dataset_from_live()
        if not any(ds.get('blocks') for ds in self.datasets):
            messagebox.showwarning("Empty", "No output to export.")
            return

        names = [ds.get('name', '') for ds in self.datasets]
        active_name = names[self._active_dataset_index]

        win = tk.Toplevel(self)
        win.title("Export Table")
        win.configure(bg=DARK_BG)
        win.transient(self)

        frm = ttk.Frame(win, style='Dark.TFrame')
        frm.pack(fill='both', expand=True, padx=14, pady=14)

        row1 = ttk.Frame(frm, style='Dark.TFrame')
        row1.pack(fill='x', pady=(0, 8))
        ttk.Label(row1, text="Dataset:", style='Dark.TLabel', width=10).pack(side='left')
        dataset_var = tk.StringVar(value=active_name)
        dataset_combo = ttk.Combobox(row1, textvariable=dataset_var, values=names,
                                      state='readonly', width=28)
        dataset_combo.pack(side='left')
        dataset_combo.current(self._active_dataset_index)

        row2 = ttk.Frame(frm, style='Dark.TFrame')
        row2.pack(fill='x', pady=(0, 14))
        ttk.Label(row2, text="Format:", style='Dark.TLabel', width=10).pack(side='left')
        format_var = tk.StringVar(value="TXT")
        ttk.Combobox(row2, textvariable=format_var, values=list(self._EXPORT_FORMATS),
                     state='readonly', width=28).pack(side='left')

        def _do_export():
            # Resolve by combobox *position*, not by the displayed name text
            # -- dataset names are not required to be unique (spec: "do not
            # need to be globally unique"), so matching by name text could
            # silently export the wrong same-named dataset.
            idx = dataset_combo.current()
            if idx < 0 or idx >= len(self.datasets):
                messagebox.showerror("Export Table", "Choose a dataset.")
                return
            ds = self.datasets[idx]
            fmt = format_var.get()
            if fmt not in self._EXPORT_FORMATS:
                messagebox.showerror("Export Table", "Choose a format.")
                return

            ext = self._EXPORT_EXTENSIONS[fmt]
            default_name = annotation_model.sanitize_dataset_filename(ds.get('name', '')) + ext
            path = filedialog.asksaveasfilename(
                title="Export Table",
                defaultextension=ext,
                initialfile=default_name,
                filetypes=[(fmt, f"*{ext}"), ("All", "*.*")],
            )
            if not path:
                return  # cancelling must create no file

            try:
                self._write_dataset_export(ds, fmt, path)
            except Exception as e:
                messagebox.showerror("Export error", str(e))
                return

            win.destroy()
            messagebox.showinfo("Saved", f"Saved to:\n{path}")

        btns = ttk.Frame(frm, style='Dark.TFrame')
        btns.pack(fill='x')
        ttk.Button(btns, text="Export", style='Dark.TButton', command=_do_export).pack(side='left')
        ttk.Button(btns, text="Cancel", style='Dark.TButton', command=win.destroy).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        win.grab_set()

    def _write_dataset_export(self, ds, fmt, path):
        """Write one dataset to `path` in the given format. Never mutates
        `ds`/its blocks and never calls the annotation pipeline -- every
        helper here works from a private deep copy when it needs to
        renumber tokens for display."""
        blocks = ds.get('blocks', [])
        extra_headers = ds.get('extra_headers', [])
        is_active = (ds is self.datasets[self._active_dataset_index])

        if fmt == "TXT":
            sheet_rows = self._get_sheet_rows_for_export() if is_active else None
            if sheet_rows is not None:
                text = self._sheet_rows_to_txt(sheet_rows)
            else:
                text = annotation_model.reconstruct_text_from_blocks(copy.deepcopy(blocks), extra_headers)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return

        if fmt == "CSV":
            sheet_rows = self._get_sheet_rows_for_export() if is_active else None
            if sheet_rows is not None:
                headers = self._get_sheet_headers_for_export()
                rows = sheet_rows
            else:
                headers = list(self._core_headers) + list(extra_headers)
                rows = annotation_model.blocks_to_export_rows(blocks, extra_headers)
            with open(path, "w", encoding="utf-8", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(headers)
                for rr in rows:
                    w.writerow(rr)
            return

        if fmt == "CoNLL":
            text = annotation_model.blocks_to_conll(blocks)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return

        if fmt == "JSONL":
            text = annotation_model.blocks_to_jsonl(blocks, ds.get('name', ''))
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return

        raise ValueError(f"Unknown export format: {fmt}")

    # Grid context actions
    def copy_selected_cells(self):
        if self.sheet is None:
            return

        try:
            cells = self.sheet.get_selected_cells()
        except Exception:
            cells = None

        if not cells:
            self.bell()
            return

        # Determine rectangular bounds
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rmin, rmax = min(rows), max(rows)
        cmin, cmax = min(cols), max(cols)

        data = []
        for r in range(rmin, rmax + 1):
            row_vals = []
            for c in range(cmin, cmax + 1):
                try:
                    v = self.sheet.get_cell_data(r, c)
                except Exception:
                    v = ""
                row_vals.append("" if v is None else str(v))
            data.append(row_vals)

        self._grid_clipboard = {
            "rows": rmax - rmin + 1,
            "cols": cmax - cmin + 1,
            "data": data,
        }

    def cut_selected_cells(self):
        # Cut = Copy + Clear
        self.copy_selected_cells()
        # If copy succeeded (clipboard filled), clear the selected cells
        if self._grid_clipboard:
            self.clear_selected_cells()

    def paste_selected_cells(self):
        if self.sheet is None or not self._grid_clipboard:
            self.bell()
            return

        try:
            cells = self.sheet.get_selected_cells()
        except Exception:
            cells = None
        if not cells:
            self.bell()
            return

        self._ensure_sheet_focus()
        self._cancel_edit_if_any()

        # start position = top-left of current selection
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)

        data = self._grid_clipboard.get('data', [])
        if not data:
            return

        hdrs = self._all_headers()

        for dr, row_vals in enumerate(data):
            r = r0 + dr
            if r in self._sep_rows:
                continue
            bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
            if bidx is None:
                continue

            for dc, val in enumerate(row_vals):
                c = c0 + dc
                if c == 0 or c >= len(hdrs):
                    continue

                # Matrix/Embed constraint only for Label column
                if hdrs[c] == "Label":
                    tok = self.blocks[bidx][ridx].get('token')
                    if annotation_model.is_matrixembed_locked(tok, val):
                        continue

                # update model
                if hdrs[c] == "Item":
                    self.blocks[bidx][ridx]['token'] = val
                elif hdrs[c] == "Label":
                    self.blocks[bidx][ridx]['label'] = val
                elif hdrs[c] == "Gloss":
                    self.blocks[bidx][ridx]['gloss'] = val
                else:
                    self.blocks[bidx][ridx][hdrs[c]] = val

                # update UI
                try:
                    self.sheet.set_cell_data(r, c, val)
                except Exception:
                    pass

        # renumber tokens if Item column affected
        self._renumber_tokens()
        self._refresh_sheet_idx_column()
        self._mark_dirty()

        try:
            if hasattr(self.sheet, "refresh"):
                self.sheet.refresh()
        except Exception:
            pass

    def clear_selected_cells(self):
        if self.sheet is None:
            return
        try:
            cells = self.sheet.get_selected_cells()
        except Exception:
            cells = None
        if not cells:
            self.bell()
            return

        self._ensure_sheet_focus()
        self._cancel_edit_if_any()

        hdrs = self._all_headers()
        for r, c in cells:
            if r in self._sep_rows:
                continue
            # idx column is not clearable
            if c == 0:
                continue
            bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
            if bidx is None:
                continue

            # update model
            if c == 1:
                self.blocks[bidx][ridx]['token'] = ""
                self._renumber_tokens()
            elif c == 2:
                self.blocks[bidx][ridx]['label'] = ""
            elif c == 3:
                self.blocks[bidx][ridx]['gloss'] = ""
            else:
                if 0 <= c < len(hdrs):
                    key = hdrs[c]
                    if key not in ("Token", "Item", "Label", "Gloss"):
                        self.blocks[bidx][ridx][key] = ""

            # update UI cell
            try:
                self.sheet.set_cell_data(r, c, "")
            except Exception:
                pass

        # refresh idx column if tokens changed
        self._refresh_sheet_idx_column()
        self._mark_dirty()
        try:
            if hasattr(self.sheet, "refresh"):
                self.sheet.refresh()
        except Exception:
            pass

    def insert_row_before(self):
        if self.sheet is None:
            return
        try:
            sel = self.sheet.get_currently_selected()
            r = getattr(sel, 'row', None)
        except Exception:
            r = None
        if r is None or r in self._sep_rows:
            self.bell()
            return

        bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
        if bidx is None:
            self.bell()
            return

        # create empty row respecting dynamic columns
        new_row = {"idx": "", "token": "", "label": "", "gloss": ""}
        for h in self._extra_headers:
            new_row[h] = ""

        self.blocks[bidx].insert(ridx, new_row)

        # renumber tokens
        self._renumber_tokens()
        self._mark_dirty()

        # rebuild grid directly from the model so blank row is preserved
        self._rebuild_grid_from_model(select_row=r, select_col=2)

    def remove_selected_row(self):
        if self.sheet is None:
            return
        try:
            sel = self.sheet.get_currently_selected()
            r = getattr(sel, 'row', None)
        except Exception:
            r = None
        if r is None or r in self._sep_rows:
            self.bell()
            return

        bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
        if bidx is None:
            self.bell()
            return

        try:
            del self.blocks[bidx][ridx]
        except Exception:
            self.bell()
            return

        # remove empty blocks if needed
        if not self.blocks[bidx]:
            del self.blocks[bidx]

        self._renumber_tokens()
        self._mark_dirty()
        self._rebuild_grid_from_model(select_row=None, select_col=2)

    # --- Merge Cells (main annotation table) --------------------------------
    #
    # Reuses the sheet's ALREADY-enabled multi-cell selection (drag_select /
    # shift_select / ctrl_select, see _build_body's enable_bindings call --
    # nothing about the sheet's selection configuration was changed) and the
    # same annotation_model.merge_token_rows / rows_are_adjacent_same_block
    # used for the (now-removed) Confidence Review Tool merge feature. Operates on
    # self.blocks directly; never runs Stanza, fastText, or the reranker.

    def merge_selected_cells(self):
        if self.sheet is None:
            return
        try:
            cells = self.sheet.get_selected_cells()
        except Exception:
            cells = None
        if not cells:
            messagebox.showinfo("Merge Cells", "Select two or more adjacent token rows to merge.")
            return

        vis_rows = sorted(set(r for r, _c in cells))
        if len(vis_rows) < 2:
            messagebox.showinfo("Merge Cells", "Select two or more adjacent token rows to merge.")
            return

        resolved = []
        bidx_set = set()
        for r in vis_rows:
            if r in self._sep_rows:
                messagebox.showerror("Merge Cells", "The selection includes a blank separator row; cannot merge.")
                return
            bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
            if bidx is None:
                messagebox.showerror("Merge Cells", "The selection includes a row that could not be resolved; cannot merge.")
                return
            resolved.append((r, bidx, ridx))
            bidx_set.add(bidx)

        if len(bidx_set) != 1:
            messagebox.showerror("Merge Cells", "All selected rows must belong to the same sentence.")
            return

        bidx = next(iter(bidx_set))
        ridx_list = [ridx for (_r, _b, ridx) in resolved]
        if not annotation_model.rows_are_adjacent_same_block(self.blocks, bidx, ridx_list):
            messagebox.showerror(
                "Merge Cells",
                "Selected rows must be adjacent, within the same sentence, and must not include a "
                "MatrixLang/EmbedLang/SentenceID row.")
            return

        sorted_pairs = sorted(zip(ridx_list, (r for r, _b, _ri in resolved)), key=lambda p: p[0])
        sorted_ridx = [p[0] for p in sorted_pairs]
        first_vis_r = sorted_pairs[0][1]
        tokens = [self.blocks[bidx][ri]['token'] for ri in sorted_ridx]
        # Established token-merge convention: direct concatenation (the same
        # convention split relies on in reverse -- see annotation_model).
        joined = "".join(tokens)

        self._open_merge_cells_confirm_dialog(bidx, sorted_ridx, first_vis_r, tokens, joined)

    def _open_merge_cells_confirm_dialog(self, bidx, ridx_list, first_vis_r, tokens, joined_default):
        win = tk.Toplevel(self)
        win.title('Merge Cells')
        win.configure(bg=DARK_BG)
        win.transient(self)

        frm = ttk.Frame(win, style='Dark.TFrame')
        frm.pack(fill='both', expand=True, padx=14, pady=14)

        ttk.Label(frm, text=" + ".join(tokens) + "  →  " + joined_default,
                  style='Dark.TLabel').pack(anchor='w')

        tok_var = tk.StringVar(value=joined_default)
        lab_var = tk.StringVar(value='')
        glo_var = tk.StringVar(value='')

        row1 = ttk.Frame(frm, style='Dark.TFrame')
        row1.pack(fill='x', pady=(10, 0))
        ttk.Label(row1, text="Merged token:", style='Dark.TLabel').pack(side='left')
        ttk.Entry(row1, textvariable=tok_var, style='Dark.TEntry', width=28).pack(side='left', padx=(6, 0))

        row2 = ttk.Frame(frm, style='Dark.TFrame')
        row2.pack(fill='x', pady=(8, 0))
        ttk.Label(row2, text="Label:", style='Dark.TLabel').pack(side='left')
        ttk.Combobox(row2, textvariable=lab_var, values=list(self.ALL_LABELS), width=8, state='readonly').pack(side='left', padx=(6, 0))

        row3 = ttk.Frame(frm, style='Dark.TFrame')
        row3.pack(fill='x', pady=(8, 0))
        ttk.Label(row3, text="Gloss:", style='Dark.TLabel').pack(side='left')
        ttk.Entry(row3, textvariable=glo_var, style='Dark.TEntry', width=28).pack(side='left', padx=(6, 0))

        def _confirm():
            if not tok_var.get().strip():
                messagebox.showwarning("Merge Cells", "Merged token must not be blank.")
                return
            if not lab_var.get():
                messagebox.showwarning("Merge Cells", "Choose a label for the merged token.")
                return
            old_rows = copy.deepcopy(self.blocks[bidx])
            try:
                annotation_model.merge_token_rows(self.blocks, bidx, ridx_list,
                                                   tok_var.get(), lab_var.get(), glo_var.get())
            except ValueError as e:
                messagebox.showerror("Merge Cells", str(e))
                return
            self._renumber_tokens()
            self._update_block_matrix_embed(bidx)
            self._merge_cells_undo_stack.append({'bidx': bidx, 'old_rows': old_rows})
            if len(self._merge_cells_undo_stack) > 50:
                self._merge_cells_undo_stack.pop(0)
            self._mark_dirty()
            self._rebuild_grid_from_model(select_row=first_vis_r, select_col=2)
            self._uid_on_structural_change()
            win.destroy()

        btns = ttk.Frame(frm, style='Dark.TFrame')
        btns.pack(fill='x', pady=(14, 0))
        ttk.Button(btns, text="Confirm", style='Dark.TButton', command=_confirm).pack(side='left')
        ttk.Button(btns, text="Cancel", style='Dark.TButton', command=win.destroy).pack(side='right')
        win.bind('<Escape>', lambda e: win.destroy())
        win.grab_set()

    def undo_merge_cells(self):
        if not self._merge_cells_undo_stack:
            self.bell()
            messagebox.showinfo("Undo Merge Cells", "No merge to undo.")
            return
        entry = self._merge_cells_undo_stack.pop()
        bidx = entry['bidx']
        try:
            self.blocks[bidx] = entry['old_rows']
        except Exception:
            self.bell()
            return
        self._renumber_tokens()
        self._update_block_matrix_embed(bidx)
        self._mark_dirty()
        self._rebuild_grid_from_model()
        self._uid_on_structural_change()

    # Focus & Edit utils

    def _set_cell_on_all_sheets(self, r: int, c: int, value: str):
        """Set a cell value in the main sheet and the full edit sheet (if open)."""
        for sh in (getattr(self, 'sheet', None), getattr(self, '_full_sheet', None)):
            if sh is None:
                continue
            try:
                sh.set_cell_data(r, c, value)
            except Exception:
                pass
            try:
                if hasattr(sh, 'refresh'):
                    sh.refresh()
            except Exception:
                pass
    def _ensure_sheet_focus(self):
        if self.sheet is None:
            return
        try:
            self.sheet.focus_set()
        except Exception:
            pass
        try:
            if hasattr(self.sheet, "activate_bindings"):
                self.sheet.activate_bindings(True)
        except Exception:
            pass

    def _cancel_edit_if_any(self):
        if self.sheet is None:
            return
        for meth in ("end_edit_cell", "quit_editing", "edit_cell_cancel",
                     "end_editing", "cancel_editing"):
            fn = getattr(self.sheet, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def _ensure_valid_selection(self, prefer_col=2):
        """Geçerli, ayıraç olmayan bir seçim olsun. Yoksa ilk gerçek satırın prefer_col'unu seç."""
        if self.sheet is None:
            return None, None
        try:
            sel = self.sheet.get_currently_selected()
            r = getattr(sel, "row", None) if sel is not None else None
            c = getattr(sel, "column", None) if sel is not None else None
            if r is None or c is None:
                cells = self.sheet.get_selected_cells()
                if cells:
                    r, c = list(cells)[0]
        except Exception:
            r = c = None

        if r is None or c is None or r in self._sep_rows:
            fr = self._first_real_row()
            if fr is None:
                return None, None
            try:
                self.sheet.select_cell(fr, prefer_col)
                self.sheet.see(fr, prefer_col)
                self._last_pos = (fr, prefer_col)
            except Exception:
                pass
            return fr, prefer_col
        return r, c

    #
    def paste_to_selected(self, new_value):
        """Seçili hücreye tek tıkla yapıştır.
           - Edit modunu kapat, odağı sheet'e al.
           - Eğer seçim yoksa/ayıraçtaysa ilk uygun hücreyi seç.
           - Token sütunu (0) düzenlenemez.
           - MatrixLang/EmbedLang satırında Label sadece TR/EN olabilir.
        """
        if self.sheet is None:
            return

        self._ensure_sheet_focus()
        self._cancel_edit_if_any()

        r, c = self._ensure_valid_selection(prefer_col=2)
        if r is None or c is None:
            self.bell()
            return

        bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
        if bidx is None:
            self.bell()
            return

        # idx
        if c == 0:
            self.bell()
            return

        # Matrix/Embed
        if c == 2:
            tok = self.blocks[bidx][ridx].get('token')
            if annotation_model.is_matrixembed_locked(tok, new_value):
                self.bell()
                return
            self.blocks[bidx][ridx]['label'] = new_value
        elif c == 1:
            # Item
            self.blocks[bidx][ridx]['token'] = new_value
            self._renumber_tokens()
            self._refresh_sheet_idx_column()
        elif c == 3:
            # Gloss
            self.blocks[bidx][ridx]['gloss'] = new_value
        else:
            # Extra user-defined columns
            hdrs = self._all_headers()
            if 0 <= c < len(hdrs):
                key = hdrs[c]
                if key not in ("Token", "Item", "Label", "Gloss"):
                    self.blocks[bidx][ridx][key] = new_value

        self._mark_dirty()

        try:
            self.sheet.set_cell_data(r, c, new_value)
            if hasattr(self.sheet, "refresh"):
                self.sheet.refresh()
        except Exception:
            pass

        #
        self._ensure_sheet_focus()
        self._last_pos = (r, c)

    # matrix/Embed
    def _ensure_matrix_embed_consistency(self, text):
        if not self.cfg.get('FEATURE_EMBEDDED_LANGUAGE', False):
            return text
        want_matrix = self.cfg.get('FEATURE_MATRIX_LANGUAGE', False)
        blocks = text.split("\n\n")
        new_blocks = []
        for b in blocks:
            if not b.strip():
                new_blocks.append(b)
                continue
            lines = b.splitlines()
            has_embed = any(ln.startswith("EmbedLang") for ln in lines)
            has_matrix = any(ln.startswith("MatrixLang") for ln in lines)
            if want_matrix and has_embed and not has_matrix:
                tr = en = 0
                for ln in lines:
                    if "\t" in ln and not (ln.startswith("MatrixLang") or ln.startswith("EmbedLang")):
                        try:
                            parts = ln.split("\t")
                            if len(parts) >= 2:
                                lab = parts[-1].strip()
                                if lab == 'TR': tr += 1
                                elif lab == 'EN': en += 1
                        except Exception:
                            pass
                    else:
                        m = re.match(r"^(\S+)\s+(TR|EN|MIXED|UID|NE|OTHER|LANG3)\s*$", ln)
                        if m:
                            lab = m.group(2)
                            if lab == 'TR': tr += 1
                            elif lab == 'EN': en += 1
                mx = 'TR' if tr >= en else 'EN'
                lines.append(f"MatrixLang\t{mx}")
            new_blocks.append("\n".join(lines))
        return "\n\n".join(new_blocks)

    # (tksheet)
    def _populate_table(self, text):
        """Annotate çıktısını bloklara ayır, modelini kur, grid'e doldur (global idx ile).

        Transactional: parsing, renumbering, and grid construction all
        happen on local variables first. Only once every preparation step
        has succeeded does this method replace self.blocks/_row_index_map/
        _sep_rows and push the result into the visible sheet. If parsing or
        grid construction raises, self.blocks (and therefore the active
        dataset, once the caller syncs it) and the visible table are left
        completely unchanged -- run_pipeline's except branch is what the
        caller sees, never a partial or empty result.
        """
        if self.sheet is None:
            return

        new_blocks = annotation_model.parse_annotated_text_to_blocks(text, self._extra_headers)
        annotation_model.renumber_tokens(new_blocks)
        data, row_index_map, sep_rows = annotation_model.build_grid_view(
            new_blocks, self._extra_headers, skip_separator_after_empty_block=True
        )

        # Commit point: everything above can raise (and must leave prior
        # state untouched if it does); everything from here on is local
        # widget mutation, not project data, so it's safe to apply now.
        self.blocks = new_blocks
        self._row_index_map = row_index_map
        self._sep_rows = sep_rows

        try:
            self.sheet.headers(self._all_headers())
        except Exception:
            pass
        self.sheet.set_sheet_data(data)

        if self._sep_rows:
            try:
                self.sheet.set_row_colors(rows=list(self._sep_rows), bg="#151515", fg="#666666")
            except Exception:
                pass
        self._select_first_cell()

    def _reconstruct_text_from_blocks(self):
        return annotation_model.reconstruct_text_from_blocks(self.blocks, self._extra_headers)

    def _is_meta_row_token(self, tok: str) -> bool:
        return annotation_model.is_meta_row_token(tok)

    def _renumber_tokens(self):
        annotation_model.renumber_tokens(self.blocks)

    def _refresh_sheet_idx_column(self):
        """Modeldeki idx değerlerini grid'in 0. kolonuna geri bas.
        Meta satırlar idx boş kalır. Separator satırlar el değmeden kalır.
        """
        if self.sheet is None and self._full_sheet is None:
            return
        try:
            for vis_r, mapping in self._row_index_map.items():
                if vis_r in self._sep_rows:
                    continue
                bidx, ridx = mapping
                if bidx is None:
                    continue

                idxv = self.blocks[bidx][ridx].get("idx", "")
                idxs = "" if idxv is None else str(idxv)

                for sh in (self.sheet, self._full_sheet):
                    if sh is None:
                        continue
                    try:
                        sh.set_cell_data(vis_r, 0, idxs)
                    except Exception:
                        pass

            for sh in (self.sheet, self._full_sheet):
                if sh is not None and hasattr(sh, "refresh"):
                    sh.refresh()
        except Exception:
            pass

    def _rebuild_grid_from_model(self, select_row=None, select_col=2):
        """Rebuild grid UI from self.blocks without re-parsing text.
        Preserves blank rows, separator rows, and dynamic columns.
        """
        if self.sheet is None and self._full_sheet is None:
            return

        data, self._row_index_map, self._sep_rows = annotation_model.build_grid_view(
            self.blocks, self._extra_headers, skip_separator_after_empty_block=False
        )

        for sh in (self.sheet, self._full_sheet):
            if sh is None:
                continue
            try:
                sh.headers(self._all_headers())
            except Exception:
                pass
            try:
                sh.set_sheet_data(data)
            except Exception:
                continue

        if self._sep_rows:
            for sh in (self.sheet, self._full_sheet):
                if sh is None:
                    continue
                try:
                    sh.set_row_colors(rows=list(self._sep_rows), bg="#151515", fg="#666666")
                except Exception:
                    pass

        if select_row is not None:
            for sh in (self.sheet, self._full_sheet):
                if sh is None:
                    continue
                try:
                    sh.select_cell(select_row, select_col)
                    sh.see(select_row, select_col)
                except Exception:
                    pass
            self._last_pos = (select_row, select_col)
        else:
            self._select_first_cell()

    # tksheet events
    def _on_sheet_end_edit(self, event, sheet_obj=None):
        try:
            sheet = sheet_obj if sheet_obj is not None else self.sheet
            if sheet is None:
                return
            r, c = event.row, event.column
            if r in self._sep_rows:
                return
            bidx, ridx = annotation_model.resolve_row(self._row_index_map, self._sep_rows, r)
            if bidx is None:
                return

            # idx
            if c == 0:
                ov = self.blocks[bidx][ridx].get("idx", "")
                ov = "" if ov is None else str(ov)
                sheet.set_cell_data(r, 0, ov)
                if hasattr(sheet, "refresh"):
                    sheet.refresh()
                self.bell()
                return

            nv = sheet.get_cell_data(r, c)

            if c == 2:
                # Label
                if annotation_model.is_matrixembed_locked(self.blocks[bidx][ridx].get('token'), nv):
                    ov = self.blocks[bidx][ridx].get('label', '')
                    sheet.set_cell_data(r, 2, ov)
                    if hasattr(sheet, "refresh"):
                        sheet.refresh()
                    self.bell()
                    return
                self.blocks[bidx][ridx]['label'] = nv
            elif c == 1:
                # Item
                self.blocks[bidx][ridx]['token'] = nv
                # Token may have changed into (MatrixLang/EmbedLang/SentenceID)
                self._renumber_tokens()
                self._refresh_sheet_idx_column()
            elif c == 3:
                # Gloss
                self.blocks[bidx][ridx]['gloss'] = nv
            else:
                # Extra columns
                hdrs = self._all_headers()
                if 0 <= c < len(hdrs):
                    key = hdrs[c]
                    if key not in ("Token", "Item", "Label", "Gloss"):
                        self.blocks[bidx][ridx][key] = nv

            self._mark_dirty()

            # propagate change to the other sheet (if open)
            for other in (self.sheet, self._full_sheet):
                if other is None or other is sheet:
                    continue
                try:
                    other.set_cell_data(r, c, nv)
                    if hasattr(other, "refresh"):
                        other.refresh()
                except Exception:
                    pass

        except Exception:
            pass

    def _on_sheet_arrow(self, event):
        # ok
        if event.keysym == 'Up':
            self._move_cell(-1)
            return "break"
        if event.keysym == 'Down':
            self._move_cell(1)
            return "break"

    def _move_cell(self, delta):
        if self.sheet is None:
            return
        try:
            sel = self.sheet.get_currently_selected()
            r = getattr(sel, "row", None) if sel is not None else None
            c = getattr(sel, "column", None) if sel is not None else None
            if r is None or c is None:
                cells = self.sheet.get_selected_cells()
                if cells:
                    r, c = list(cells)[0]
        except Exception:
            r = c = None
        if r is None or c is None:
            return
        total = self.sheet.total_rows()
        n = r + delta
        while 0 <= n < total:
            if n not in self._sep_rows:
                try:
                    self.sheet.select_cell(n, c)
                    self.sheet.see(n, c)
                    self._last_pos = (n, c)
                except Exception:
                    pass
                return
            n += delta

    def _on_sheet_cell_select(self, event):
        try:
            r, c = event.row, event.column
        except Exception:
            return

        # mark sheet as active area
        self._active_area = "sheet"

        try:
            if r is not None and c is not None:
                self._last_pos = (r, c)
        except Exception:
            pass

        if r in self._sep_rows:
            total = self.sheet.total_rows() if self.sheet is not None else 0
            n = r + 1
            while n < total:
                if n not in self._sep_rows:
                    try:
                        self.sheet.select_cell(n, c)
                        self.sheet.see(n, c)
                        self._last_pos = (n, c)
                    except Exception:
                        pass
                    return
                n += 1

    def _select_first_cell(self):
        if self.sheet is None:
            return
        rows = self.sheet.total_rows()
        for r in range(rows):
            if r not in self._sep_rows:
                try:
                    self.sheet.select_cell(r, 2)  # Label
                    self.sheet.see(r, 2)
                    self._last_pos = (r, 2)
                except Exception:
                    pass
                break

    def _first_real_row(self):
        if self.sheet is None:
            return None
        rows = self.sheet.total_rows()
        for r in range(rows):
            if r not in self._sep_rows:
                return r
        return None


if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.freeze_support()
        # Prevent Tk app from starting inside multiprocessing worker processes
        if multiprocessing.current_process().name != "MainProcess":
            raise SystemExit(0)
    except Exception:
        pass

    # Single-instance lock (prevents double-launch and "reopen" effects)
    try:
        lock_dir = os.path.expanduser("~/.cs_annotator")
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, "tren.lock")
        _lock_fh = open(lock_path, "w")
        try:
            import fcntl
            fcntl.flock(_lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        # If locking is not available for some reason, continue rather than crash
        _lock_fh = None

    app = App()
    app.mainloop()