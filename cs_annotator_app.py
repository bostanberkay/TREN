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

from cs_pipeline import Annotator, DEFAULTS

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

        for vis_r in sorted(getattr(self, '_row_index_map', {}).keys()):
            if vis_r in getattr(self, '_sep_rows', set()):
                continue
            bidx, ridx = self._row_index_map.get(vis_r, (None, None))
            if bidx is None:
                continue
            try:
                row = self.blocks[bidx][ridx]
            except Exception:
                continue

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

        row['label'] = self._ag_label_var.get()
        row['gloss'] = self._ag_gloss_var.get()

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
        self.current_text = ""
        self.current_output = ""
        self.current_path = None

        # tablw model
        self.blocks = []
        self._row_index_map = {}
        self._sep_rows = set()


        self._core_headers = ["Token", "Item", "Label", "Gloss"]
        self._extra_headers = []  # user-added columns

        self.cfg = DEFAULTS.copy()

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
        filem.add_command(label="Save Output As... (⌘S / Ctrl+S)", command=self.save_output)
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
        menubar.add_cascade(label="Annotation", menu=annm)

        editm = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        editm.add_command(label="View Full Edit Window...", command=self.open_full_edit_window)
        menubar.add_cascade(label="Edit Window", menu=editm)

        toolsm = tk.Menu(menubar, tearoff=False, bg=DARK_BG, fg=DARK_FG)
        toolsm.add_command(label="Auto-Glossing Tool...", command=self.open_auto_glossing_tool)
        toolsm.add_separator()
        toolsm.add_command(label="Concordance (KWIC)...", command=self.open_concordance)
        toolsm.add_command(label="Show Sentence (Context)", command=self.show_sentence_context)
        toolsm.add_command(label="Word Frequency List...", command=self.open_word_frequency)
        menubar.add_cascade(label="Tools", menu=toolsm)

        self.config(menu=menubar)


    # Window close handling
    def _has_unsaved_progress(self):
        try:
            if self.txt_input.get("1.0", "end-1c").strip():
                return True
        except Exception:
            pass
        try:
            if getattr(self, 'blocks', None):
                # non-empty blocks means there is annotation state
                return True
        except Exception:
            pass
        return False

    def _on_close_request(self):
        # Ask to save progress before closing
        if not self._has_unsaved_progress():
            try:
                self.destroy()
            except Exception:
                pass
            return

        res = messagebox.askyesnocancel(
            "Close",
            "Save project progress before closing?"
        )
        if res is None:
            return  # cancel
        if res is True:
            self.save_project_progress()
        try:
            self.destroy()
        except Exception:
            pass

    # Project Save/Load
    def new_project(self):
        if messagebox.askyesno("New Project", "Start a new project? Unsaved progress will be lost.") is False:
            return

        self.blocks = []
        self._row_index_map = {}
        self._sep_rows = set()
        self._extra_headers = []
        self.cfg = DEFAULTS.copy()

        try:
            self.txt_input.delete("1.0", "end")
        except Exception:
            pass

        if self.sheet is not None:
            try:
                self.sheet.headers(self._all_headers())
                self.sheet.set_sheet_data([])
                if hasattr(self.sheet, "refresh"):
                    self.sheet.refresh()
            except Exception:
                pass

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
        name = simpledialog.askstring("Save Project", "Name your project save:")
        if not name:
            return

        try:
            os.makedirs(APP_DIR, exist_ok=True)
        except Exception:
            pass

        path = os.path.join(APP_DIR, name + PROJECT_EXT)

        # confirm overwrite
        if os.path.isfile(path):
            if messagebox.askyesno("Overwrite?", f"A save named '{name}' already exists. Overwrite?") is False:
                return

        try:
            text_cursor = self.txt_input.index("insert")
        except Exception:
            text_cursor = None

        try:
            sheet_sel = self.sheet.get_currently_selected() if self.sheet is not None else None
            grid_pos = (sheet_sel.row, sheet_sel.column) if sheet_sel is not None else None
        except Exception:
            grid_pos = None

        payload = {
            "version": 1,
            "name": name,
            "input_text": self.txt_input.get("1.0", "end-1c"),
            "cfg": self.cfg,
            "blocks": self.blocks,
            "extra_headers": self._extra_headers,
            "text_cursor": text_cursor,
            "grid_pos": grid_pos,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return

        try:
            with open(LAST_PROJECT_PTR, "w", encoding="utf-8") as f:
                json.dump({"path": path}, f)
        except Exception:
            pass

        messagebox.showinfo("Project saved", f"Saved project:\n{path}")

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

        self.cfg = payload.get("cfg", DEFAULTS.copy())
        self.blocks = payload.get("blocks", [])
        self._extra_headers = payload.get("extra_headers", [])

        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", payload.get("input_text", ""))

        self._renumber_tokens()
        self._rebuild_grid_from_model()

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
            self.cfg = payload.get("cfg", DEFAULTS.copy())
            self.blocks = payload.get("blocks", [])
            self._extra_headers = payload.get("extra_headers", [])
        except Exception:
            return

        try:
            self.txt_input.delete("1.0", "end")
            self.txt_input.insert("1.0", payload.get("input_text", ""))
        except Exception:
            pass

        try:
            self._renumber_tokens()
            self._rebuild_grid_from_model()
        except Exception:
            pass

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

        btn_save = ttk.Button(bar, text="Save", command=self.save_output, style='Dark.TButton')
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

            def _popup_grid_menu(e):
                try:
                    self._active_area = "sheet"
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
        for vis_r in sorted(self._row_index_map.keys()):
            if vis_r in self._sep_rows:
                continue
            bidx, ridx = self._row_index_map.get(vis_r, (None, None))
            if bidx is None:
                continue
            row = self.blocks[bidx][ridx]

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
        """Normalize token for frequency counting.
        - Casefold ON
        - Keep hyphenated forms (umbrella-ya stays)
        - Strip leading/trailing punctuation
        """
        if tok is None:
            return None
        s = str(tok).strip()
        if not s:
            return None
        # strip punctuation at edges only
        s = re.sub(r"^[\W_]+|[\W__]+$", "", s, flags=re.UNICODE)
        if not s:
            return None
        return s.casefold()

    def _compute_word_frequencies(self, allowed_labels=None):
        """Compute word frequencies from annotated blocks.
        Returns:
            freq: dict[token] -> total count
            by_label: dict[token] -> {label: count}
            total_tokens: int
        Meta rows (MatrixLang, EmbedLang, SentenceID, blanks) are excluded.
        """
        freq = {}
        by_label = {}
        total = 0

        for blk in getattr(self, 'blocks', []) or []:
            for r in blk:
                tok = r.get('token', '')
                if self._is_meta_row_token(tok):
                    continue
                norm = self._freq_normalize_token(tok)
                if not norm:
                    continue
                lab = str(r.get('label', '') or '').strip()
                if allowed_labels is not None and lab not in allowed_labels:
                    continue

                total += 1
                freq[norm] = freq.get(norm, 0) + 1
                if norm not in by_label:
                    by_label[norm] = {}
                if lab:
                    by_label[norm][lab] = by_label[norm].get(lab, 0) + 1

        return freq, by_label, total
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
        bidx, ridx = self._row_index_map.get(r, (None, None))
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
        bidx, ridx = self._row_index_map.get(r, (None, None))
        if bidx is None:
            self.bell()
            return
        tok = self.blocks[bidx][ridx].get('token')
        if tok in ("MatrixLang", "EmbedLang") and new_value not in ("TR", "EN"):
            self.bell()
            return
        self.blocks[bidx][ridx]['label'] = new_value
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
        self.cfg[key] = bool(val)

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

    def run_pipeline(self):
        txt = self.txt_input.get("1.0", "end-1c")
        if not txt.strip():
            messagebox.showwarning("Empty", "No input text.")
            return
        try:
            if self.annotator is None:
                self.annotator = Annotator()
            out = self.annotator.annotate(txt, self.cfg)
            out = self._ensure_matrix_embed_consistency(out)
            self._populate_table(out)
            self.current_output = out
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
        """Convert grid rows to TXT exactly like the UI: tab-separated rows.
        Blank grid rows become blank lines (block separators).

        Behavior:
        - If idx is blank (meta rows), omit the idx column in the output.
        - Trailing empty fields are removed.
        """
        out_lines = []
        for rr in rows:
            if rr is None:
                out_lines.append("")
                continue

            r = ["" if v is None else str(v) for v in rr]
            ncol = len(self._all_headers())
            while len(r) < ncol:
                r.append("")
            r = r[:ncol]

            if all(x.strip() == "" for x in r):
                out_lines.append("")
                continue

            idx = r[0].strip()
            fields = r[1:] if idx == "" else r

            while fields and fields[-1].strip() == "":
                fields.pop()

            out_lines.append("\t".join(fields))

        while out_lines and out_lines[-1] == "":
            out_lines.pop()
        return "\n".join(out_lines)

    def save_output(self):
        if not getattr(self, 'blocks', None):
            messagebox.showwarning("Empty", "No output to save.")
            return

        path = filedialog.asksaveasfilename(
            title="Save output",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("CSV", "*.csv"), ("All", "*.*")]
        )
        if not path:
            return

        base, ext = os.path.splitext(path)
        ext = ext.lower()

        # Default to .txt
        if ext not in (".txt", ".csv"):
            ext = ".txt"
            path = base + ext

        # Prefer exporting from the visible grid
        sheet_rows = self._get_sheet_rows_for_export()

        if ext == ".txt":
            if sheet_rows is not None:
                text = self._sheet_rows_to_txt(sheet_rows)
            else:
                text = self._reconstruct_text_from_blocks()
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
            return

        # Save CSV only
        try:
            if sheet_rows is None:
                # fallback: approximate from blocks and preserve block separators
                sheet_rows = []
                for bi, b in enumerate(getattr(self, 'blocks', [])):
                    for r in b:
                        rr = [
                            str(r.get('idx', '')),
                            str(r.get('token', '')),
                            str(r.get('label', '')),
                            str(r.get('gloss', '')),
                        ]
                        for h in self._extra_headers:
                            rr.append(str(r.get(h, '')))
                        sheet_rows.append(rr)
                    if bi < len(getattr(self, 'blocks', [])) - 1:
                        sheet_rows.append(["" for _ in self._all_headers()])  # block separator

            with open(path, "w", encoding="utf-8", newline="") as cf:
                w = csv.writer(cf)
                # Preserve the UI headers exactly as shown in the grid
                w.writerow(self._get_sheet_headers_for_export())
                for rr in sheet_rows:
                    w.writerow(rr)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return

        messagebox.showinfo("Saved", f"Saved to:\n{path}")

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
            bidx, ridx = self._row_index_map.get(r, (None, None))
            if bidx is None:
                continue

            for dc, val in enumerate(row_vals):
                c = c0 + dc
                if c == 0 or c >= len(hdrs):
                    continue

                # Matrix/Embed constraint only for Label column
                if hdrs[c] == "Label":
                    tok = self.blocks[bidx][ridx].get('token')
                    if tok in ("MatrixLang", "EmbedLang") and val not in ("TR", "EN"):
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
            bidx, ridx = self._row_index_map.get(r, (None, None))
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

        bidx, ridx = self._row_index_map.get(r, (None, None))
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

        bidx, ridx = self._row_index_map.get(r, (None, None))
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
        self._rebuild_grid_from_model(select_row=None, select_col=2)

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

        bidx, ridx = self._row_index_map.get(r, (None, None))
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
            if tok in ("MatrixLang", "EmbedLang") and new_value not in ("TR", "EN"):
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
        """Annotate çıktısını bloklara ayır, modelini kur, grid'e doldur (global idx ile)."""
        self.blocks = []
        self._row_index_map = {}
        self._sep_rows = set()
        if self.sheet is None:
            return

        # 1) blocks
        raw_blocks = []
        for b in text.split("\n\n"):
            rows = []
            for ln in b.splitlines():
                ln = ln.strip()
                if not ln:
                    continue

                parts = ln.split("\t")
                token = label = None

                if len(parts) >= 2:
                    if len(parts) == 2:
                        token, label = parts[0].strip(), parts[1].strip()
                    else:
                        token, label = parts[1].strip(), parts[2].strip()
                else:
                    m = re.match(r"^(\S+)\s+(TR|EN|MIXED|UID|NE|OTHER|LANG3)\s*$", ln)
                    if m:
                        token, label = m.group(1), m.group(2)

                if token is None:
                    continue

                rr = {"idx": 0, "token": token, "label": label, "gloss": ""}
                for h in self._extra_headers:
                    rr.setdefault(h, "")
                rows.append(rr)
            raw_blocks.append(rows)

        self.blocks = raw_blocks

        # 2) Global
        self._renumber_tokens()

        # 3) Grid
        data = []
        row_cursor = 0
        for bidx, rows in enumerate(self.blocks):
            for ridx, r in enumerate(rows):
                idxv = r.get("idx", "")
                idxs = "" if idxv is None else str(idxv)
                vals = [idxs, r.get("token", ""), r.get("label", ""), r.get("gloss", "")]
                for h in self._extra_headers:
                    vals.append(r.get(h, ""))
                data.append(vals)
                self._row_index_map[row_cursor] = (bidx, ridx)
                row_cursor += 1
            if bidx < len(self.blocks) - 1 and rows:
                data.append(["" for _ in self._all_headers()])
                self._row_index_map[row_cursor] = (None, None)
                self._sep_rows.add(row_cursor)
                row_cursor += 1

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
        """Fallback TXT reconstruction from the Python model.

        Includes extra user-defined columns. For meta rows (blank idx), idx is omitted.
        Trailing empty fields are trimmed.
        """
        self._renumber_tokens()
        out_blocks = []
        for rows in getattr(self, 'blocks', []):
            lines = []
            for r in rows:
                idx = str(r.get('idx', '') or '').strip()
                tok = str(r.get('token', '') or '')
                lab = str(r.get('label', '') or '')
                glo = str(r.get('gloss', '') or '')

                if not tok:
                    continue

                extras = [str(r.get(h, '') or '') for h in self._extra_headers]

                if idx == "":
                    fields = [tok, lab, glo] + extras
                else:
                    fields = [idx, tok, lab, glo] + extras

                while fields and str(fields[-1]).strip() == "":
                    fields.pop()

                lines.append("\t".join(fields))
            out_blocks.append("\n".join(lines))
        return "\n\n".join(out_blocks)

    def _is_meta_row_token(self, tok: str) -> bool:
        """Rows that should NOT be counted as tokens for numbering."""
        if tok is None:
            return False
        t = str(tok).strip()
        if not t:
            # blank rows are structural/editable placeholders
            return True
        if t in ("MatrixLang", "EmbedLang"):
            return True
        if t.lower() in ("sentenceid", "sentid", "sentence_id", "sent_id"):
            return True
        return False

    def _renumber_tokens(self):
        """Assign sequential token ids only to non-meta rows."""
        g = 1
        for rows in self.blocks:
            for r in rows:
                tok = r.get("token", "")
                if self._is_meta_row_token(tok):
                    r["idx"] = ""
                    continue
                r["idx"] = g
                g += 1

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

        self._row_index_map = {}
        self._sep_rows = set()

        data = []
        row_cursor = 0
        for bidx, rows in enumerate(self.blocks):
            for ridx, r in enumerate(rows):
                # ensure extra keys exist
                for h in self._extra_headers:
                    r.setdefault(h, "")

                idxv = r.get("idx", "")
                idxs = "" if idxv is None else str(idxv)
                vals = [idxs, r.get("token", ""), r.get("label", ""), r.get("gloss", "")]
                for h in self._extra_headers:
                    vals.append(r.get(h, ""))

                data.append(vals)
                self._row_index_map[row_cursor] = (bidx, ridx)
                row_cursor += 1

            if bidx < len(self.blocks) - 1:
                data.append(["" for _ in self._all_headers()])
                self._row_index_map[row_cursor] = (None, None)
                self._sep_rows.add(row_cursor)
                row_cursor += 1

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
            bidx, ridx = self._row_index_map.get(r, (None, None))
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
                if self.blocks[bidx][ridx].get('token') in ("MatrixLang", "EmbedLang"):
                    if nv not in ("TR", "EN"):
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