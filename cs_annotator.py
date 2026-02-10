#!/usr/bin/env python


import os
import json
import re
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from tkinter.scrolledtext import ScrolledText

APP_TITLE = "TR–EN CS Text Annotator (Sprint 1)"
DARK_BG = "#1e1e1e"
DARK_FG = "#eaeaea"
DARK_FG_DIM = "#aaaaaa"
GUTTER_BG = "#252525"
GUTTER_FG = "#777777"
STATUS_BG = "#2a2a2a"

LABEL_COLORS = {
    "TR": "#5cb85c",
    "EN": "#5bc0de",
    "MIXED": "#b074d0",
    "UID": "#f0ad4e",
    "NE": "#d2b55b",
    "OTHER": "#999999",
    "MATRIX": "#bbbbbb",    # MatrixLang line
    "EMBED": "#aaaaaa",     # EmbedLang line
}

DEFAULT_CONFIG = {
    "ner_enabled": True,
    "feature_language_per_item": True,
    "feature_matrix_language": False,
    "feature_embedded_language": False,
    "thresholds": {
        "FT_EN_MIN": 0.80,
        "FT_TR_MIN": 0.80,
        "MIXED_STRICT": True
    },
    "boundary": {
        "mode": "blank_lines",
        "separator": "---"
    }
}

class LineNumberCanvas(tk.Canvas):
    """Left gutter with line numbers."""
    def __init__(self, master, text_widget, **kwargs):
        super().__init__(master, **kwargs)
        self.text_widget = text_widget
        self.config(bg=GUTTER_BG, highlightthickness=0, width=50)
        self.bind("<Button-1>", lambda e: "break")
        self.text_widget.bind("<KeyRelease>", self.redraw)
        self.text_widget.bind("<MouseWheel>", self.redraw)
        self.text_widget.bind("<Button-4>", self.redraw)
        self.text_widget.bind("<Button-5>", self.redraw)
        self.text_widget.bind("<<Change>>", self.redraw)
        self.text_widget.bind("<Configure>", self.redraw)

    def redraw(self, event=None):
        self.delete("all")
        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(45, y, anchor="ne", text=linenum, fill=GUTTER_FG, font=("Menlo", 11))
            i = self.text_widget.index(f"{i}+1line")


class Editor(ScrolledText):
    """Main editor with dark theme and per-label tag colors."""
    def __init__(self, master, **kwargs):
        super().__init__(master, undo=True, wrap="none", **kwargs)
        # dark palette
        self.config(bg=DARK_BG, fg=DARK_FG, insertbackground=DARK_FG,
                    selectbackground="#444", padx=6, pady=6, font=("Menlo", 12))
        # define tags for labels
        for label, color in LABEL_COLORS.items():
            self.tag_configure(f"LABEL_{label}", foreground=color)
        self.tag_configure("ITALIC", font=("Menlo", 12, "italic"), foreground=DARK_FG_DIM)

        self.tag_configure("KWIC_HIT", background="#3a3a3a")

        # event to update statusbar line/col
        self.bind("<KeyRelease>", self._notify_cursor)
        self.bind("<ButtonRelease>", self._notify_cursor)

        # custom event for gutter redraw
        self.bind("<<TextModified>>", lambda e: None)  # placeholder kısmı

    def _notify_cursor(self, event=None):
        if hasattr(self.master.master, "update_status"):
            self.master.master.update_status()

    def apply_syntax_colors(self):
        """Color label lines by their label on each line end: '\\tLABEL'."""
        # Clear all label tags first
        for label in LABEL_COLORS.keys():
            self.tag_remove(f"LABEL_{label}", "1.0", "end")
        self.tag_remove("ITALIC", "1.0", "end")

        # apply tags per line
        total = int(self.index('end-1c').split('.')[0])
        for ln in range(1, total + 1):
            line = self.get(f"{ln}.0", f"{ln}.end")
            if not line.strip():
                continue

            if line.startswith("MatrixLang\t"):
                self.tag_add("LABEL_MATRIX", f"{ln}.0", f"{ln}.end")
                self.tag_add("ITALIC", f"{ln}.0", f"{ln}.end")
                continue
            if line.startswith("EmbedLang\t"):
                self.tag_add("LABEL_EMBED", f"{ln}.0", f"{ln}.end")
                self.tag_add("ITALIC", f"{ln}.0", f"{ln}.end")
                continue

            # normal token line: TOKEN<TAB>LABEL
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    lbl = parts[-1].strip()
                    if lbl in LABEL_COLORS:
                        self.tag_add(f"LABEL_{lbl}", f"{ln}.0", f"{ln}.end")


class AnnotatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1300x800")
        self.configure(bg=DARK_BG)

        self.project_dir = None
        self.config_data = json.loads(json.dumps(DEFAULT_CONFIG))  # copy
        self.current_file = None

        self._build_menubar()
        self._build_toolbar()
        self._build_main_panes()
        self._build_statusbar()

        self.bind_shortcuts()

        # concordance (KWIC) window state
        self._conc_win = None
        self._conc_query_var = tk.StringVar(value="")
        self._conc_ctx_var = tk.IntVar(value=30)
        self._conc_ci_var = tk.BooleanVar(value=True)
        self._conc_regex_var = tk.BooleanVar(value=False)
        self._conc_tree = None
        self._conc_count_var = tk.StringVar(value="0 matches")
        self._conc_hit_spans = {}  # tree_iid -> (start, end)

    # UI Builders
    def _build_menubar(self):
        menubar = tk.Menu(self, tearoff=False)

        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="New Project", command=self.new_project)
        m_file.add_command(label="Open Project", command=self.open_project)
        m_file.add_separator()
        m_file.add_command(label="Save Project", command=self.save_project)
        m_file.add_command(label="Save As...", command=self.save_as)
        m_file.add_command(label="Export .conll", command=self.export_conll)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=m_file)

        m_project = tk.Menu(menubar, tearoff=False)
        m_project.add_command(label="Load Text (file)", command=self.load_text_file)
        m_project.add_command(label="Paste Text", command=self.paste_text)
        m_project.add_separator()
        m_project.add_command(label="Preprocess", command=self.preprocess_text)
        m_project.add_separator()
        m_project.add_command(label="Settings...", command=self.open_settings)
        menubar.add_cascade(label="Project", menu=m_project)

        m_process = tk.Menu(menubar, tearoff=False)
        m_process.add_command(label="Run", command=self.run_pipeline)
        m_process.add_command(label="Stop", command=self.stop_pipeline)
        m_process.add_separator()
        m_process.add_checkbutton(label="Toggle NER (default ON)",
                                  command=self.toggle_ner, onvalue=True, offvalue=False,
                                  variable=tk.BooleanVar(value=self.config_data["ner_enabled"]))
        menubar.add_cascade(label="Process", menu=m_process)

        m_annot = tk.Menu(menubar, tearoff=False)
        m_annot.add_command(label="Set TR", command=lambda: self.set_label_for_current("TR"))
        m_annot.add_command(label="Set EN", command=lambda: self.set_label_for_current("EN"))
        m_annot.add_command(label="Set MIXED", command=lambda: self.set_label_for_current("MIXED"))
        m_annot.add_command(label="Set UID", command=lambda: self.set_label_for_current("UID"))
        m_annot.add_command(label="Set NE", command=lambda: self.set_label_for_current("NE"))
        m_annot.add_command(label="Set OTHER", command=lambda: self.set_label_for_current("OTHER"))
        menubar.add_cascade(label="Annotate", menu=m_annot)

        m_tools = tk.Menu(menubar, tearoff=False)
        m_tools.add_command(label="Concordance (KWIC)...", command=self.open_concordance)
        menubar.add_cascade(label="Tools", menu=m_tools)

        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="Manual", command=self.show_manual)
        m_help.add_command(label="About", command=lambda: messagebox.showinfo("About", APP_TITLE))
        menubar.add_cascade(label="Help", menu=m_help)

        self.config(menu=menubar)

    def _build_toolbar(self):
        tb = ttk.Frame(self)
        tb.pack(side="top", fill="x")
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TFrame", background=DARK_BG)
        style.configure("TButton", background="#333", foreground=DARK_FG)
        style.map("TButton", background=[("active", "#444")])

        ttk.Button(tb, text="New", command=self.new_project).pack(side="left", padx=4, pady=4)
        ttk.Button(tb, text="Open", command=self.open_project).pack(side="left", padx=4, pady=4)
        ttk.Button(tb, text="Run", command=self.run_pipeline).pack(side="left", padx=4, pady=4)
        ttk.Button(tb, text="Save", command=self.save_project).pack(side="left", padx=4, pady=4)
        ttk.Button(tb, text="Export", command=self.export_conll).pack(side="left", padx=4, pady=4)

    def _build_main_panes(self):
        # 3-paned layout
        self.main_panes = ttk.Panedwindow(self, orient="horizontal")
        self.main_panes.pack(fill="both", expand=True)

        # Explorer
        left = ttk.Frame(self.main_panes)
        left.config(width=250)
        ttk.Label(left, text="Project Explorer", background=DARK_BG, foreground=DARK_FG).pack(anchor="w", padx=6, pady=6)
        self.explorer = tk.Listbox(left, bg=DARK_BG, fg=DARK_FG, selectbackground="#444", height=30)
        self.explorer.pack(fill="both", expand=True, padx=6, pady=(0,6))
        self.explorer.bind("<<ListboxSelect>>", self.on_explorer_select)
        self.main_panes.add(left, weight=1)

        # Center
        center = ttk.Frame(self.main_panes)
        center.pack_propagate(False)
        self.editor = Editor(center)
        self.editor.pack(side="right", fill="both", expand=True)

        self.linenumbers = LineNumberCanvas(center, text_widget=self.editor, width=50)
        self.linenumbers.pack(side="left", fill="y")
        self.main_panes.add(center, weight=3)

        # Annotation Panel
        right = ttk.Frame(self.main_panes)
        ttk.Label(right, text="Annotation", background=DARK_BG, foreground=DARK_FG).pack(anchor="w", padx=6, pady=6)

        btns = ttk.Frame(right)
        btns.pack(anchor="n", padx=6, pady=6)

        for lbl in ["TR", "EN", "MIXED", "UID", "NE", "OTHER"]:
            b = ttk.Button(btns, text=lbl, command=lambda L=lbl: self.set_label_for_current(L))
            b.pack(fill="x", pady=2)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=6, pady=10)
        self.ner_var = tk.BooleanVar(value=self.config_data["ner_enabled"])
        chk = ttk.Checkbutton(right, text="NER enabled", variable=self.ner_var, command=self.toggle_ner)
        chk.pack(anchor="w", padx=6, pady=4)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=6, pady=10)

        self.log = ScrolledText(right, height=10, bg=DARK_BG, fg=DARK_FG, insertbackground=DARK_FG)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        self.main_panes.add(right, weight=1)

    def _build_statusbar(self):
        self.status = tk.Label(self, text="Ready", anchor="w", bg=STATUS_BG, fg=DARK_FG)
        self.status.pack(side="bottom", fill="x")

    # Helpers
    def update_status(self, msg=None):
        # cursor position
        try:
            idx = self.editor.index(tk.INSERT)
            line, col = idx.split(".")
            prefix = f"Ln {line}, Col {int(col)+1}"
        except Exception:
            prefix = "Ln -, Col -"
        text = prefix + (" | " + msg if msg else "")
        self.status.config(text=text)
        # redraw gutter
        self.linenumbers.redraw()

    def log_msg(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def refresh_explorer(self):
        self.explorer.delete(0, "end")
        if not self.project_dir:
            return
        files = []
        for root, dirs, fs in os.walk(self.project_dir):
            for f in fs:
                if f.startswith("."):
                    continue
                files.append(os.path.relpath(os.path.join(root, f), self.project_dir))
        for f in sorted(files):
            self.explorer.insert("end", f)

    # Project Ops
    def new_project(self):
        name = simpledialog.askstring("New Project", "Project name:")
        if not name:
            return
        base = os.path.abspath("projects")
        os.makedirs(base, exist_ok=True)
        proj = os.path.join(base, name)
        if os.path.exists(proj):
            messagebox.showerror("Error", "Project already exists.")
            return
        os.makedirs(proj, exist_ok=True)
        os.makedirs(os.path.join(proj, "logs"), exist_ok=True)
        self.project_dir = proj
        # write default config
        self.save_config()
        # clear editor
        self.editor.delete("1.0", "end")
        self.current_file = None
        self.refresh_explorer()
        self.update_status("New project created.")

    def open_project(self):
        proj = filedialog.askdirectory(title="Open Project Folder")
        if not proj:
            return
        if not os.path.isdir(proj):
            messagebox.showerror("Error", "Not a folder.")
            return
        self.project_dir = proj
        # read config if exists
        cfg_path = os.path.join(proj, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
        else:
            self.save_config()
        self.refresh_explorer()
        self.editor.delete("1.0", "end")
        self.current_file = None
        self.update_status("Project opened.")

    def save_project(self):
        if not self.project_dir:
            messagebox.showwarning("No project", "Create or open a project first.")
            return
        # if current file is set, save editor content there
        if not self.current_file:
            # default save raw.txt
            path = os.path.join(self.project_dir, "raw.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.editor.get("1.0", "end-1c"))
            self.current_file = path
        else:
            with open(self.current_file, "w", encoding="utf-8") as f:
                f.write(self.editor.get("1.0", "end-1c"))
        self.save_config()
        self.refresh_explorer()
        self.update_status("Project saved.")

    def save_as(self):
        if not self.project_dir:
            messagebox.showwarning("No project", "Create or open a project first.")
            return
        path = filedialog.asksaveasfilename(initialdir=self.project_dir, defaultextension=".txt")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.editor.get("1.0", "end-1c"))
        self.current_file = path
        self.refresh_explorer()
        self.update_status("Saved as.")

    def export_conll(self):
        if not self.project_dir:
            messagebox.showwarning("No project", "Create/open a project first.")
            return
        path = os.path.join(self.project_dir, "output_token.conll")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.editor.get("1.0", "end-1c"))
        self.refresh_explorer()
        self.update_status("Exported to output_token.conll")

    def save_config(self):
        if not self.project_dir:
            return
        path = os.path.join(self.project_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)

    def on_explorer_select(self, event):
        if not self.project_dir:
            return
        sel = self.explorer.curselection()
        if not sel:
            return
        relpath = self.explorer.get(sel[0])
        path = os.path.join(self.project_dir, relpath)
        if os.path.isdir(path):
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", content)
        self.editor.apply_syntax_colors()
        self.current_file = path
        self.update_status(f"Opened {relpath}")

    # Project Flow
    def load_text_file(self):
        if not self.project_dir:
            messagebox.showwarning("No project", "Create or open a project first.")
            return
        p = filedialog.askopenfilename(title="Select text file", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not p:
            return
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", content)
        self.editor.apply_syntax_colors()
        # save as raw.txt
        raw_path = os.path.join(self.project_dir, "raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.current_file = raw_path
        self.refresh_explorer()
        self.update_status(f"Loaded text into raw.txt")

    def paste_text(self):
        if not self.project_dir:
            messagebox.showwarning("No project", "Create or open a project first.")
            return
        try:
            content = self.clipboard_get()
        except tk.TclError:
            content = ""
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", content)
        self.editor.apply_syntax_colors()
        raw_path = os.path.join(self.project_dir, "raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.current_file = raw_path
        self.refresh_explorer()
        self.update_status(f"Pasted text into raw.txt")

    def preprocess_text(self):
        """
        Split into items & tokens:
        - If multiple lines raw text: each non-empty line = sentence
        - Tokenize by whitespace, keep punctuation as tokens if separate
        - Output format:
            WORD<TAB>UID   (default placeholder label)
            [blank line between sentences]
        """
        text = self.editor.get("1.0", "end-1c")
        lines = text.splitlines()
        out_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                out_lines.append("")
                continue
            # naive tokenize
            tokens = s.split()
            for tok in tokens:
                # Initialize with UID
                out_lines.append(f"{tok}\tUID")
            out_lines.append("")  # blank line per sentence
        new_text = "\n".join(out_lines).rstrip() + "\n"

        # put into editor and save as preprocessed.txt
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", new_text)
        self.editor.apply_syntax_colors()
        if self.project_dir:
            pp = os.path.join(self.project_dir, "preprocessed.txt")
            with open(pp, "w", encoding="utf-8") as f:
                f.write(new_text)
            self.current_file = pp
        self.refresh_explorer()
        self.update_status("Preprocess done.")

    # Settings ve process
    def open_settings(self):
        dlg = tk.Toplevel(self)
        dlg.title("Settings")
        dlg.config(bg=DARK_BG)
        dlg.geometry("380x320")

        v_ner = tk.BooleanVar(value=self.config_data["ner_enabled"])
        v_lang = tk.BooleanVar(value=self.config_data["feature_language_per_item"])
        v_matrix = tk.BooleanVar(value=self.config_data["feature_matrix_language"])
        v_embed = tk.BooleanVar(value=self.config_data["feature_embedded_language"])

        frm = ttk.Frame(dlg)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Checkbutton(frm, text="NER enabled", variable=v_ner).pack(anchor="w", pady=4)
        ttk.Checkbutton(frm, text="Language per item", variable=v_lang).pack(anchor="w", pady=4)
        ttk.Checkbutton(frm, text="Matrix language", variable=v_matrix).pack(anchor="w", pady=4)
        ttk.Checkbutton(frm, text="Embedded language", variable=v_embed).pack(anchor="w", pady=4)

        ttk.Separator(frm, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(frm, text="Thresholds (for pipeline, not used yet)").pack(anchor="w")

        th = self.config_data["thresholds"]
        en_var = tk.DoubleVar(value=th["FT_EN_MIN"])
        tr_var = tk.DoubleVar(value=th["FT_TR_MIN"])
        strict_var = tk.BooleanVar(value=th["MIXED_STRICT"])

        f2 = ttk.Frame(frm); f2.pack(fill="x", pady=4)
        ttk.Label(f2, text="FT_EN_MIN").grid(row=0, column=0, sticky="w")
        ttk.Entry(f2, textvariable=en_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(f2, text="FT_TR_MIN").grid(row=1, column=0, sticky="w")
        ttk.Entry(f2, textvariable=tr_var, width=8).grid(row=1, column=1, padx=6)
        ttk.Checkbutton(f2, text="MIXED_STRICT", variable=strict_var).grid(row=2, column=0, columnspan=2, sticky="w")

        def save_and_close():
            self.config_data["ner_enabled"] = v_ner.get()
            self.config_data["feature_language_per_item"] = v_lang.get()
            self.config_data["feature_matrix_language"] = v_matrix.get()
            self.config_data["feature_embedded_language"] = v_embed.get()
            self.config_data["thresholds"]["FT_EN_MIN"] = float(en_var.get())
            self.config_data["thresholds"]["FT_TR_MIN"] = float(tr_var.get())
            self.config_data["thresholds"]["MIXED_STRICT"] = bool(strict_var.get())
            self.save_config()
            dlg.destroy()
            self.update_status("Settings saved.")

        ttk.Button(frm, text="Save", command=save_and_close).pack(pady=10)

    def run_pipeline(self):
        """
        Stub: In Sprint 2 we will:
          - read editor blocks (items)
          - call your LID/MIXED/Matrix/Embed pipeline
          - re-write results into the same editor content:
            TOKEN<TAB>LABEL
            [MatrixLang<TAB>TR/EN]  if selected
            [EmbedLang<TAB>TR/EN]   if selected and present
            [blank line]
          - re-color tags
        """
        self.log_msg("[Run] Starting pipeline... (stub)")
        # Here we simply re-color
        self.editor.apply_syntax_colors()
        self.update_status("Run finished. (stub)")

    def stop_pipeline(self):
        # if we implement threaded processing later, set cancel flags here
        self.log_msg("[Stop] Requested. (stub)")
        self.update_status("Stopped. (stub)")

    def toggle_ner(self):
        self.config_data["ner_enabled"] = self.ner_var.get()
        self.save_config()
        self.update_status("NER toggled.")

    # Annotation ops
    def set_label_for_current(self, label):
        """
        Change label for current line from TAB-split format: TOKEN<TAB>LABEL
        """
        try:
            idx = self.editor.index(tk.INSERT)
            line_no = int(idx.split(".")[0])
            line_text = self.editor.get(f"{line_no}.0", f"{line_no}.end")
            if not line_text.strip():
                return
            if line_text.startswith("MatrixLang\t") or line_text.startswith("EmbedLang\t"):
                # ignore summary lines
                return
            if "\t" not in line_text:
                # not a token line
                return
            parts = line_text.split("\t")
            parts[-1] = label
            new_line = "\t".join(parts)
            self.editor.delete(f"{line_no}.0", f"{line_no}.end")
            self.editor.insert(f"{line_no}.0", new_line)
            self.editor.apply_syntax_colors()
            self.update_status(f"Set label: {label}")
        except Exception as e:
            self.log_msg(f"[Annot] Error: {e}")

    # Tools
    def open_concordance(self):
        """Open KWIC concordance window."""
        # raise existing
        if self._conc_win is not None and self._conc_win.winfo_exists():
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

        top = ttk.Frame(win)
        top.pack(side="top", fill="x", padx=10, pady=10)

        ttk.Label(top, text="Query:").grid(row=0, column=0, sticky="w")
        ent = ttk.Entry(top, textvariable=self._conc_query_var, width=40)
        ent.grid(row=0, column=1, sticky="we", padx=(6, 10))

        ttk.Label(top, text="Context (chars):").grid(row=0, column=2, sticky="w")
        spn = ttk.Spinbox(top, from_=5, to=200, textvariable=self._conc_ctx_var, width=6)
        spn.grid(row=0, column=3, sticky="w", padx=(6, 10))

        chk_ci = ttk.Checkbutton(top, text="Case-insensitive", variable=self._conc_ci_var)
        chk_ci.grid(row=0, column=4, sticky="w", padx=(0, 10))

        chk_rx = ttk.Checkbutton(top, text="Regex", variable=self._conc_regex_var)
        chk_rx.grid(row=0, column=5, sticky="w", padx=(0, 10))

        btn = ttk.Button(top, text="Search", command=self._conc_run_search)
        btn.grid(row=0, column=6, sticky="e")

        top.columnconfigure(1, weight=1)

        mid = ttk.Frame(win)
        mid.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("left", "kwic", "right")
        tree = ttk.Treeview(mid, columns=cols, show="headings", selectmode="browse")
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

        bottom = ttk.Frame(win)
        bottom.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        ttk.Label(bottom, textvariable=self._conc_count_var).pack(side="left")

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

        # store refs
        self._conc_win = win
        self._conc_tree = tree

        # bindings
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
        self._conc_hit_spans = {}
        try:
            self.editor.tag_remove("KWIC_HIT", "1.0", "end")
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
        if ctx < 1:
            ctx = 1
        if ctx > 500:
            ctx = 500

        text = self.editor.get("1.0", "end-1c")
        if not text:
            self._conc_clear()
            return

        flags = 0
        if self._conc_ci_var.get():
            flags |= re.IGNORECASE

        # compile matcher
        try:
            if self._conc_regex_var.get():
                pattern = re.compile(query, flags)
            else:
                pattern = re.compile(re.escape(query), flags)
        except Exception as e:
            messagebox.showerror("Regex error", str(e))
            return

        # clear previous
        self._conc_hit_spans = {}
        self._conc_clear()

        def _norm(s: str) -> str:
            return " ".join(s.replace("\n", " ").replace("\t", " ").split())

        max_hits = 5000
        n = 0
        for m in pattern.finditer(text):
            start, end = m.span()
            left = text[max(0, start - ctx):start]
            try:
                kwic = m.group(0)
            except Exception:
                kwic = text[start:end]
            right = text[end:min(len(text), end + ctx)]

            left_s = _norm(left)
            kwic_s = _norm(kwic)
            right_s = _norm(right)

            iid = f"h{start}_{end}_{n}"
            try:
                self._conc_tree.insert("", "end", iid=iid, values=(left_s, kwic_s, right_s))
                self._conc_hit_spans[iid] = (start, end)
            except Exception:
                # if iid collision, fall back to auto
                iid2 = ""
                try:
                    iid2 = self._conc_tree.insert("", "end", values=(left_s, kwic_s, right_s))
                    self._conc_hit_spans[iid2] = (start, end)
                except Exception:
                    pass

            n += 1
            if n >= max_hits:
                break

        if n >= max_hits:
            self._conc_count_var.set(f"{n} matches (truncated)")
        else:
            self._conc_count_var.set(f"{n} matches")

        # Auto-select first hit if any
        try:
            first = self._conc_tree.get_children()
            if first:
                self._conc_tree.selection_set(first[0])
                self._conc_tree.focus(first[0])
        except Exception:
            pass

        try:
            self._conc_tree.focus_set()
        except Exception:
            pass
        try:
            kids = list(self._conc_tree.get_children())
            if kids:
                self._conc_tree.selection_set(kids[0])
                self._conc_tree.focus(kids[0])
                self._conc_tree.see(kids[0])
                self._conc_jump_to_hit()
        except Exception:
            pass

    def _conc_nav(self, delta: int):
        if self._conc_tree is None:
            return
        kids = list(self._conc_tree.get_children())
        if not kids:
            return

        sel = None
        try:
            s = self._conc_tree.selection()
            if s:
                sel = s[0]
        except Exception:
            sel = None

        if sel in kids:
            i = kids.index(sel)
        else:
            i = 0

        j = i + delta
        if j < 0:
            j = 0
        if j >= len(kids):
            j = len(kids) - 1

        try:
            self._conc_tree.selection_set(kids[j])
            self._conc_tree.focus(kids[j])
            self._conc_tree.see(kids[j])
        except Exception:
            pass

        self._conc_jump_to_hit()

    def _conc_jump_to_hit(self, event=None):
        if self._conc_tree is None:
            return
        sel = None
        try:
            s = self._conc_tree.selection()
            if s:
                sel = s[0]
        except Exception:
            sel = None
        if not sel:
            return

        span = self._conc_hit_spans.get(sel)
        if not span:
            return
        start, end = span

        try:
            s_idx = f"1.0+{start}c"
            e_idx = f"1.0+{end}c"
            self.editor.tag_remove("KWIC_HIT", "1.0", "end")
            self.editor.tag_add("KWIC_HIT", s_idx, e_idx)
            self.editor.see(s_idx)
            self.editor.mark_set("insert", s_idx)
            self.editor.focus_set()
            self.update_status("KWIC jump")
        except Exception:
            pass

    # Help
    def show_manual(self):
        t = (
            "Workflow:\n"
            "1) File → New Project (or Open Project)\n"
            "2) Project → Load Text / Paste Text (saved as raw.txt)\n"
            "3) Project → Preprocess (token-per-line; blank line per sentence)\n"
            "4) Process → Run (pipeline will be added in Sprint 2)\n"
            "5) Manual edits: use right panel buttons or hotkeys 1..6\n"
            "6) File → Save Project / Export\n\n"
            "Hotkeys:\n"
            "  1..6 → TR, EN, MIXED, UID, NE, OTHER\n"
            "  (Next/Prev token/sentence hotkeys will be added next)\n"
        )
        messagebox.showinfo("Manual", t)

    # Shortcuts
    def bind_shortcuts(self):
        self.bind("1", lambda e: self.set_label_for_current("TR"))
        self.bind("2", lambda e: self.set_label_for_current("EN"))
        self.bind("3", lambda e: self.set_label_for_current("MIXED"))
        self.bind("4", lambda e: self.set_label_for_current("UID"))
        self.bind("5", lambda e: self.set_label_for_current("NE"))
        self.bind("6", lambda e: self.set_label_for_current("OTHER"))

if __name__ == "__main__":
    app = AnnotatorApp()
    app.mainloop()