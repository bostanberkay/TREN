"""Packaging self-test for TREN: exercises the real pipeline and GUI end to end
without user interaction and writes a report.

Packaged build (bundled by packaging/TREN_windows.spec):
    TREN.exe --self-test REPORT_DIR [--expect-ner-cached]
Source tree:
    python packaging/tren_selftest.py REPORT_DIR [--expect-ner-cached]

REPORT_DIR receives selftest_report.json, the raw pipeline output and one
export per format, so a packaged run can be diffed against a source run.
Dialogs are replaced by recorders; nothing is written outside REPORT_DIR
except Stanza's model cache (first NER start downloads it). Exit code 0 means
every check passed. This does not replace a person using the GUI: windows are
driven programmatically, not by real mouse/keyboard input.
"""
import csv
import json
import os
import platform
import sys
import time
import traceback

SAMPLE_TEXT = (
    "Yarın sabah meeting'e geç kalacağım çünkü Ahmet Ankara'dan geliyor.\n"
    "Bu kitap gerçekten amazing, deadline'ı da cuma."
)
SECOND_TEXT = "Bugün hava çok güzel ve project'imizi bitirdik."
# Spaces and Turkish characters on purpose: exercises non-ASCII paths.
SAVE_NAME = "Proje Çalışma ğüşiöç"
EXPORT_DIR_NAME = "dışa aktarım klasörü"
APP_HOME_NAME = "uygulama evi Ğ"


class Report:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.checks = []
        self.info = {}

    def run(self, name, fn):
        start = time.time()
        try:
            detail = fn()
            self.checks.append({"name": name, "status": "pass", "detail": detail,
                                "seconds": round(time.time() - start, 2)})
        except Exception as e:
            self.checks.append({"name": name, "status": "fail",
                                "detail": f"{type(e).__name__}: {e}",
                                "traceback": traceback.format_exc(),
                                "seconds": round(time.time() - start, 2)})

    def ok(self):
        return bool(self.checks) and all(c["status"] == "pass" for c in self.checks)

    def write(self):
        data = {"ok": self.ok(), "info": self.info, "checks": self.checks}
        with open(os.path.join(self.out_dir, "selftest_report.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        lines = [f"[{c['status'].upper()}] {c['name']}: {c['detail']}" for c in self.checks]
        lines.append("RESULT: " + ("PASS" if self.ok() else "FAIL"))
        with open(os.path.join(self.out_dir, "selftest_report.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class Dialogs:
    """Stands in for every messagebox/filedialog/simpledialog call the app makes."""

    def __init__(self, caa):
        self.calls = []
        self.answers = {"askokcancel": True, "askyesno": True, "askyesnocancel": None}
        self.askstring_answer = None
        self.open_path = None
        for name in ("showinfo", "showwarning", "showerror", "askokcancel", "askyesno", "askyesnocancel"):
            setattr(caa.messagebox, name, self._make(name))
        caa.simpledialog.askstring = lambda *a, **k: self._record("askstring", a, self.askstring_answer)
        caa.filedialog.askopenfilename = lambda *a, **k: self._record("askopenfilename", a, self.open_path)
        caa.filedialog.asksaveasfilename = lambda *a, **k: self._record("asksaveasfilename", a, "")

    def _record(self, name, args, answer):
        self.calls.append((name, [str(x) for x in args]))
        return answer

    def _make(self, name):
        return lambda *a, **k: self._record(name, a, self.answers.get(name))

    def titles(self, name):
        return [args[0] for n, args in self.calls if n == name and args]

    def errors(self):
        return [args for n, args in self.calls if n == "showerror"]


def check_frozen_isolation():
    """In the packaged app every imported module must come from the bundle,
    never from a Python installation or source checkout on the machine."""
    if not getattr(sys, "frozen", False):
        return "not frozen (source run); skipped by design"
    bundle = os.path.normcase(os.path.abspath(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))))
    app_dir = os.path.normcase(os.path.abspath(os.path.dirname(sys.executable)))

    def inside(p):
        p = os.path.normcase(os.path.abspath(p))
        return p.startswith(bundle + os.sep) or p == bundle or p.startswith(app_dir + os.sep)

    outside = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and not inside(f):
            outside.append(f"{name}: {f}")
    bad_path = [p for p in sys.path if p and not inside(p)]
    _check(not outside, f"modules loaded from outside the bundle: {outside[:10]}")
    _check(not bad_path, f"sys.path entries outside the bundle: {bad_path}")
    return f"{len(sys.modules)} modules, all inside {bundle}"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    out_dir = os.path.abspath(argv[0])
    expect_ner_cached = "--expect-ner-cached" in argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    report = Report(out_dir)
    try:
        _run_all(report, out_dir, expect_ner_cached)
    except Exception:
        report.checks.append({"name": "self-test harness", "status": "fail",
                              "detail": traceback.format_exc()})
    report.write()
    return 0 if report.ok() else 1


def _run_all(report, out_dir, expect_ner_cached):
    report.info.update({
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd_at_start": os.getcwd(),
    })

    import cs_annotator_app as caa
    import cs_pipeline
    import reranking
    import confidence
    import annotation_model
    import tdk

    app_home = os.path.join(out_dir, APP_HOME_NAME)
    os.makedirs(app_home, exist_ok=True)
    caa.APP_DIR = app_home
    caa.LAST_PROJECT_PTR = os.path.join(app_home, "last_project.json")
    export_dir = os.path.join(out_dir, EXPORT_DIR_NAME)
    os.makedirs(export_dir, exist_ok=True)

    report.run("bundle isolation", check_frozen_isolation)

    def lock():
        lock_dir = os.path.join(out_dir, "kilit dizini ş")
        first = caa.acquire_single_instance_lock(lock_dir)
        _check(first is not None, "first lock not acquired")
        second = caa.acquire_single_instance_lock(lock_dir)
        _check(second is None, "second lock acquired while the first is held")
        first.close()
        third = caa.acquire_single_instance_lock(lock_dir)
        _check(third is not None, "lock not re-acquirable after release")
        third.close()
        return "held / refused / re-acquired"
    report.run("single-instance lock", lock)

    state = {}

    def resources():
        caa.App._set_runtime_workdir(None)
        report.info["resource_cwd"] = os.getcwd()
        for f in ("frequent_tr_words.txt", "frequent_en_words.txt", "lid.176.ftz"):
            _check(os.path.isfile(f), f"missing resource {f} in {os.getcwd()}")
        for f in ("model.joblib", "vectorizer.joblib", "metadata.json"):
            p = os.path.join(reranking.DEFAULT_MODEL_DIR, f)
            _check(os.path.isfile(p), f"missing reranker file {p}")
        return os.getcwd()
    report.run("bundled resources found", resources)

    def fasttext_lid():
        ann = cs_pipeline.Annotator()
        state["annotator"] = ann
        tr = ann._ft_predict("merhaba")
        en = ann._ft_predict("hello")
        _check(tr[0] == "TR" and en[0] == "EN", f"unexpected fastText predictions {tr} {en}")
        return f"merhaba={tr}, hello={en}"
    report.run("fastText language ID", fasttext_lid)

    def reranker():
        bundle = reranking.load_reranker_bundle()
        _check(bundle is not None, f"reranker unavailable: {reranking.last_load_failure}")
        state["bundle"] = bundle
        import sklearn
        return f"loaded (scikit-learn {sklearn.__version__})"
    report.run("MIXED reranker bundle", reranker)

    def ner_cache_state():
        cached = cs_pipeline.ner_models_cached()
        report.info["ner_models_cached_before_run"] = cached
        from stanza.resources.common import DEFAULT_MODEL_DIR
        report.info["stanza_model_dir"] = DEFAULT_MODEL_DIR
        if expect_ner_cached:
            _check(cached, f"expected cached Stanza models in {DEFAULT_MODEL_DIR}")
        return f"cached={cached} dir={DEFAULT_MODEL_DIR}"
    report.run("NER model cache state", ner_cache_state)

    def pipeline_with_ner():
        ann = state["annotator"]
        cfg = dict(cs_pipeline.DEFAULTS)
        _check(cfg["NER_ENABLED"], "NER is not enabled by default")
        out = ann.annotate(SAMPLE_TEXT, cfg)
        _check(ann.ner is not None, "NER pipeline was not created")
        out = reranking.apply_reranker(out, ann, cfg, state.get("bundle"))
        with open(os.path.join(out_dir, "pipeline_output.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        labels = dict(line.split("\t")[:2] for line in out.splitlines() if line.count("\t") >= 1)
        mixed_label = labels.get("meeting'e")
        _check(mixed_label == "MIXED", f"meeting'e labelled {mixed_label!r}")
        return {k: labels.get(k) for k in ("meeting'e", "Ahmet", "Ankara'dan", "amazing", "deadline'ı")}

    # GUI first, so a fresh Stanza cache is downloaded through the GUI's notice.
    _run_gui_checks(report, caa, annotation_model, confidence, tdk, export_dir)
    report.run("annotation pipeline with NER", pipeline_with_ner)


def _run_gui_checks(report, caa, annotation_model, confidence, tdk, export_dir):
    dialogs = Dialogs(caa)
    app = caa.App()
    app.update()

    def token_rows(block_index=0):
        return [(ri, r) for ri, r in enumerate(app.blocks[block_index])
                if not annotation_model.is_meta_row_token(r.get("token"))]

    def vis_row(bidx, ridx):
        return next(v for v, (b, r) in app._row_index_map.items() if b == bidx and r == ridx)

    def run_gui_pipeline():
        app._set_txt_input_text(SAMPLE_TEXT)
        app.run_pipeline()
        app.update()
        shown = "Download NER models" in dialogs.titles("askokcancel")
        report.info["ner_download_notice_shown"] = shown
        _check(not dialogs.errors(), f"error dialog during Run: {dialogs.errors()}")
        _check(token_rows(), "no token rows after Run")
        cached_before = report.info.get("ner_models_cached_before_run")
        _check(shown != bool(cached_before),
               f"download notice shown={shown} but models cached before run={cached_before}")
        with_conf = sum(1 for _ri, r in token_rows() if confidence.get_confidence(r) is not None)
        _check(with_conf == len(token_rows()), "confidence missing on some token rows")
        status = app._reranker_status_var.get()
        _check(status == "MIXED reranker: active", f"reranker status in the toolbar: {status!r}")
        return f"{len(token_rows())} tokens, toolbar: {status!r}"
    report.run("GUI Run (real pipeline)", run_gui_pipeline)

    def manual_edit():
        ri, row = token_rows()[0]
        vr = vis_row(0, ri)
        new_label = "EN" if row["label"] != "EN" else "TR"
        app.sheet.set_cell_data(vr, 2, new_label)
        app._on_sheet_end_edit(type("E", (), {"row": vr, "column": 2})(), sheet_obj=app.sheet)
        app.sheet.set_cell_data(vr, 3, "book-LOC")
        app._on_sheet_end_edit(type("E", (), {"row": vr, "column": 3})(), sheet_obj=app.sheet)
        app.update()
        row = app.blocks[0][ri]
        _check(row["label"] == new_label, f"label edit not stored: {row['label']}")
        _check(row["gloss"] == "book-LOC", f"gloss edit not stored: {row['gloss']}")
        _check(app._dirty, "edit did not mark the project dirty")
        return f"{row['token']}: {new_label} / book-LOC"
    report.run("manual label and gloss edit", manual_edit)

    def merge_cells():
        rows = token_rows()
        (ra, _), (rb, _) = rows[0], rows[1]
        before = len(rows)
        va, vb = vis_row(0, ra), vis_row(0, rb)
        app.sheet.deselect()
        app.sheet.create_selection_box(min(va, vb), 2, max(va, vb) + 1, 3)
        app.merge_selected_cells()
        app.update()
        dlg = next(w for w in app.winfo_children() if isinstance(w, caa.tk.Toplevel) and w.title() == "Merge Cells")
        widgets = []
        stack = [dlg]
        while stack:
            w = stack.pop()
            widgets.append(w)
            stack.extend(w.winfo_children())
        next(w for w in widgets if isinstance(w, caa.ttk.Combobox)).set("TR")
        next(w for w in widgets if isinstance(w, caa.ttk.Button) and w.cget("text") == "Confirm").invoke()
        app.update()
        _check(len(token_rows()) == before - 1, "merge did not reduce the token count")
        app.undo_merge_cells()
        app.update()
        _check(len(token_rows()) == before, "undo merge did not restore the token count")
        return f"{before} -> {before - 1} -> {before}"
    report.run("Merge Cells and undo", merge_cells)

    def full_edit():
        app.open_full_edit_window()
        app.update()
        _check(app._full_sheet is not None, "full edit sheet missing")
        _check(app._full_sheet.total_rows() == app.sheet.total_rows(), "full edit rows differ from main grid")
        app._full_win.destroy()
        app.update()
        return f"{app.sheet.total_rows()} rows"
    report.run("Full Edit window", full_edit)

    def confidence_review():
        app.open_uid_review_tool()
        app.update()
        _check(app._uid_win is not None and app._uid_win.winfo_exists(), "review window not open")
        n = len(getattr(app, "_uid_items", []) or [])
        app._uid_win.destroy()
        app.update()
        return f"{n} review items"
    report.run("Confidence Review tool", confidence_review)

    def tdk_checker():
        app._tdk_provider = tdk.MockDictionaryProvider(default_status=tdk.STATUS_NOT_FOUND)
        ri, _row = token_rows()[0]
        vr = vis_row(0, ri)
        app.sheet.deselect()
        app.sheet.create_selection_box(vr, 2, vr + 1, 3)
        app.update()
        app.open_tdk_checker_from_grid()
        deadline = time.time() + 10
        while app._tdk_status_var.get() in ("", "checking...") and time.time() < deadline:
            app.update()
            time.sleep(0.02)
        _check(app._tdk_win is not None and app._tdk_win.winfo_exists(), "TDK window not open")
        status = app._tdk_status_var.get()
        token = app._tdk_token_var.get()
        parser_status = app._tdk_parser_status_var.get()
        segments = app._tdk_segments_var.get()
        app._tdk_win.destroy()
        app.update()
        _check(token, "TDK Checker was not filled from the grid")
        _check(parser_status != "parser unavailable", "TDK parser unavailable")
        return f"token={token!r} lookup={status!r} parser={parser_status!r} segments={segments!r}"
    report.run("TDK Checker (mock dictionary, no network)", tdk_checker)

    def second_dataset():
        ds = app._create_dataset_from_text("Veri 2", SECOND_TEXT)
        app.datasets.append(ds)
        app._mark_dirty()
        app._load_dataset_into_live(len(app.datasets) - 1)
        app.update()
        _check(len(app.datasets) == 2 and app._active_dataset_index == 1, "second dataset not active")
        _check(token_rows(), "second dataset has no tokens")
        app._switch_dataset(0)
        app.update()
        _check(app._active_dataset_index == 0, "switch back failed")
        return [d["name"] for d in app.datasets]
    report.run("multiple datasets", second_dataset)

    def save_and_reopen():
        app._sync_active_dataset_from_live()
        before = json.dumps(annotation_model.datasets_to_payload(app.datasets, app._active_dataset_index),
                            ensure_ascii=False, sort_keys=True)
        dialogs.askstring_answer = SAVE_NAME
        _check(app.save_project_progress() is True, "save_project_progress returned False")
        path = os.path.join(caa.APP_DIR, SAVE_NAME + caa.PROJECT_EXT)
        _check(os.path.isfile(path), f"project file not written at {path}")
        with open(path, "rb") as f:
            raw = f.read()
        _check(b"\r" not in raw, "project file contains CR line endings")
        _check(not app._dirty, "project still dirty after save")
        payload = json.loads(raw.decode("utf-8"))
        _check(payload.get("version") == annotation_model.CURRENT_PROJECT_SCHEMA_VERSION, "unexpected version")
        app._set_txt_input_text("x")
        app.datasets[0]["name"] = "changed"
        app._mark_clean()
        dialogs.open_path = path
        app.open_project_save()
        app.update()
        app._sync_active_dataset_from_live()
        after = json.dumps(annotation_model.datasets_to_payload(app.datasets, app._active_dataset_index),
                           ensure_ascii=False, sort_keys=True)
        _check(json.loads(before)["datasets"] == json.loads(after)["datasets"], "reopened project differs from saved one")
        if caa.IS_WINDOWS:
            dialogs.askstring_answer = "bad:name"
            _check(app.save_project_progress() is False, "invalid Windows name was accepted")
            _check(not os.path.exists(os.path.join(caa.APP_DIR, "bad")), "':' name created a file")
        return path
    report.run("project save and reopen (non-ASCII path)", save_and_reopen)

    def exports():
        written = {}
        for ds_i, ds in enumerate(app.datasets):
            for fmt in app._EXPORT_FORMATS:
                path = os.path.join(export_dir, f"veri {ds_i + 1} çıktı{app._EXPORT_EXTENSIONS[fmt]}")
                app._write_dataset_export(ds, fmt, path)
                with open(path, "rb") as f:
                    raw = f.read()
                _check(raw.strip(), f"{fmt} export is empty")
                text = raw.decode("utf-8")
                if fmt == "CSV":
                    rows = list(csv.reader(text.splitlines()))
                    _check(rows[0][:4] == list(app._core_headers), f"CSV header {rows[0]}")
                else:
                    _check(b"\r" not in raw, f"{fmt} export contains CR line endings")
                if fmt == "JSONL":
                    for line in text.splitlines():
                        json.loads(line)
                written[os.path.basename(path)] = len(raw)
        return written
    report.run("TXT/CSV/CoNLL/JSONL export (non-ASCII path)", exports)

    def unsaved_work_on_run():
        snapshot = json.dumps(app.blocks, ensure_ascii=False, sort_keys=True, default=str)
        dialogs.answers["askokcancel"] = False
        app.run_pipeline()
        app.update()
        dialogs.answers["askokcancel"] = True
        _check(json.dumps(app.blocks, ensure_ascii=False, sort_keys=True, default=str) == snapshot,
               "declining the Run confirmation changed the annotations")
        _check("Run" in dialogs.titles("askokcancel"), "Run over existing annotations did not ask")
        return "Run asked before replacing annotations; declining kept them"
    report.run("unsaved work kept on Run", unsaved_work_on_run)

    def platform_bindings():
        mod = "Command" if caa.IS_MACOS else "Control"
        _check(app.bind_all(f"<{mod}-f>"), f"<{mod}-f> not bound")
        table = app.sheet.MT  # tksheet forwards Sheet.bind() to its table canvas
        if not caa.IS_MACOS:
            _check(not table.bind("<Button-2>"), "middle button opens the context menu")
        _check(table.bind("<Button-3>"), "right-click context menu not bound")
        return f"{mod}-f search, Button-3 context menu"
    report.run("platform key/mouse bindings", platform_bindings)

    def close_guard():
        ri, _row = token_rows()[0]
        app.blocks[0][ri]["gloss"] = "edited"
        app._mark_dirty()
        dialogs.answers["askyesnocancel"] = None
        app._on_close_request()
        app.update()
        _check(app.winfo_exists(), "Cancel on close still closed the window")
        dialogs.answers["askyesnocancel"] = True
        dialogs.askstring_answer = None
        app._on_close_request()
        app.update()
        _check(app.winfo_exists(), "Save cancelled at the name prompt, yet the window closed")
        dialogs.answers["askyesnocancel"] = False
        app._on_close_request()
        try:
            still_open = bool(app.winfo_exists())
        except Exception:
            still_open = False
        _check(not still_open, "Discard did not close the window")
        return "Cancel and aborted Save kept the window; Discard closed it"
    report.run("unsaved work kept on close", close_guard)

    try:
        app.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main(sys.argv[1:]))
