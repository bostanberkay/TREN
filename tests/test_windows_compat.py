"""Regression tests for the Windows port: the single-instance lock (fcntl does
not exist on Windows, and the old code exited silently there), Windows file-name
rules for project saves, LF-only text exports so files are byte-identical
across platforms, BOM handling for Notepad input files, the first-use NER
download notice, and startup without a console (stdout/stderr are None).
None of these needs a Tk display; they run on every platform.
"""
import json
import os
import subprocess
import sys
import textwrap
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cs_annotator_app as caa  # noqa: E402
import cs_pipeline  # noqa: E402
import annotation_model  # noqa: E402

TURKISH_DIR = "Kullanıcı Şükrü Çağlar ğüöı"

ANNOTATED = (
    "SentenceID\t1\nkitap\tTR\namazing\tEN\nboss'um\tMIXED\nMatrixLang\tTR\nEmbedLang\tEN\n\n"
    "SentenceID\t2\nbugün\tTR\nMatrixLang\tTR\nEmbedLang\t-\n"
)


# --- single-instance lock -------------------------------------------------

def test_lock_is_exclusive_and_released_on_close(tmp_path):
    lock_dir = str(tmp_path / TURKISH_DIR / ".cs_annotator")
    first = caa.acquire_single_instance_lock(lock_dir)
    assert first is not None
    try:
        assert caa.acquire_single_instance_lock(lock_dir) is None
    finally:
        first.close()
    again = caa.acquire_single_instance_lock(lock_dir)
    assert again is not None
    again.close()


def test_lock_held_by_another_process_refuses(tmp_path):
    lock_dir = str(tmp_path / TURKISH_DIR)
    holder = caa.acquire_single_instance_lock(lock_dir)
    assert holder is not None
    try:
        code = textwrap.dedent(f"""
            import sys, types
            from unittest.mock import MagicMock
            sys.modules["stanza"] = MagicMock()
            sys.path.insert(0, {ROOT!r})
            import cs_annotator_app as caa
            print("refused" if caa.acquire_single_instance_lock({lock_dir!r}) is None else "acquired")
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             encoding="utf-8", timeout=120)
        assert out.stdout.strip().splitlines()[-1] == "refused", out.stderr
    finally:
        holder.close()


def test_entry_point_no_longer_requires_fcntl():
    src = open(os.path.join(ROOT, "cs_annotator_app.py"), encoding="utf-8").read()
    main = src[src.index('if __name__ == "__main__":'):]
    assert "import fcntl" not in main
    assert "acquire_single_instance_lock(APP_DIR)" in main


# --- Windows file-name rules for project saves -----------------------------

@pytest.mark.parametrize("name", ["proje", "Proje Çalışma ğüşiöç", "v1.2 final", "console", "a-b_c"])
def test_valid_windows_save_names(name):
    assert annotation_model.windows_save_name_error(name) is None


@pytest.mark.parametrize("name", ["a:b", "a/b", "a\\b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b",
                                  "tab\tname", "trailing ", "trailing.", "CON", "con", "nul.txt", "COM1", "LPT9"])
def test_invalid_windows_save_names(name):
    assert annotation_model.windows_save_name_error(name)


def _save_stub(monkeypatch, tmp_path, name):
    errors = []
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: name)
    monkeypatch.setattr(caa.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(caa, "APP_DIR", str(tmp_path))
    return errors


def test_save_rejects_colon_name_on_windows_without_writing(monkeypatch, tmp_path):
    errors = _save_stub(monkeypatch, tmp_path, "proje:1")
    monkeypatch.setattr(caa, "IS_WINDOWS", True)
    assert caa.App.save_project_progress(types.SimpleNamespace()) is False
    assert errors and "not allowed" in errors[0][1]
    assert os.listdir(tmp_path) == []


def test_save_name_check_is_windows_only(monkeypatch, tmp_path):
    """macOS behaviour is unchanged: the Windows rule is not applied there."""
    _save_stub(monkeypatch, tmp_path, "proje:1")
    monkeypatch.setattr(caa, "IS_WINDOWS", False)
    reached = []

    class Stub:
        def __getattr__(self, attr):
            reached.append(attr)
            raise RuntimeError("stop after validation")

    with pytest.raises(RuntimeError):
        caa.App.save_project_progress(Stub())
    assert reached


# --- exports and project files are LF-only and identical on every platform --

def _export_stub(blocks):
    ds = annotation_model.make_dataset("Veri 1", "kitap amazing boss'um. bugün", blocks)
    other = annotation_model.make_dataset("Other", "", [])
    stub = types.SimpleNamespace(datasets=[other, ds], _active_dataset_index=0,
                                 _core_headers=["Token", "Item", "Label", "Gloss"])
    return stub, ds


@pytest.mark.parametrize("fmt,expected", [
    ("TXT", lambda b: annotation_model.reconstruct_text_from_blocks(b, [])),
    ("CoNLL", annotation_model.blocks_to_conll),
    ("JSONL", lambda b: annotation_model.blocks_to_jsonl(b, "Veri 1")),
])
def test_text_exports_are_lf_only_and_exact(tmp_path, fmt, expected):
    blocks = annotation_model.parse_annotated_text_to_blocks(ANNOTATED, [])
    annotation_model.renumber_tokens(blocks)
    stub, ds = _export_stub(blocks)
    out_dir = tmp_path / TURKISH_DIR
    out_dir.mkdir()
    path = str(out_dir / f"çıktı dosyası.{fmt.lower()}")
    caa.App._write_dataset_export(stub, ds, fmt, path)
    raw = open(path, "rb").read()
    assert b"\r" not in raw
    assert raw.decode("utf-8") == expected(blocks)


def test_csv_export_uses_csv_module_line_endings(tmp_path):
    """CSV keeps the csv module's own \\r\\n terminator on every platform (as before)."""
    blocks = annotation_model.parse_annotated_text_to_blocks(ANNOTATED, [])
    stub, ds = _export_stub(blocks)
    path = str(tmp_path / "çıktı.csv")
    caa.App._write_dataset_export(stub, ds, "CSV", path)
    raw = open(path, "rb").read()
    assert raw.startswith(b"Token,Item,Label,Gloss\r\n")
    assert b"\r\r" not in raw


def test_project_save_writes_lf_only(monkeypatch, tmp_path):
    app_dir = tmp_path / TURKISH_DIR
    monkeypatch.setattr(caa, "APP_DIR", str(app_dir))
    monkeypatch.setattr(caa, "LAST_PROJECT_PTR", str(app_dir / "last_project.json"))
    monkeypatch.setattr(caa.simpledialog, "askstring", lambda *a, **k: "Proje Ğ")
    monkeypatch.setattr(caa.messagebox, "showinfo", lambda *a, **k: None)
    blocks = annotation_model.parse_annotated_text_to_blocks(ANNOTATED, [])
    ds = annotation_model.make_dataset("Veri 1", "kitap", blocks)
    stub = types.SimpleNamespace(
        cfg=dict(caa.DEFAULTS), datasets=[ds], _active_dataset_index=0, sheet=None,
        txt_input=types.SimpleNamespace(index=lambda *_: "1.0"),
        _sync_active_dataset_from_live=lambda: None, _mark_clean=lambda: None,
    )
    assert caa.App.save_project_progress(stub) is True
    raw = (app_dir / ("Proje Ğ" + caa.PROJECT_EXT)).read_bytes()
    assert b"\r" not in raw
    datasets, _ = annotation_model.datasets_from_payload(json.loads(raw.decode("utf-8")))
    assert datasets[0]["blocks"] == ds["blocks"]


def test_input_file_bom_is_not_part_of_first_token(tmp_path):
    path = tmp_path / "notepad girdi ş.txt"
    path.write_bytes("﻿kitap amazing\r\nbugün".encode("utf-8"))
    text = caa.App._read_utf8_text_file(None, str(path))
    assert text == "kitap amazing\nbugün"


def test_input_file_still_rejects_non_utf8(tmp_path):
    path = tmp_path / "cp1254.txt"
    path.write_bytes("çalışma".encode("cp1254"))
    with pytest.raises(UnicodeDecodeError):
        caa.App._read_utf8_text_file(None, str(path))


# --- first-use NER model download notice ------------------------------------

def _ner_stub(ner_enabled=True, ner_loaded=False):
    return types.SimpleNamespace(cfg={"NER_ENABLED": ner_enabled},
                                 annotator=types.SimpleNamespace(ner=object() if ner_loaded else None))


def test_ner_notice_shown_when_models_missing_and_cancel_stops_run(monkeypatch):
    asked = []
    monkeypatch.setattr(caa, "ner_models_cached", lambda: False)
    monkeypatch.setattr(caa.messagebox, "askokcancel", lambda *a, **k: asked.append(a) or False)
    stub = _ner_stub()
    with pytest.raises(RuntimeError, match="NER models were not downloaded"):
        caa.App._confirm_ner_model_download(stub)
    assert asked and asked[0][0] == "Download NER models"
    assert stub.cfg["NER_ENABLED"] is True  # never silently turned off


def test_ner_notice_ok_proceeds(monkeypatch):
    monkeypatch.setattr(caa, "ner_models_cached", lambda: False)
    monkeypatch.setattr(caa.messagebox, "askokcancel", lambda *a, **k: True)
    caa.App._confirm_ner_model_download(_ner_stub())


@pytest.mark.parametrize("stub,cached", [
    (_ner_stub(ner_enabled=False), False),
    (_ner_stub(ner_loaded=True), False),
    (_ner_stub(), True),
])
def test_ner_notice_not_shown_when_not_needed(monkeypatch, stub, cached):
    monkeypatch.setattr(caa, "ner_models_cached", lambda: cached)
    monkeypatch.setattr(caa.messagebox, "askokcancel",
                        lambda *a, **k: pytest.fail("download notice shown unnecessarily"))
    caa.App._confirm_ner_model_download(stub)


def test_ner_models_cached_reads_stanza_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "stanza önbellek"
    monkeypatch.setitem(sys.modules, "stanza.resources.common",
                        types.SimpleNamespace(DEFAULT_MODEL_DIR=str(model_dir)))
    assert cs_pipeline.ner_models_cached() is False
    (model_dir / "tr" / "tokenize").mkdir(parents=True)
    (model_dir / "tr" / "ner").mkdir()
    assert cs_pipeline.ner_models_cached() is False  # no resources.json yet
    (model_dir / "resources.json").write_text("{}", encoding="utf-8")
    assert cs_pipeline.ner_models_cached() is True


def test_ner_models_cached_never_blocks_on_its_own_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "stanza.resources.common", None)
    assert cs_pipeline.ner_models_cached() is True


# --- windowed startup: no console means sys.stdout/sys.stderr are None -------

def test_import_without_console_redirects_output_to_log(tmp_path):
    home = tmp_path / TURKISH_DIR
    home.mkdir()
    code = textwrap.dedent(f"""
        import sys
        from unittest.mock import MagicMock
        sys.modules["stanza"] = MagicMock()
        sys.stdout = None
        sys.stderr = None
        sys.path.insert(0, {ROOT!r})
        import cs_annotator_app
        print("stderr-marker", file=sys.stderr)
        print("stdout-marker")
        sys.stderr.flush()
    """)
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    log = (home / ".cs_annotator" / "tren.log").read_text(encoding="utf-8")
    assert "stderr-marker" in log and "stdout-marker" in log


# --- platform-specific UI strings ------------------------------------------

def test_monospace_font_is_platform_appropriate():
    assert caa.MONO_FONT == ("Consolas" if sys.platform == "win32" else "Menlo")


# --- Windows dependency installation: one fail-fast installer for CI and packaging

def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_windows_ci_and_packaging_share_the_installer():
    ci = _read(".github/workflows/ci.yml")
    split = ci.index("  test-windows:")
    linux_job, windows_job = ci[:split], ci[split:]
    package = _read(".github/workflows/windows-package.yml")
    for text in (windows_job, package):
        assert "./packaging/install_windows_deps.ps1" in text
        assert "pip install -r requirements.txt" not in text
    # macOS/Linux keep installing the unpatched PyPI release.
    assert "pip install -r requirements.txt" in linux_job
    assert "install_windows_deps" not in linux_job


def test_installer_builds_fasttext_first_stops_on_failure_and_checks_model():
    ps = _read("packaging/install_windows_deps.ps1")
    body = ps[ps.index("try {"):]  # skip the header comment, which names the same files
    steps = [body.index("build_fasttext_windows.py"), body.index("--no-deps"),
             body.index('"requirements.txt"'), body.index("check_fasttext.py")]
    assert steps == sorted(steps)
    assert '$ErrorActionPreference = "Stop"' in ps
    assert "if ($LASTEXITCODE -ne 0) { throw" in ps
