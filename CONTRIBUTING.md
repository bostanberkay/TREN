# Contributing

Thank you for your interest in contributing to TREN.

## Development setup

1. Clone the repository:

   ```bash
   git clone https://github.com/bostanberkay/TREN.git
   cd TREN
   ```

2. Create and activate a virtual environment (Python 3.11 or newer, with Tk support):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

   On Windows, use `powershell -ExecutionPolicy Bypass -File packaging\install_windows_deps.ps1 -Dev` instead of these two commands: `fasttext` has no Windows wheel and does not build with plain `pip`. The script also installs `requirements.txt` and `requirements-dev.txt`; see [Requirements in the README](README.md#requirements) for what it does and needs.

   The code is the `tren` package in `src/tren/`. Tests import it from `src/` (`pyproject.toml` sets this for pytest); to start the app with `python -m tren` from your checkout, also run `pip install --no-deps -e .`.

## Running tests

Run the automated test suite:

```bash
python -m pytest
```

Check that the core modules compile without syntax errors:

```bash
python -m py_compile src/tren/annotation_model.py src/tren/cs_pipeline.py src/tren/cs_annotator_app.py
```

Run the quickstart example as a smoke check:

```bash
python examples/quickstart.py
```

The quickstart is a deterministic demonstration with small built-in word lists and a stubbed fastText prediction (NER off), so it does not exercise the real models.

These are the same commands run automatically by the project's CI workflow on every push and pull request.

For an end-to-end check with the real fastText model, reranker, and NER models, run `python packaging/tren_selftest.py REPORT_DIR`. It drives the GUI programmatically and downloads the NER models on first use unless they are cached (add `--expect-ner-cached` to require the cache instead). CI does not run it on every push; the manual Windows package workflow runs it.

The GUI tests (`tests/test_*_gui.py`, and part of `tests/test_confidence_integration.py`) need a display and skip themselves when Tk cannot open one; run `python -m pytest -rs` to see any skip reasons, which include the actual Tk error (the full traceback is printed under "Tk start-up probes" in the pytest summary; see `tests/tk_probe.py`). With `TREN_REQUIRE_TK=1`, as in the Windows CI job, they are never skipped: a Tk start-up failure fails the run instead. `pyproject.toml` sets `--capture=sys` for pytest; keep it, since pytest's default fd-level capture makes Tk start-up fail intermittently on Windows (`invalid command name "tcl_findLibrary"`). On a headless Linux machine, run them under a virtual display as CI does: `xvfb-run -a python -m pytest -rs`. GUI tests must never open a real modal dialog or post a real popup menu: patch `messagebox`/`filedialog`/`simpledialog` calls and `Menu.tk_popup`, or the run will hang (a posted menu blocks on macOS until it is dismissed).

## Reporting bugs and getting help

Open an [issue](https://github.com/bostanberkay/TREN/issues) for bugs and suggestions, and include the TREN version (or commit), your operating system, and the steps to reproduce the problem. For other questions, write to bostanberkay@outlook.com.

## Pull requests

- Keep each pull request to one logical change.
- Keep commits focused and easy to review individually.
- Make sure CI passes before opening a pull request.
- Update relevant documentation (README, CHANGELOG, docstrings, etc.) when your change affects documented behavior.

## Coding principles

- Preserve existing annotation behavior. TREN's label set and annotation logic are treated as a stable contract; changes that alter labeling output require explicit discussion before being merged.
- Accompany behavior changes with tests. New or modified logic in `src/tren/annotation_model.py` or `src/tren/cs_pipeline.py` should come with corresponding tests in `tests/`.
- Prefer small, reviewable commits over large, sweeping changes.
- Avoid unrelated refactoring in the same change as a bug fix or feature addition.
