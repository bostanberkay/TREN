# Contributing

Thank you for your interest in contributing to TREN.

## Development setup

1. Clone the repository:

   ```bash
   git clone https://github.com/bostanberkay/TREN.git
   cd TREN
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Running tests

Run the automated test suite:

```bash
python -m pytest
```

Check that the core modules compile without syntax errors:

```bash
python -m py_compile annotation_model.py cs_pipeline.py cs_annotator_app.py
```

Run the quickstart example as a smoke check:

```bash
python examples/quickstart.py
```

These are the same commands run automatically by the project's CI workflow on every push and pull request.

## Pull requests

- Keep each pull request to one logical change.
- Keep commits focused and easy to review individually.
- Make sure CI passes before opening a pull request.
- Update relevant documentation (README, docstrings, etc.) when your change affects documented behavior.

## Coding principles

- Preserve existing annotation behavior. TREN's label set and annotation logic are treated as a stable contract; changes that alter labeling output require explicit discussion before being merged.
- Accompany behavior changes with tests. New or modified logic in `annotation_model.py` or `cs_pipeline.py` should come with corresponding tests in `tests/`.
- Prefer small, reviewable commits over large, sweeping changes.
- Avoid unrelated refactoring in the same change as a bug fix or feature addition.
