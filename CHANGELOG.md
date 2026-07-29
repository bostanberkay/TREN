# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.2.0] - 2026-07-29

### Added

- Integrated the validated MIXED-token reranker into the production annotation pipeline (`reranker_integration.py`, wired into `cs_annotator_app.py`'s `run_pipeline()`): it now runs automatically, as a post-processing stage after `Annotator.annotate()`, on every annotation request, promoting eligible `UID`/`NE`/`TR` labels to `MIXED` when the frozen Phase 5F model's probability clears its validated 0.85 threshold. The model (`resources/models/model.joblib`, `vectorizer.joblib`, `metadata.json`) loads lazily on the first annotation request and is cached for the session; if dependencies or model resources are missing, corrupted, or fail metadata validation, annotation falls back to the original rule-based output with no crash and no interruption. `Annotator.annotate()`'s internals, the parser, and the `.trenproj`/TXT/CSV schemas are unchanged.

### Documentation

- Synchronized README.md, CHANGELOG.md, and CLAUDE.md with the now-production-integrated reranker: architecture, the exact pipeline order, loading/fallback behavior, the frozen feature-batch configuration, and the current (560-test) test count.

## [1.1.0] - 2026-07-29

### Added

- Extracted GUI-independent annotation-state logic from `cs_annotator_app.py` into a new `annotation_model.py` module.
- Added an automated unit test suite (`tests/`) covering `annotation_model.py` and `cs_pipeline.py`.
- Added a GitHub Actions CI workflow that runs a syntax check, the quickstart example, and the test suite on every push and pull request.
- Added a runnable quickstart example (`examples/quickstart.py`) demonstrating the annotation pipeline.
- Developed an isolated MIXED-token reranker research module (`mixed_reranker.py`, `tools/build_reranker_dataset.py`, `tools/train_mixed_reranker.py`) with its own test suite (`tests/test_mixed_reranker.py`), kept out of the production annotation pipeline during its development and evaluation. The frozen baseline (Batch A + Batch C + a pruned Batch G) reaches MIXED precision 0.893 / recall 0.781 / F1 0.8333 on a held-out test split; two additional feature groups (Batch B, Batch D) were evaluated and experimentally rejected, with rationale documented in the README's "MIXED-Token Reranker" section. This module was developed with substantial AI (Claude) assistance across an iterative, benchmarked experimentation process.

### Documentation

- Added `CONTRIBUTING.md` with development setup, testing, and pull request guidelines.
- Added `CODE_OF_CONDUCT.md`.

## [1.0.0]

### Added

- Initial public release of TREN.