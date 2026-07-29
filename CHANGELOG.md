# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Extracted GUI-independent annotation-state logic from `cs_annotator_app.py` into a new `annotation_model.py` module.
- Added an automated unit test suite (`tests/`) covering `annotation_model.py` and `cs_pipeline.py`.
- Added a GitHub Actions CI workflow that runs a syntax check, the quickstart example, and the test suite on every push and pull request.
- Added a runnable quickstart example (`examples/quickstart.py`) demonstrating the annotation pipeline.
- Added an experimental, isolated MIXED-token reranker research module (`mixed_reranker.py`, `tools/build_reranker_dataset.py`, `tools/train_mixed_reranker.py`) with its own test suite (`tests/test_mixed_reranker.py`). Not imported by, and does not affect, the production annotation pipeline. The active experimental baseline (Batch A + Batch C + a pruned Batch G) reaches MIXED precision 0.893 / recall 0.781 / F1 0.8333 on a frozen held-out test split; two additional feature groups (Batch B, Batch D) were evaluated and experimentally rejected, with rationale documented in the README's "Experimental: MIXED-Token Reranker" section. This module was developed with substantial AI (Claude) assistance across an iterative, benchmarked experimentation process.

### Documentation

- Added `CONTRIBUTING.md` with development setup, testing, and pull request guidelines.
- Added `CODE_OF_CONDUCT.md`.
- Documented the experimental MIXED-token reranker (architecture, active baseline, feature-batch inventory, and benchmark results) in the README.

## [1.0.0]

### Added

- Initial public release of TREN.