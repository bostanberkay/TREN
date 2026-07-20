# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Extracted GUI-independent annotation-state logic from `cs_annotator_app.py` into a new `annotation_model.py` module.
- Added an automated unit test suite (`tests/`) covering `annotation_model.py` and `cs_pipeline.py`.
- Added a GitHub Actions CI workflow that runs a syntax check, the quickstart example, and the test suite on every push and pull request.
- Added a runnable quickstart example (`examples/quickstart.py`) demonstrating the annotation pipeline.

### Documentation

- Added `CONTRIBUTING.md` with development setup, testing, and pull request guidelines.
- Added `CODE_OF_CONDUCT.md`.

## [1.0.0]

### Added

- Initial public release of TREN.