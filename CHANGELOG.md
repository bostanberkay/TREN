# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **Tokenizer** (`cs_pipeline.tokenize`): `@mentions`, `#hashtags`, emoji, and hyphen/underscore-joined alphanumeric codes (e.g. `A7K-204`, `TR-9081-ZX`) are now preserved as single tokens instead of being silently dropped (`@`/`#`/emoji, since `\w` never matches them) or fragmented on the separator (codes). `is_other_token` gained a `CODE_RE` check so these codes are correctly classified `OTHER`. This changes token boundaries (and therefore TXT-export row counts) for input containing any of the above — call this out when reviewing/consuming exports of text that includes them. Also fixes a related regression found during evaluation of this change: the separator-joined-number alternative previously matched on a bare leading digit run (`*`), so it wrongly won over the general word pattern and split digit-prefixed words like `20li`/`3d`/`6da` into two tokens; it now requires at least one actual separator group (`+`).
- **`resources/frequent_tr_words.txt`**: removed three English function-word entries (`the`, `a`, `i`) that were present in the top-1000 Turkish frequency tier, causing them to resolve `TR` ahead of the English lexicon check regardless of context. `in` was also tried and reverted — a controlled evaluation against the real annotated corpus found one concrete regression (a genuine Turkish-context use of `in` that flipped from correctly `TR` to incorrectly `EN`), so it stays in the Turkish list pending a more targeted fix. `her` (a genuinely common Turkish word, "every/each") was intentionally left in place — its collision with the English possessive pronoun is a real cross-lingual ambiguity with no context-free fix, not a data error.

Two further changes — a Turkish-stem morphological fallback for `Annotator._choose_label` (UID over-generation), and matching an apostrophe-split base against recognized NER entities in `Annotator._build_ne_map` (NE priority for `Ankara'da`-style tokens) — were implemented and evaluated against both the real annotated corpus and a synthetic benchmark, but reverted after the evaluation showed real regressions (the Turkish-stem fallback pre-empts MIXED detection and converts genuine `MIXED`/`UID` tokens to `TR`; the NE-priority change suppresses at least one genuine apostrophe-bearing `MIXED` form). Automatic `LANG3` production was also implemented and reverted: it conflicts with the previously-documented manual-only `LANG3` behavior, and real-corpus evaluation found real false positives with no real corpus examples to confirm a true positive. All three remain candidates for future work, not shipped here.

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