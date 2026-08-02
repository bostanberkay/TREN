# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.3.0] - 2026-08-02

### Added

- **NE Policy C** (`cs_pipeline.Annotator._build_ne_map`): a token backed only by a `TIME`-subtype Stanza entity match (`POLICY_C_EXCLUDED_NE_SUBTYPES = {"TIME"}`) is no longer classified `NE`, falling through to the existing non-NE labeling logic instead. Scoped to `TIME` only after evaluation against the real corpus found genuine gold-`NE` `MONEY` collisions that ruled out excluding that subtype too.
- **NE Policy D** (`cs_pipeline.Annotator._qualifies_for_policy_d`): a token backed only by an NE match is no longer kept `NE` when it is a direct English-lexicon match, absent from the Turkish lexicon, not the anonymization placeholder literal, not an acronym/all-caps/alphanumeric token, and has no valid Turkish nominal-suffix analysis — with a further guard requiring the other piece of a multi-word `ORGANIZATION` span to be a bare, alphabetic, capitalized word before the override applies (distinguishing genuine compound entity names from NER span-boundary noise). Any piece backed by `PERSON`/`LOCATION` evidence is unconditionally protected from Policy D; a single-piece `ORGANIZATION` span receives no such protection and may still lose `NE` status if it independently satisfies every other Policy D condition — only genuine multi-word `ORGANIZATION` compounds are guarded.
- **Strict residual verbal MIXED detector** (`mixed_reranker.evaluate_residual_verbal_promotion`, wired into `reranker_integration.apply_reranker()` as a second internal pass, strictly after the frozen Phase 5F reranker pass): may promote a token whose label is still `UID`/`TR` after the reranker pass to `MIXED`, using strict direct-English-lexicon evidence only (no fastText fallback), gated by 11 conditions including an explicit Turkish verbalizer/passive-inchoative marker, Turkish-lexicon absence of the stem, no lexicon-confirmed competing nominal analysis, uniqueness of the selected analysis, and exclusion of proper names/acronyms/codes/URLs/mentions/hashtags/apostrophe-bearing tokens. Adds one new closed-class suffix table (1st/3rd-person-plural verbal agreement, e.g. `-dık`/`-ık`/`-ız`) on top of the existing, unmodified `PHASE_4E` verbal-suffix level; the frozen model's own default level (`PHASE_4C1`) is never referenced or changed. Never touches `NE`, `EN`, `OTHER`, `LANG3`, or a token the reranker pass already promoted. A per-token exception here is caught and the token's label left unchanged, without interrupting the rest of the block.
- **Matrix/Embedded recomputation after promotions**: `reranker_integration.apply_reranker()` recomputes `MatrixLang`/`EmbedLang` (via the existing, unmodified `Annotator._decide_matrix_embed`) whenever either the reranker pass or the residual verbal pass changes a label in a sentence; blocks with no label change are returned byte-for-byte identical to the rule-based annotator's own output.
- **Expanded regression coverage**: 707 tests now pass (up from 560 at 1.2.0), including full coverage for NE Policies C/D (`tests/test_cs_pipeline.py`) and the residual verbal detector (`tests/test_mixed_reranker.py`, `tests/test_reranker_integration.py`) — parsing/candidate-generation, all 8 named regression targets, 26 native-Turkish negative controls, placement-after-reranker, fail-safe behavior, and Matrix/Embedded recomputation.

### Fixed

- **Tokenizer** (`cs_pipeline.tokenize`): `@mentions`, `#hashtags`, emoji, and hyphen/underscore-joined alphanumeric codes (e.g. `A7K-204`, `TR-9081-ZX`) are now preserved as single tokens instead of being silently dropped (`@`/`#`/emoji, since `\w` never matches them) or fragmented on the separator (codes). `is_other_token` gained a `CODE_RE` check so these codes are correctly classified `OTHER`. This changes token boundaries (and therefore TXT-export row counts) for input containing any of the above — call this out when reviewing/consuming exports of text that includes them. Also fixes a related regression found during evaluation of this change: the separator-joined-number alternative previously matched on a bare leading digit run (`*`), so it wrongly won over the general word pattern and split digit-prefixed words like `20li`/`3d`/`6da` into two tokens; it now requires at least one actual separator group (`+`).
- **`resources/frequent_tr_words.txt`**: removed three English function-word entries (`the`, `a`, `i`) that were present in the top-1000 Turkish frequency tier, causing them to resolve `TR` ahead of the English lexicon check regardless of context. `in` was also tried and reverted — a controlled evaluation against the real annotated corpus found one concrete regression (a genuine Turkish-context use of `in` that flipped from correctly `TR` to incorrectly `EN`), so it stays in the Turkish list pending a more targeted fix. `her` (a genuinely common Turkish word, "every/each") was intentionally left in place — its collision with the English possessive pronoun is a real cross-lingual ambiguity with no context-free fix, not a data error.

Two further changes — a Turkish-stem morphological fallback for `Annotator._choose_label` (UID over-generation), and matching an apostrophe-split base against recognized NER entities in `Annotator._build_ne_map` (NE priority for `Ankara'da`-style tokens) — were implemented and evaluated against both the real annotated corpus and a synthetic benchmark, but reverted after the evaluation showed real regressions (the Turkish-stem fallback pre-empts MIXED detection and converts genuine `MIXED`/`UID` tokens to `TR`; the NE-priority change suppresses at least one genuine apostrophe-bearing `MIXED` form). Automatic `LANG3` production was also implemented and reverted: it conflicts with the previously-documented manual-only `LANG3` behavior, and real-corpus evaluation found real false positives with no real corpus examples to confirm a true positive. All three remain candidates for future work, not shipped here — the UID→TR resolver evaluated alongside them is likewise **not integrated** into this release.

### Validation

- Production validated end-to-end on the real, manually-annotated corpus (primary evidence; accuracy 0.8725, MIXED F1 0.8778) and two synthetic, LLM-authored benchmarks used as secondary diagnostic evidence only, not external validation (first synthetic: accuracy 0.8660, MIXED F1 0.7312; benchmark v2, adjudicated gold: accuracy 0.9212, MIXED F1 0.8571).
- The frozen Phase 5F reranker (integrated in 1.2.0) is **unchanged** in this release — no retraining, no threshold change (remains `0.85`), no feature-configuration change.
- `LANG3` remains manual-only; no automatic-LANG3 behavior was added.
- No `.trenproj`/TXT/CSV schema changes.

### Known limitations

- UID remains weak and heterogeneous (low support across every evaluation source).
- NE precision remains limited by third-party Stanza NER behavior.
- Rare Turkish lexicon coverage causes UID predictions on otherwise-ordinary Turkish words.
- TR/EN dual-lexicon collision words are resolved the same way regardless of sentence context.
- Proper-name vs. common-word ambiguity (e.g. a Turkish common noun used as a given name) is not resolved.
- A pre-existing rule-based nominal-MIXED false-positive pattern can occur on bare English loanwords whose ending coincidentally matches a Turkish case suffix (observed on `office`/`remote`).
- Complex nominal MIXED chains (English stem + Turkish possessive + case suffix, e.g. `cloudumuza`) are outside the residual verbal detector's scope and may still be missed.

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