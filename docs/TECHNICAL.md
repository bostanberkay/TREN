# TREN Technical Documentation

This file collects the detailed technical material for TREN: the annotation pipeline, the MIXED-token reranker and its evaluation, file-format specifications, tool internals, and the formal description of the labeling logic. For installation and everyday use, see the [README](../README.md).

## Contents

- [Architecture](#architecture)
- [Production Status (v1.4.1)](#production-status-v141)
- [MIXED-Token Reranker](#mixed-token-reranker)
- [Input and Tokenization](#input-and-tokenization)
- [Multiple Data Sets](#multiple-data-sets)
- [Project Files (`.trenproj`)](#project-files-trenproj)
- [Export Formats](#export-formats)
- [Confidence Review Tool](#confidence-review-tool)
- [TDK Checker](#tdk-checker)
- [Word Frequency Computation](#word-frequency-computation)
- [Computational Design & Formalization](#computational-design--formalization)
- [Tests](#tests)

## Architecture

TREN's code is the `tren` package in `src/tren/` (started with `python -m tren`; `src/tren/__main__.py` is also the PyInstaller entry script):

| Module | Role |
|---|---|
| `cs_annotator_app.py` | Tkinter GUI (`App`) and its `main()`: UI, project save/load, grid editing, tools, export |
| `cs_pipeline.py` | `Annotator`: rule-based language ID (fastText + frequency lexicons), MIXED detection, Turkish suffix segmentation, NER (Stanza), Matrix/Embedded Language. No GUI imports |
| `reranking.py` | Post-processing stages run after `Annotator.annotate()`: the frozen MIXED-token reranker, the residual verbal MIXED detector, and the UID→TR resolver (single entry point `apply_reranker()`) |
| `confidence.py` | Read-only confidence/uncertainty scores used by the Confidence Review Tool; never changes a label |
| `annotation_model.py` | GUI-independent annotation-state helpers and the TXT/CoNLL/JSONL serializers |
| `tdk.py` | TDK dictionary lookup provider and the morphological parser used only by the TDK Checker |

Resources (`src/tren/resources/frequent_tr_words.txt`, `frequent_en_words.txt`, `lid.176.ftz`, `models/`) are package data, loaded at runtime from the `resources/` folder next to the modules, in a source checkout, an installed package, and the packaged apps alike. Scripts for rebuilding the reranker dataset and retraining it are in `tools/`.

## Production Status (v1.4.1)

### Pipeline

```
Input -> tokenizer -> rule-based annotator -> NE Policies C/D -> frozen Phase 5F reranker
      -> strict residual verbal MIXED detector -> UID->TR resolver -> Matrix/Embedded consistency -> output
```

### Active components

| Component | Role | Status |
|---|---|---|
| Rule-based annotator (`cs_pipeline.py`) | Primary language ID, MIXED/NE detection, Turkish suffix segmentation | Active; core labeling flow retained, with NE arbitration updated by Policies C/D |
| NE Policy C | Withholds NE status from `TIME`-subtype-only entity matches | Active |
| NE Policy D | Withholds NE status from guarded English-lexical/compound-entity matches | Active |
| Frozen Phase 5F reranker | Promotes eligible `UID`/`NE`/`TR` candidates to `MIXED` at threshold 0.85 | Active, frozen |
| Residual verbal MIXED detector | Promotes eligible `UID`/`TR` verbal candidates to `MIXED`, strict-lexicon evidence only | Active |
| UID→TR resolver (`reranking.py`) | Promotes eligible, still-`UID` tokens to `TR` using a conservative, multi-signal, explainable evidence model — after both stages above, `UID`→`TR` only | **Active**, behind `reranking.UID_TR_RESOLVER_ENABLED` (default `True`; set `False` to restore the exact pre-integration output) |

### Real-corpus metrics (primary evidence)

**Historical baseline** (measured at commit `5d644ae`, 2026-08-02, before the UID→TR resolver):

| Accuracy | Weighted F1 | Auto macro F1 | Lexical macro F1 | MIXED P/R/F1 |
|---:|---:|---:|---:|---|
| 0.8725 | 0.9041 | 0.6877 | 0.8698 | 0.8661 / 0.8899 / 0.8778 |

**Re-measured, 2026-08-09** (same unchanged pipeline code/corpus files — see the UID→TR resolver section below for why this differs from the row above by ~0.3pp; the exact original prediction file no longer exists, so this is a from-scratch, methodology-reconciled re-measurement, not a replacement of the historical number):

| Condition | Accuracy | Weighted F1 | Auto macro F1 | Lexical macro F1 | MIXED P/R/F1 |
|---|---:|---:|---:|---:|---|
| Without UID→TR resolver | 0.8751 | 0.9070 | 0.6916 | 0.8684 | 0.8655 / 0.8935 / 0.8793 |
| **With UID→TR resolver (current production)** | **0.8846** | **0.9121** | **0.6950** | 0.8684 (MIXED F1 unchanged) | 0.8655 / 0.8935 / 0.8793 (unchanged) |

### Synthetic benchmarks (secondary, diagnostic — not external validation)

| Benchmark | Condition | Accuracy | Weighted F1 | MIXED P/R/F1 |
|---|---|---:|---:|---|
| First synthetic (100 sent.) | Without UID→TR resolver | 0.8655 | 0.8894 | 0.7907 / 0.6800 / 0.7312 |
| First synthetic (100 sent.) | **With UID→TR resolver** | **0.8828** | **0.8987** | 0.7907 / 0.6800 / 0.7312 (unchanged) |
| Synthetic v2, adjudicated gold (100 sent., 622 tok.) | Without UID→TR resolver | 0.9212 | 0.9406 | 0.8000 / 0.9231 / 0.8571 |
| Synthetic v2, adjudicated gold (100 sent., 622 tok.) | **With UID→TR resolver** | **0.9341** | **0.9476** | 0.8000 / 0.9231 / 0.8571 (unchanged) |

Both benchmarks are synthetic and LLM-authored — diagnostic secondary evidence, not external validation. Benchmark v2's frozen original gold was preserved unchanged; three high-confidence gold corrections (`soundtrack` MIXED→EN, `dress` TR→EN, `code` TR→EN) live only in a separate adjudicated-gold file, never in the frozen original. (First-synthetic numbers above were re-measured together with the UID→TR resolver work after fixing a gold-alignment bug in the evaluation harness itself — see CHANGELOG.)

### UID→TR resolver

Integrated after offline validation across the real corpus and both synthetic benchmarks above: **zero harmful `UID`→`TR` changes and zero regression on any other label (`NE`/`EN`/`OTHER`/`LANG3`/`MIXED` all byte-identical) on every data source measured.** Scope is deliberately narrow — `UID → TR` only, never `UID → EN` — using an additive, explainable, multi-signal evidence model (trusted-lexicon match, valid Turkish suffix-chain analysis, strong fastText support, Turkish orthographic evidence, plus a capped sentence-level `MatrixLang` bonus that alone is never sufficient) behind a conservative promotion threshold, with hard exclusion gates for URLs/mentions/hashtags/emails/codes/numbers/punctuation/emoji/apostrophe-bearing tokens/acronyms/probable proper names/direct English matches, and — critically — for any token with an English-root-plus-Turkish-suffix analysis, so it can never pre-empt a MIXED promotion the way an earlier, reverted Turkish-stem fallback once did. Runs strictly after the residual verbal detector, on whatever is still `UID`, inside `reranking.apply_reranker()`. See `reranking.py`'s UID→TR resolver section (its module docstring) for the full design and `tests/test_uid_resolver.py` / `tests/test_reranker_integration.py` for its test coverage (46 + 15 tests).

### Production guarantees

- Frozen Phase 5F model and threshold (0.85) unchanged.
- Residual verbal detection runs after the reranker, strict-lexicon evidence only, and can only promote `UID`/`TR` to `MIXED` — it never touches `NE`/`EN`/`OTHER`/`LANG3`/an already-`MIXED` token.
- The UID→TR resolver runs after the residual verbal detector, only ever promotes `UID`→`TR`, and never touches `TR`/`EN`/`MIXED`/`NE`/`OTHER`/`LANG3`/`SentenceID`/`MatrixLang`/`EmbedLang` otherwise; a resolver exception for one token leaves that token unchanged and never interrupts the rest of the block; disabling `UID_TR_RESOLVER_ENABLED` restores the exact prior output.
- `LANG3` remains manual-only. No benchmark-specific rules are present anywhere in production.

### Known limitations

UID remains weak and heterogeneous; NE precision is limited by third-party Stanza behavior; rare Turkish lexicon coverage causes UID predictions; TR/EN dual-lexicon collisions are context-insensitive; proper-name/common-word ambiguity remains; nominal suffix substring false positives can occur on bare English words (e.g. `office`/`remote`); complex nominal MIXED chains (e.g. `cloudumuza`) may still be missed.

The frozen reranker can promote a proper name carrying a Turkish case suffix from `NE` to `MIXED`, although the reference annotation labels such tokens `NE` (e.g. `Almanya’ya`, `Türkiye'den`, `İran'dan`). Both of the reranker's harmful changes on its held-out test split were of this kind (`Store'da`, `Obscur'ün`: gold `NE` → `MIXED`). In spot checks with NER enabled, `İstanbul'a`, `Ankara'da`, `İzmir'den`, `Bursa'ya`, and `Antalya'ya` all ended up `MIXED`. Review suffixed proper names in the Confidence Review Tool or the grid. This behavior is unchanged pending a decision on the frozen model.

## MIXED-Token Reranker

TREN's annotation pipeline includes a statistical reranker that runs automatically after the rule-based annotator, on every annotation request, to catch MIXED intra-word code-switching calls the rule-based pass missed. `Annotator.annotate()` (`cs_pipeline.py`) remains the primary rule-based annotation engine — its core labeling flow is unchanged by the reranker below, though NE arbitration within it was subsequently updated by Policies C/D (see "Production Status" above); the reranker (`reranking.py`, which also holds the MIXED-candidate generation logic it calls into) is a post-processing stage that runs after it, in `cs_annotator_app.py`'s `run_pipeline()`:

```
Annotator.annotate() -> apply_reranker() -> _ensure_matrix_embed_consistency() -> _populate_table()
```

The reranker may only **promote** an already-produced `UID`, `NE`, or `TR` label to `MIXED` when its frozen model's probability clears the validated threshold — it never invents a label the rule-based pass didn't already consider, and it cannot alter `TR`/`EN`/`MIXED`/`OTHER`/`LANG3` labels otherwise. A token's final label may therefore come from either the rule-based annotator alone or from this reranking stage. It was developed under a strict experimental protocol before being integrated — every candidate feature group ("batch") was benchmarked in isolation against a frozen train/dev/test split before being considered for adoption, and negative results were recorded, not discarded (see below).

### Architecture

The reranker is a hybrid, two-stage pipeline built on top of the existing rule-based annotator:

1. **Candidate generation** — for every token the rule-based pass predicted as `UID`, `NE`, or `TR`, the reranker enumerates plausible stem/suffix splits by calling the existing, unmodified `Annotator._parse_tr_suffixes_full` (plus a separate, additional experimental verbal-suffix table used only as a fallback) and selects the best split with a deterministic tie-break policy.
2. **Statistical reranking** — a `LogisticRegression` classifier, trained on character n-grams of the token plus a set of structured features described below, estimates the probability that the token is genuinely MIXED. If a candidate's probability clears a data-driven threshold (selected on a held-out dev set to guarantee ≥0.75 precision), the reranker promotes the label to MIXED; otherwise the original prediction is kept unchanged.

The structured features are organized into independently-gated, opt-in **batches**, each isolated behind its own flag so it can be included or excluded without touching any other batch's code.

### Loading and fallback behavior

The trained model (`src/tren/resources/models/model.joblib`, `vectorizer.joblib`, `metadata.json` — a byte-for-byte copy of the frozen Phase 5F experiment artifacts) is loaded lazily: nothing is loaded at application startup, only on the first annotation request, and the result is cached for the rest of the session. If `joblib`/`scikit-learn` aren't installed, the model files are missing or corrupted, or the metadata fails validation against the frozen Phase 5F configuration, loading fails safely and annotation falls back to exactly the original rule-based output — no crash, no dialog, no interruption. The same applies when the installed scikit-learn can load the model file but not run it (scikit-learn 1.7 and older): the loader runs one probe prediction and rejects the model if it fails. After the first annotation request, the toolbar shows **MIXED reranker: active** or **MIXED reranker: unavailable (rule-based only)**; hovering over the unavailable status shows the reason. Saved projects (`.trenproj`) and the TXT/CSV export formats are unaffected either way: the reranker's output uses the identical text format `Annotator.annotate()` itself produces.

### Frozen production configuration: Phase 5F

The current active baseline combines the following, on top of the shared character n-gram + baseline structured features:

| Batch | Description | Status |
|---|---|---|
| **Batch A** | Parser metadata — which parse path produced the candidate analysis, why the token was flagged as a candidate, and where the stem/suffix split falls in the token. | **Active** |
| **Batch C** | Confidence interaction — how the token's own language-ID confidence compares to its extracted stem's, and how strongly lexicon/fastText evidence agrees. | **Active** |
| **Batch G** (pruned) | Candidate ambiguity — how many plausible analyses were considered, whether the selection was unique, and whether nominal and verbal parses competed for the same token. | **Active** |
| Batch B | Morphological complexity — tag counts, case/plural/possessive/derivational/verbal flags derived from the parser's own morphological tags. | Evaluated, **rejected** |
| Batch D | English-stem quality — fastText/lexicon-derived confidence and contrast measures for the extracted stem. | Evaluated, **rejected** |

Batches B and D are **not hidden** — their code, CLI flags, and tests remain in the repository and fully reproducible; they are simply off by default because they did not clear the bar for inclusion:

- **Batch B (morphological complexity) was rejected** because it produced no improvement in active-policy cascade performance over the Batch A+C baseline, and its own coefficients showed a sign-flip on the pre-existing `suffix_segment_count` feature — evidence that it duplicates information the model already had rather than adding new signal.
- **Batch D (English-stem quality) was rejected** because, at the operating threshold actually used in production-realistic evaluation, it introduced two new *neutral* misclassifications (tokens that were already mislabeled and remained mislabeled, just differently) and a net regression in MIXED F1 relative to the Phase 5F baseline, despite having individually large model coefficients — which the evaluation protocol explicitly treats as insufficient grounds for adoption on its own.

### Benchmark (Phase 5F, frozen production configuration)

Evaluated on a frozen, held-out test split via a candidate-gated cascade simulation (non-candidate tokens are always left unchanged; candidates flip to MIXED only if the reranker's probability clears the dev-selected precision≥0.75 threshold):

| Metric | Value |
|---|---|
| MIXED precision | 0.893 |
| MIXED recall | 0.781 |
| MIXED F1 | 0.8333 |
| Beneficial changes (fixed a real MIXED miss) | 17 |
| Harmful changes (broke a previously-correct prediction) | 2 |
| Neutral changes (already wrong, still wrong) | 0 |

These numbers are the reranker's held-out benchmark results — a fixed measurement against one frozen test split, not a live guarantee about every future document. This is the same frozen model (threshold 0.85, Batch A + Batch C + pruned Batch G) that now runs automatically as part of normal annotation, per "Loading and fallback behavior" above.

### Reproducing / testing

The reranker's own automated tests (`tests/test_mixed_reranker.py`, `tests/test_reranker_integration.py`) are included in the project's main test suite; run the whole repository's suite with `python -m pytest`. To rebuild the active-baseline dataset and retrain the model yourself:

```bash
python tools/build_reranker_dataset.py --gold <gold.csv> --pred <pred.csv> \
  --exclusions <exclusions.csv> --segmentation-mismatches <segmentation_mismatches.csv> \
  --resources-dir src/tren/resources --out-dir <out-dir> \
  --include-batch-a-features --include-batch-g-features

python tools/train_mixed_reranker.py --dataset <out-dir>/dataset.json \
  --split-manifest <out-dir>/split_manifest.json --out-dir <out-dir>
```

Batch C is included by default; pass `--exclude-batch-c-features` to disable it. Batch B and Batch D remain available via `--include-batch-b-features` / `--include-batch-d-features` for reproducing the rejected experiments, but are not part of the active baseline shown above.

## Input and Tokenization

**Sentence units and tokenization.** The annotation pipeline treats **each non-empty line** of the input as one sentence (one `SentenceID` block); it does not split sentences at `.`, `?`, or `!`, so put one sentence per line if you want sentence-level Matrix/Embedded Language values. Blank lines only separate blocks. Punctuation marks such as `.`, `,`, `!`, `?`, and `:)` are **not emitted as tokens**; they are dropped during tokenization, so they never appear in the grid or in exports. Numbers (`100`, `3.5`), URLs, `@mentions`, `#hashtags`, emoji, and a stand-alone apostrophe are kept as tokens and labeled `OTHER`. An apostrophe inside a word is kept (`meeting'e` is one token).

The **Label** column of a token row only accepts the seven schema labels (`TR`, `EN`, `MIXED`, `UID`, `NE`, `LANG3`, `OTHER`), typed in any letter case (`mixed` is stored as `MIXED`), or an empty value (the same as Clear). Any other typed or pasted value is rejected and the previous label is kept. Meta rows keep their own values: the `SentenceID` row's Label cell holds the sentence number, and `MatrixLang`/`EmbedLang` rows accept only `TR` or `EN` (a `-` written by the pipeline for "no embedded language" is preserved).

**Running the pipeline again** on a dataset that already has annotations replaces all of them (labels, glosses, extra-column values, and reviewed marks) and cannot be undone, so **Run** asks for confirmation first; Cancel leaves the current annotations untouched. To keep them and annotate the text again, use **Add New Data ▸ Re-run Current Text**, which creates a separate dataset.

## Multiple Data Sets

Each dataset has its own source text, annotation rows, labels/glosses, and Matrix/Embedded Language values. Editing one dataset never changes another. Only one dataset is displayed at a time; switching tabs swaps the grid and input panel to that dataset's content without re-running the annotation pipeline or reloading any NLP model.

Each dataset also maintains its own Merge Cells / Confidence Review Tool undo history while the project stays open in the current session. This undo history is **not** saved to `.trenproj` and is not restored across an application restart — reopening a project always starts every dataset with empty undo history, even though its annotation data is fully restored.

### Add New Data

Clicking **+** opens an **Add New Data** dialog with a name field (defaulting to `Data 2`, `Data 3`, ...) and three choices:

- **Open New File** — pick a `.txt` file (or, via "All Files", any text file) through the native file chooser. The file is read as strict UTF-8 (a leading byte-order mark, as written by older Windows Notepad, is dropped); its complete contents become the new dataset's source text, and the annotation pipeline runs on it only after you press Create. If the Name field still holds its untouched automatic `Data N` default at the moment you pick a file, it's pre-filled with the file's stem — typing your own name, or a name already set from a previous pick, is never overwritten. Only the file's **name** (e.g. `corpus.txt`, never its full local path) is kept as optional, display-only metadata on the dataset — a full path could expose your account name and local folder structure if the project is later shared, and reopening the project never depends on the original file still existing anyway (the file's full text is what's actually saved in `.trenproj`).
- **Enter New Text** — paste or type new text in the dialog; the production annotation pipeline runs on it only when you confirm, and the result becomes a new dataset. The currently active dataset is left untouched.
- **Re-run Current Text** — re-runs the pipeline on the *active* dataset's own source text and stores the result as a new, independent dataset, useful for comparing an alternative annotation pass against the original.

Only the controls belonging to the currently selected mode are enabled; switching between modes preserves whatever text you've typed and whichever file you've selected for as long as the dialog stays open. If the name is blank, no file is selected (in Open New File mode), the resulting text is empty, or annotation fails, no tab is created and the active dataset, its table, and its undo history are all left exactly as they were — selecting a file by itself never marks the project as having unsaved changes; only successfully creating the dataset does.

Switching datasets automatically closes dataset-scoped tool windows (Confidence Review Tool, Auto-Glossing Tool, Concordance, Word Frequency List, Full Edit Window, Show Sentence) so none of them can be left editing the wrong dataset by a stale row reference.

## Project Files (`.trenproj`)

Project files store every dataset (name, source text, annotation blocks, and Matrix/Embedded Language values), which dataset was active, configuration settings, and cursor/selection position. They do **not** store Merge Cells / Confidence Review Tool undo history, which is session-only (see [Multiple Data Sets](#multiple-data-sets)).

New Project, Open Project Save, and closing the application (the window's close button, File ▸ Exit, or on macOS Quit / ⌘Q) all share the same unsaved-changes check: if nothing has changed since the project was last saved (or opened/created), the action proceeds immediately with no prompt. Otherwise you're asked to Save, Discard, or Cancel — Save only proceeds once the save has actually completed, Discard proceeds without saving, and Cancel leaves the current project untouched. Opening a project defers this check until after the selected file has been fully read and validated, so cancelling the file chooser or picking an invalid file never touches your current work.

`.trenproj` files carry a schema version, and which version a file declares strictly determines how it must be shaped — the loader never guesses the shape from whichever keys happen to be present. Version 2 is the current format and requires a non-empty `"datasets"` list. Version 1 is the older, single-dataset format from before this feature and requires the legacy `"blocks"`/`"input_text"`/`"extra_headers"` top-level shape; a version 1 file that also contains a `"datasets"` key (for example a hand-edited or half-migrated file) is rejected as malformed rather than silently accepted. A file with no `"version"` key at all predates versioning and is treated as version 1, so the same requirement applies to it. A version 1 project still opens correctly and appears as a single dataset named `Data 1`; it is not rewritten on disk (and stays version 1) until you explicitly save it again, at which point it is written as version 2. An unrecognized future version is likewise rejected with a clear error rather than being misread.

## Export Formats

Choose which dataset to export (defaulting to the currently active one) and a format — **TXT**, **CSV**, **TREN CoNLL-style**, or **JSONL** — then choose a destination file. Exactly one dataset is written per export; datasets are never merged into a single file. The suggested filename is derived from the dataset's name (sanitized for the filesystem). Cancelling at any point creates no file, and exporting never modifies the annotation data or re-runs the annotation pipeline. TXT, CoNLL and JSONL files (and `.trenproj` saves) always use `\n` line endings, so the same data exports to byte-identical files on Windows and macOS; CSV rows end with the `csv` module's `\r\n` on every platform.

### TXT and CSV

Unchanged from previous versions. Exporting the active dataset reflects exactly what the grid currently displays; exporting any other dataset is built from its stored annotation data.

**TXT** (`.txt`): UTF-8 plain text, one grid row per line, fields separated by a tab, in grid column order: `Token` (running index), `Item`, `Label`, `Gloss`, then any user-added columns. A blank line separates sentence blocks.

- Token rows start with the running index: `3	boss'um	MIXED	boss-POSS.1SG`.
- Meta rows have no index, so their line starts with the row name: `SentenceID	1`, `MatrixLang	TR`, `EmbedLang	EN`.
- Trailing empty fields are omitted, so a token row without a gloss is `2	amazing	EN`. The number of fields per line therefore varies; recognize meta rows by their first field.

```
SentenceID	1
1	kitap	TR
2	amazing	EN
3	boss'um	MIXED	boss-POSS.1SG
MatrixLang	TR
EmbedLang	EN

SentenceID	2
4	bugün	TR
...
```

**CSV** (`.csv`): UTF-8, comma-separated with standard quoting, a header row with the grid's column names (`Token,Item,Label,Gloss,...`), then one row per grid row with every column present (blank separator rows included as empty rows).

TXT and CSV are export-only: **Open Input** always treats a file as raw text to annotate, so an exported TXT cannot be loaded back as annotations. Use a project save (`.trenproj`) to resume work.

### TREN CoNLL-style (`.conll`)

A deterministic, TREN-specific export format — **not** CoNLL-U and not compatible with any particular external shared-task format. A file-level header, then one comment block plus one tab-separated line per token for each sentence, with token indices restarting at `1` in every sentence:

```
# TREN CoNLL export
# columns = TokenIndex Token Label Gloss

# sent_id = 1
# matrix_lang = TR
# embedded_lang = EN
1	Bugün	TR	_
2	meetinge	MIXED	meeting-DAT
3	katıldım	TR	_
```

- `sent_id` comes from the sentence's `SentenceID` row when present, otherwise its 1-based position in the dataset.
- `matrix_lang` / `embedded_lang` comment lines are included only when the corresponding `MatrixLang`/`EmbedLang` row is present (never fabricated).
- An empty gloss is written as `_`. A stray tab or newline inside a token/label/gloss is replaced with a space so it can never split or extend a line.
- Sentences are separated by a single blank line; `SentenceID`/`MatrixLang`/`EmbedLang` are never emitted as token lines. The file is valid UTF-8 with Unicode preserved exactly.

### JSONL (`.jsonl`)

One JSON object per sentence, one physical line per object, produced with Python's standard `json` module (`ensure_ascii=False`, so Unicode is written directly rather than escaped):

```json
{"sentence_id": "1", "dataset": "Data 1", "source_text": "Bugün meetinge katıldım", "matrix_lang": "TR", "embedded_lang": "EN", "tokens": [{"index": 1, "token": "Bugün", "label": "TR", "gloss": ""}, {"index": 2, "token": "meetinge", "label": "MIXED", "gloss": "meeting-DAT"}, {"index": 3, "token": "katıldım", "label": "TR", "gloss": ""}]}
```

- `source_text` is the sentence's token sequence joined with spaces (reflecting current annotation state, not necessarily the original raw input line).
- `matrix_lang`/`embedded_lang` are empty strings, not omitted, when their meta row is absent.
- Empty glosses are empty strings (`""`), not `_`. `SentenceID`/`MatrixLang`/`EmbedLang` rows are excluded from `tokens`.
- Token indices also restart at `1` per sentence, matching the CoNLL-style export.

Both `blocks_to_conll` and `blocks_to_jsonl` (in `annotation_model.py`) are pure functions with no GUI or pipeline dependency, and are deterministic for identical input.

## Confidence Review Tool

The **Confidence Review Tool** (`Tools → Confidence Review Tool`) is a focused window for sequentially reviewing and correcting tokens the pipeline is uncertain about. It is not limited to **UID** — every automatically-annotated token gets a deterministic confidence score, band (**HIGH** / **MEDIUM** / **LOW**), and set of evidence/uncertainty reasons (see `confidence.py`), and the tool can list tokens for any of the 7 labels, filtered to any combination of confidence bands.

**The tool opens with "All Uncertain" selected by default**: every token, across all 7 labels, whose confidence record is flagged for review (its `review_recommended` flag — see `confidence.is_review_required`) — never decided by label name. A confidently-labeled token never appears in this default view no matter its label, and an uncertain token always appears no matter its label.

### View

A **View** combobox above the label/confidence filters selects between three presets:

- **All Uncertain** (default): every token, any label, currently flagged for review by its confidence score.
- **UID Only**: the tool's original view — every token currently labeled **UID**, regardless of confidence.
- **Custom**: whatever the **Labels**/**Confidence** checkboxes below currently say; selected automatically the moment either checkbox row is touched directly.

### Scope and Data Selection

- **Labels**: checkboxes for all 7 labels (TR/EN/MIXED/UID/NE/OTHER/LANG3) choose which currently-labeled tokens appear in the list when the view is **UID Only** or **Custom**.
- **Confidence**: checkboxes for High/Medium/Low further restrict the list to those confidence bands (also only under **UID Only**/**Custom** — **All Uncertain** already filters by the review flag directly).
- **Hide reviewed**: excludes tokens already marked reviewed (see Synchronization below) from the list, under any view.
- Each item shows the full sentence context surrounding the token, and the list itself shows each token's current label, confidence band/score, and reviewed status.

### Interface Elements

- **Search**: filter the list by typing part of a token; press Return or click **Search**.
- **Find All Occurrences**: locate every occurrence of the selected (or searched) token, regardless of its current label.
- **Evidence**: shows the selected token's current label, confidence score/band, uncertainty reasons, and the relevant pipeline evidence behind them (lexicon/fastText/morphology signals, frozen-reranker or UID→TR-resolver evidence, MatrixLang/EmbedLang consistency, etc., as applicable).
- **Label** and **Gloss** fields: edit the current token's label and gloss.
- **First / Previous / Next / Last**: move through the list.
- **Apply**: commit the edited label and gloss.
- **Undo**: reverse the most recent Apply.

### Synchronization

- Applying an edit updates the shared annotation model and the main annotation grid immediately, and marks the token reviewed.
- `MatrixLang`/`EmbedLang` for the affected sentence are recomputed automatically after each applied label change.
- A token edited so it no longer matches the active view/filter (e.g. no longer flagged uncertain, or given a label outside the active filter) drops out of the list, and the tool advances to the next remaining item.
- Switching datasets closes the tool; reopening it for the newly active dataset resets to the **All Uncertain** default and never shows another dataset's tokens.

The Confidence Review Tool works entirely offline: opening it does not invoke Stanza, fastText, the MIXED-token reranker, or any external AI service. The confidence score itself is a deterministic, rule-based estimate — it is **not** a statistically calibrated probability (see `confidence.py`'s module docstring).

## TDK Checker

The **TDK Checker** (`Tools → TDK Checker`) is a **separate tool from the Confidence Review Tool** — it does not replace or rename it. It looks up a token, a hierarchically-parsed root/lemma, and each proposed suffix segment against the Turkish Language Association's (TDK) online dictionary, to help a reviewer judge whether a token is a lexicalized Turkish word, a code-switched form, or something else. **TDK membership is evidence of Turkish lexicalization, not a language-ID verdict**: a TDK match never forces a token to `TR`, a missing match never forces `EN`/`MIXED`, and a match never automatically removes an existing `MIXED` label. This tool never reads or writes a token's **label** or **gloss** at all — Gloss is handled entirely by the main table and the Auto-Glossing Tool; this tool only ever writes a `tdk_segmentation` correction, exactly the same restraint the Confidence Review Tool's Apply applies to `label`/`gloss`/`confidence`.

### Opening it

- **From the main table**: select a normal token row and choose **Open in TDK Checker** from the grid's right-click menu (or `Tools → TDK Checker` afterward). The action is disabled/shows a warning for a `MatrixLang`/`EmbedLang`/separator row, and shows a message if nothing (or nothing valid) is selected; with several rows selected, it uses the first valid token row. Opening it this way populates the token, sentence context, sentence ID, and token index automatically, runs the morphological parser automatically, and — because clicking it is an explicit request — immediately starts the TDK lookup.
- **Standalone**: `Tools → TDK Checker` with nothing selected opens an empty checker where you can type or paste any term yourself; it never looks anything up until you click **Check TDK**.
- Switching datasets always closes the TDK Checker (it is not dataset-aware, exactly like the Confidence Review Tool and every other row-index-based tool) — it can never end up silently editing a token in a dataset that is no longer active.

### Morphological parser

The parser performs hierarchical Turkish morphological analysis rather than flat, character-by-character suffix stripping: it normalizes the token, searches for the longest genuinely valid root (never an arbitrary leftover character), applies noun suffixes in the correct order (`root → derivational → plural → possessive → case`, e.g. `film + ler + imiz + den`) or verb suffixes in the correct order (`root → derivational → tense/aspect/mood → person/agreement`, e.g. `sür + dü`, never `sürd + ü`), and rejects any candidate whose remaining suffix sequence isn't a recognized Turkish morpheme. A frequency-attested inflected surface form (e.g. "geldi" itself appears in the corpus) is never confused with a genuine base root ("kitaplar" never wins over "kitap" + "lar" just because it happens to also be independently attested) — see `tdk.py`'s morphological-parser section (its module docstring) for the full scoring policy. Every automatic candidate is classified as a **full Turkish lexical item**, an **English root + Turkish suffix** candidate, an **ambiguous candidate** (competing readings too close to call from corpus evidence alone — e.g. "kalem" could be the base noun "pen" or "kale" [fortress] + 1sg possessive), or an **invalid parser proposal**. The parser never assigns a language label from this analysis alone.

### Interface

- **Token** / **Sentence ID** / **Token Index** / **Sentence** (context): identify exactly which row is loaded.
- **Root/Lemma** and **Segments** (freely editable): the parser's proposal (e.g. `cloud` + `umuz + a` for `cloudumuza`), which you can correct by hand — segments accept `+`, `-`, or spaces as separators. **Re-parse** re-derives them from the token from scratch; typing into these fields on its own never triggers a re-parse or a lookup, and never overwrites a manual correction unless you click Re-parse yourself.
- **Explanation**: a read-only breakdown of how the current root/segments were derived — `Token` / `Root` / one `Suffix` + `Analysis` line per segment / overall `Status` (the category above) — so you can see exactly why the parser chose that split.
- **Check TDK**: looks up the full token, the **current** root/lemma, and **every current** segment — always whatever is in the fields right now, never a stale parser value. Each is sent to TDK **individually**, never the surrounding sentence.
- **Results table**: one row per term checked (full token / root / each segment), each with its own **TDK Status** (`FOUND`, `NOT_FOUND`, `UNAVAILABLE`, `NETWORK_ERROR`, or `STALE_RESULT`) and a short detail. Editing Root/Lemma or Segments after a check immediately marks every row `STALE_RESULT` (the previous query's answer, kept visible but clearly labeled) until you press **Check TDK** again — an in-flight lookup whose fields changed before it returned is marked stale the same way, and a superseded lookup can never overwrite a newer one.
- **Dictionary Detail**: selecting a row in the results table shows the **full** TDK entry for that term, not just found/not-found — headword, part of speech, every sense's definition/usage labels/examples, origin/etymology, pronunciation, compounds, idioms, proverbs, source, and the exact query sent. Any field TDK didn't provide reads **"Not provided"**, never a guessed value.
- **Status**: the overall/most recent lookup status and source.
- **Apply Correction**, **Undo**, **Find All Occurrences**, **Close**.

### Network behavior and privacy

- The TDK lookup is the **only** network-dependent part of TREN, and it is **fully opt-in per action**: it is never called at startup, never during normal annotation, and never automatically for every token. It runs **only** when you click **Check TDK** or **Open in TDK Checker** on a selected token.
- Only the specific token/root/segment being checked is ever sent — **never the sentence**, never surrounding context.
- The request runs off the Tk main thread with a short timeout, so the rest of the application never freezes while waiting on it; if you change the token and start a new lookup before an older one finishes, the older (stale) response is discarded rather than overwriting the newer one.
- Results are cached in memory for the session, keyed by the normalized query and dictionary source, so re-checking the same term doesn't repeat the request.
- A connection failure, timeout, malformed response, HTTP error, or an unrecognized/changed response shape is reported as `UNAVAILABLE` or `NETWORK_ERROR` — TREN never fabricates a `FOUND` result it can't actually confirm, and never crashes because of it. A `NOT_FOUND` result always shows TREN's own fixed English message, never the raw Turkish text TDK's own endpoint returns.
- The request is HTTPS with certificate and host-name verification always on. On Windows, TREN trusts the certificates in the Windows certificate store plus the Mozilla CA bundle from `certifi`: Python reads only the roots already present in the Windows store, and Windows adds missing roots on demand, so a fresh Windows machine can lack the ISRG root that `sozluk.gov.tr`'s Let's Encrypt certificate chains to. On macOS and Linux, Python's default trust store is used, as before. The full error behind a `NETWORK_ERROR` is written to standard error (`~/.cs_annotator/tren.log` in the packaged Windows app).
- The rest of TREN — annotation, the Confidence Review Tool, export, project save/load — remains fully usable completely offline; the TDK Checker is the one feature that needs the internet, and only when you explicitly ask it to check something.
- TDK does not publish an official, documented public API; this feature uses the widely-used but undocumented `sozluk.gov.tr` lookup endpoint on a best-effort basis (see `tdk.py`'s dictionary-provider section) and treats any unexpected response shape as a provider failure (`UNAVAILABLE`), never as evidence one way or the other.

## Word Frequency Computation

• Tokens are normalized before counting to ensure consistent frequency estimation.  
• Meta rows and non-token elements are excluded from frequency calculations.  
• Frequencies are aggregated across the entire input text.

```bash
Token filtering
T' = { t ∈ T | t is not a meta token }

Token normalization
norm(t) = strip_punctuation(lowercase(t))

Total frequency
f(w) = Σ I(norm(t_i) = w)

Label-conditioned frequency
f(w | L) = Σ I(norm(t_i) = w ∧ label_i = L)

Total token count
N = Σ_w f(w)

Sorting criterion
sort by: (-f(w), w)
```

## Computational Design & Formalization

This section formalizes the **language labeling and annotation mechanisms** implemented in TREN. The formulations below describe the principles underlying the system’s decisions. TREN implements a **symbolically constrained, probabilistic, and morphologically informed** annotation framework. Language labels emerge from hybrid decision mechanisms rather than purely statistical or purely rule-based processes, ensuring transparency, interpretability, and linguistic validity.

**Note on this formalization:** Sections 2–4 below describe an idealized model of TREN's language-identification logic, intended to communicate the underlying linguistic principles. They do not describe the literal control flow of the implementation. In the actual pipeline, a label is produced by a staged, priority-ordered sequence of checks — lexicon/frequency-list membership is consulted first, and a statistical language-model confidence score is only consulted as a last resort — rather than by evaluating a single function over a normalized two-way probability distribution.

---

### 1. Token Space and Label Set

Let

T = { t₁, t₂, …, tₙ }

be the ordered set of tokens extracted from the input text.

Each token tᵢ is assigned a label ℓᵢ from the finite label set:

L = { TR, EN, MIXED, UID, NE, OTHER, LANG3 }

Meta tokens used for sentence- or block-level information are excluded from token-level labeling:

T_meta = { SentenceID, MatrixLang, EmbedLang }

T_valid = T \ T_meta

---

### 2. Probabilistic Word-Level Language Identification

For each token t ∈ T_valid, the language identification model produces a probability distribution:

P(t) = { P_TR(t), P_EN(t) }

with the constraint:

P_TR(t) + P_EN(t) = 1

where:
• P_TR(t) denotes the probability that token t is Turkish  
• P_EN(t) denotes the probability that token t is English  

---

### 3. Lexicon Membership Constraints

Let Lex_TR and Lex_EN denote the Turkish and English lexicons, respectively.

Lexicon membership is defined as:

t ∈ Lex_TR  ⇔  t is attested in the Turkish lexicon  
t ∈ Lex_EN  ⇔  t is attested in the English lexicon  

Lexicons function as symbolic constraints in the labeling decision.

---

### 4. Hybrid Language Labeling Function

Final language labels are assigned using a **hybrid decision function** combining probabilistic confidence and lexicon membership:

label(t) =

• TR  
  if P_TR(t) ≥ θ ∧ t ∈ Lex_TR  

• EN  
  if P_EN(t) ≥ θ ∧ t ∈ Lex_EN  

• MIXED  
  if EN_stem(t) ∧ TR_suffix(t)  

• UID  
  otherwise  

where:
• θ is a confidence threshold  
• EN_stem(t) denotes the presence of an English lexical stem  
• TR_suffix(t) denotes one or more Turkish morphological suffixes  

---

### 5. Intra-Word Code-Switching (MIXED) Condition

A token t is labeled as MIXED if it satisfies the following structural condition:

t = s + σ₁ + σ₂ + … + σₖ

such that:

s ∈ Lex_EN  
σᵢ ∈ Morph_TR  for all i ≥ 1  

where Morph_TR is the set of licensed Turkish morphological suffixes.

This captures English–Turkish intra-word code-switching.

---

### 6. Morphological Validity Constraint

Suffix sequences must conform to Turkish morphotactic constraints:

{ σ₁, σ₂, …, σₖ } ⊆ Σ_TR

where Σ_TR is the inventory of morphologically valid Turkish suffixes.

Tokens violating morphotactic constraints are excluded from MIXED labeling.

---

### 7. Named Entity Precedence

If a token t is identified as a named entity:

NE(t) = true

then its label is overridden as:

label(t) = NE

Named Entity recognition takes precedence over language-based labeling.

---

### 8. Residual Category Assignment

Tokens that do not meet linguistic labeling criteria are assigned to residual categories:

label(t) = OTHER  
  if t is a number, punctuation mark, symbol, or non-lexical item  

**Note:** the tokenizer does not emit most punctuation marks as tokens (they are dropped before labeling), so in practice `OTHER` is assigned to numbers, URLs, mentions, hashtags, emoji, and stand-alone apostrophes. See [Input and Tokenization](#input-and-tokenization).

label(t) = UID  
  if t cannot be confidently identified as Turkish or English, including
  tokens belonging to a language other than Turkish or English  

**Note:** `LANG3` is available as a manual relabeling option in the
annotation grid, allowing a human annotator to mark a token as belonging to
a third language after review. It is not currently assigned automatically
by the pipeline; unidentified non-TR/EN tokens fall through to `UID` unless
a human annotator relabels them.

---

### 9. Sentence-Level Matrix Language

For a sentence S consisting of tokens { t₁, …, tₘ }, define:

count_L(S) = | { t ∈ S : label(t) = L } |

for L ∈ { TR, EN, MIXED }.

MIXED tokens contribute weighted partial votes to both languages, reflecting
that a MIXED token combines a Turkish suffix with an English stem:

score_TR(S) = count_TR(S) + w_TR · count_MIXED(S)  
score_EN(S) = count_EN(S) + w_EN · count_MIXED(S)

with w_TR = 0.6 and w_EN = 0.4 by default. NE, OTHER, and UID tokens do not
contribute to either score.

The Matrix Language is defined as:

ML(S) = TR  if score_TR(S) ≥ score_EN(S)  
      = EN  otherwise

(ties are resolved in favor of TR)

---

### 10. Embedded Language

The Embedded Language is defined as the non-matrix language, present if at
least one token in S is labeled as that language or as MIXED:

EL(S) = the language in { TR, EN } \ { ML(S) } that occurs (directly or via
a MIXED token) in S

If no such token occurs, EL(S) is undefined, rendered as "-" in the
annotation grid.

---

## Tests

An automated test suite in `tests/` covers `annotation_model.py`, `cs_pipeline.py`, `reranking.py`, `tdk.py`, and `confidence.py`, plus GUI regression tests for the main window, dataset tabs, the Confidence Review Tool, and the TDK Checker. GitHub Actions (`.github/workflows/ci.yml`) runs it on every push, with GUI tests under a virtual X display. Run it locally with:

```bash
python -m pytest
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup.
