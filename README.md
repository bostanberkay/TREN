## TREN: A Corpus Annotation Tool for Code-Switching Data

TREN is a system developed for the annotation and analysis of Turkish–English code-switching data in corpus-based linguistic research. It is designed as a semi-automatic annotation application that integrates automatic processing with user-controlled manual intervention, enabling transparent and reproducible analysis of code-switching phenomena, with a particular focus on intra-word code-switching.

The system consists of an interactive graphical annotation interface and an underlying processing pipeline. Through the interface, users can load raw textual data, preprocess it into token-based representations, and inspect or revise automatically assigned labels. The processing pipeline supports language identification, rule-based morphological analysis, and sentence-level computations, facilitating fine-grained analysis of bilingual data.

TREN’s features include, for instance:

• semi-automatic token-level language identification for Turkish and English  
• support for the annotation of intra-word code-switching structures  
• rule-based detection of Turkish morphemes attached to English stems  
• morphological glossing based on the Leipzig Glossing Rules  
• interactive manual correction and user supervision of automatic labels  
• concordance (KWIC) and frequency-based inspection tools  
• automatic computation of Matrix Language and Embedded Language at the sentence level  
• flexible export of annotated data in `.csv` and `.txt` formats  

TREN is intended for use by researchers working on bilingual and multilingual language data, particularly in contexts where fine-grained annotation of code-switching is required. Typical application areas include:
• code-switching and bilingual language use  
• corpus linguistics  
• morphology and morphosyntax   
• discourse-oriented and usage-based linguistic analysis
<p align="center">
  <img src="assets/tren_icon.png" alt="TREN icon" width="120">
</p>

## Installation

TREN is currently available for macOS. Support for other operating systems will be added in future releases.

### macOS

<ol>
 <li>
  Download the <code>.dmg</code> file from the 
  <a href="https://github.com/bostanberkay/TREN/releases" target="_blank">
    GitHub Releases page
  </a>.
</li>
  <li>Open the DMG and drag the <strong>TREN</strong> application into the <strong>Applications</strong> folder.</li>
  <li>Launch the application from the Applications folder.</li>
</ol>

If you encounter a security warning on first launch:

<ul>
  <li>Right-click (or Ctrl-click) the <strong>TREN</strong> app and select <strong>Open</strong>.</li>
  <li>Confirm the prompt from macOS Gatekeeper.</li>
</ul>

## Run from Source (Python)

Alternatively, run the application directly from source or you may clone this repository.

```bash
git clone https://github.com/bostanberkay/TREN.git
cd tren
python cs_annotator_app.py

```
### Requirements

If you choose to run the application from source, you will need **Python 3.9 or higher** and the following Python packages:

- fasttext  
- stanza  
- tksheet  

All required dependencies are listed in the `requirements.txt` file. To install them automatically, run:

```bash
pip install -r requirements.txt
```
## Version Log

### TREN v1.0.0 Initial Release
• First public release of the TREN application.  
• Stabilized application initialization and runtime behavior.  
• Improved handling of resource paths for both packaged and source-based execution.  
• Ensured consistent behavior between standalone `.app` execution and running from source.  
• Verified macOS `.dmg` distribution and application packaging workflow.  

### Bug Fixes
• Resolved a critical launch issue causing the application to open twice on startup.  
• Improved robustness of file and resource path resolution.



# Documentation of TREN

TREN is a semi-automatic corpus annotation application developed for the analysis of Turkish–English code-switching data, with a particular focus on intra-word code-switching and morphologically complex structures. The application combines automatic processing pipelines with interactive user supervision, allowing researchers to inspect, correct, and enrich annotations in a controlled environment.

This documentation provides a detailed overview of the TREN interface, workflow, and underlying mechanisms. Its purpose is to guide users through the functionalities of the application, explain how annotation decisions are produced and modified, and clarify how analytical tools and computations operate within the system. The documentation is intended both for end users conducting linguistic analysis and for researchers interested in the technical design of the application.

---

## Main Window

![Main window](images/main.png)

The main window serves as the central workspace of the TREN application. It integrates text input, annotation output, and interactive controls into a single interface, enabling a smooth transition between automatic processing and manual inspection.

### Layout Overview

The interface is divided into three primary regions:

• Input Panel (left)  
• Annotation Grid (right)  
• Control and Relabel Panel (bottom-right)  

Each region is designed to support a specific stage of the annotation workflow.

---

### Input Panel

The input panel is a free-text editor where users load or paste raw textual data containing code-switching phenomena. This panel represents the unprocessed input and remains editable throughout the annotation process.

Key characteristics:

• Displays the original text exactly as provided by the user.  
• Serves as the source for tokenization and annotation.  
• Supports direct text search and concordance highlighting.  
• Is synchronized with auxiliary tools such as KWIC and sentence context views.  

---

### Annotation Grid

The annotation grid presents the processed output in a table-based format, where each row corresponds to a token-level unit derived from the input text.

Default columns include:

• Token: the unique number attached to items.
• Item: the surface form as it appears in the input. 
• Label: a language or category label (e.g., TR, EN, MIXED, UID).  
• Gloss: a morphological or explanatory gloss, when applicable. Automatic-glossing tool will be introduced in next sections. 

The grid supports:

• Keyboard-based navigation and selection.  
• Direct cell editing and double-click editing.  
• Multi-cell selection with copy, cut, and paste operations.  
• Dynamic addition of user-defined annotation columns.  

---

### Relabel Panel

The relabel panel provides quick-access buttons for assigning or modifying labels in the currently selected grid cell. This design minimizes manual typing and helps maintain labeling consistency across the dataset.

Available labels include:

• TR=turkish

• EN=english

• MIXED=intra-word code-switching

• UID=unidentified item

• NE=named entity

• LANG3=language other than tr and en 

• OTHER=numbers, punctuation marks, symbols, and non-lexical items

Structural constraints are enforced for specific meta-rows, such as Matrix Language and Embedded Language rows, to prevent invalid label assignments.

---

### Control Elements

The toolbar at the top of the main window allows users to manage the annotation workflow.

Core actions include:

• Open: Loading input text.  
• Run: Running the annotation pipeline.  
• Save: Saving annotated output.  

Feature toggles allow users to enable or disable optional components:

• Token-level language identification.  
• Matrix Language computation.  
• Embedded Language computation.  
• Named Entity Recognition.  

---

### Interaction Model

The main window is designed around a semi-automatic interaction model.

• Automatic processing produces an initial annotation layer.  
• Users manually inspect and revise annotations as needed.  
• Auxiliary tools operate directly on the same underlying data model.  

This interaction model provides fine-grained control over annotation decisions while preserving computational efficiency.

## Menu Bar

The menu bar provides access to core application functions, project management utilities, annotation controls, and auxiliary analytical tools.

---

### File

![File menu](images/ui-menu-file.png)

The **File** menu contains basic actions for managing input and output during the annotation workflow.

• **Open Input**: Load a raw text file into the input panel.  
• **Run**: Execute the annotation pipeline on the current input text.  
• **Save Output As**: Export the annotated data to disk.  
• **Exit**: Close the application.

---

### Project

![File menu](images/ui-menu-project.png)

The **Project** menu is used to manage annotation sessions.

• **New Project**: Start a new annotation project and clear the current workspace.  
• **Open Project Save**: Load a previously saved project file (`.trenproj`).  
• **Save Project Progress**: Save the current annotation state for later continuation.

Project files store the input text, annotation data, configuration settings, and interface state.

---

### Annotation

![File menu](images/ui-menu-annotation.png)

The **Annotation** menu provides tools for manual editing and structural modification of the annotation grid.

• **Add New Column**: Create a custom annotation column.  
• **Cut / Copy / Paste**: Edit selected cells in the grid.  
• **Insert Row Before**: Insert a new annotation row.  
• **Remove Row**: Delete the selected annotation row.

---

### Edit Window

![Tools menu](images/ui-full.png)

• **View Full Edit Window**: Open a synchronized, full-size annotation grid in a separate window.

The Full Edit Window mirrors the main annotation grid and remains fully synchronized with it, allowing users to perform extensive edits, multi-cell operations, and navigation on large datasets more comfortably.

---

### Tools

![Tools menu](images/ui-menu-tools.png)

The **Tools** menu provides access to auxiliary analytical and inspection windows.

• **Auto-Glossing Tool**  
• **Concordance (KWIC)**  
• **Show Sentence (Context Viewer)**  
• **Word Frequency List**

## Auto-Glossing Tool

![Auto-Glossing Tool](images/ui-autogloss.png)

The **Auto-Glossing Tool** is an auxiliary window designed to support the annotation of **intra-word code-switching** items. It operates on tokens labeled as **MIXED** and allows users to assign or revise both **Label** and **Gloss** values in a focused workflow.

### Scope and Data Selection

• The tool collects items from the current annotation model that are labeled **MIXED**.  
• Meta rows such as **SentenceID**, **MatrixLang**, and **EmbedLang** are excluded from the tool’s item list.  
• Items are shown in the same visible order as the main annotation grid.

### Interface Elements

• **Status indicator**: shows the current position in the item list (e.g., `MIXED 3/27`).  
• **SentenceID display**: shows the sentence identifier associated with the current token, when available.  
• **Item field (read-only)**: displays the current token string.  
• **Gloss field (editable)**: stores the morphological gloss assigned to the current token.  
• **Auto-Gloss button**: generates a gloss suggestion for the current MIXED token.  
• **Label field (editable)**: stores the label of the current token.  
• **Label buttons**: quick assignment of labels (TR, EN, MIXED, UID, NE, LANG3, OTHER).  
• **Leipzig Gloss Appendix**: opens a reference list of standard glossing abbreviations.

### Navigation and Shortcuts

• **◀ / ▶ buttons**: move to the previous or next item.  
• **Left / Right arrow keys**: same as ◀ / ▶ navigation.  
• **Cmd+Enter / Ctrl+Enter**: run **Auto-Gloss** for the current item.

When navigating away from an item, the current **Label** and **Gloss** values are committed to the underlying annotation model.

### Synchronization with the Main Grid

• Changes made in the Auto-Glossing Tool are written back to the main data model.  
• The corresponding **Label** and **Gloss** cells in the main grid are updated to reflect edits.  
• The currently loaded item is also synchronized with the grid selection to support inspection in context.

### Notes on Glossing

• Gloss output is intended for morphologically interpretable intra-word structures.  
• Glossing follows a Leipzig-style abbreviation inventory and is designed for consistency across the corpus.  
• Users retain full control and may overwrite or refine automatic suggestions.

## Concordance (KWIC)

![Concordance (KWIC)](images/ui-kwic.png)

The **Concordance (KWIC)** tool provides a keyword-in-context view over the input text, allowing users to examine the distribution and local contexts of tokens within the corpus. The tool operates directly on the raw input text while remaining synchronized with the annotation workflow.

### Query Configuration

• **Query field**: accepts a search string to be matched against the input text.  
• **Context (chars)**: defines the number of characters displayed to the left and right of the match.  
• **Case-insensitive**: enables case-insensitive matching.  
• **Regex**: allows regular-expression based searches for advanced querying.

### Result Display

• Results are displayed in a three-column KWIC table:
  • **Left**: left context of the match.  
  • **KWIC**: the matched token or pattern.  
  • **Right**: right context of the match.  
• Matches are ordered according to their position in the input text.  
• The total number of matches is displayed at the bottom of the window.

### Interaction and Navigation

• Selecting a KWIC row automatically highlights the corresponding span in the input panel.  
• Double-clicking a row or pressing **Enter** jumps to the match in the input text.  
• **Prev / Next** buttons allow sequential navigation between matches.  
• Arrow keys can be used to move through the result list.

### Highlighting and Synchronization

• Matched spans are visually highlighted in the input panel.  
• The input cursor is repositioned to the selected match to support contextual inspection.  
• The tool does not modify annotations and is strictly read-only with respect to the annotation grid.

### Use Cases

• Inspecting contextual usage of specific tokens or morphemes.  
• Verifying automatic label assignments against surrounding context.  
• Exploring frequency-independent distribution patterns prior to quantitative analysis.

## Show Sentence (Context Viewer)

The **Show Sentence (Context Viewer)** displays the full sentence containing the currently selected token in the annotation grid, allowing users to inspect annotations in their immediate linguistic context.

• The sentence is extracted directly from the input text using punctuation and line breaks as boundaries.  
• The selected token is highlighted within the sentence for easy identification.  
• The viewer operates in read-only mode and does not modify annotation data.

This tool supports rapid contextual verification during manual annotation and revision.

## Word Frequency List

![Word Frequency List](images/ui-frequency.png)

The **Word Frequency List** window provides a frequency-based summary of tokens derived from the current annotation state. Frequencies are computed dynamically from the annotated data and reflect all manual edits made in the annotation grid.

### Word Frequency List

![Word Frequency List](images/ui-frequency.png)

The **Word Frequency List** window provides a frequency-based summary of tokens derived from the current annotation state. Frequencies are computed dynamically from the annotated data and reflect all manual edits made in the annotation grid.

### Frequency Computation

• Tokens are normalized before counting to ensure consistent frequency estimation.  
• Meta rows and non-token elements are excluded from frequency calculations.  
• Frequencies are aggregated across the entire input text.

```bash
# Token filtering (exclude meta rows)
T' = { t ∈ T | t is not a meta token }

# Token normalization
norm(t) = strip_punctuation(lowercase(t))

# Total frequency
f(w) = Σ I(norm(t_i) = w)

# Label-conditioned frequency
f(w | L) = Σ I(norm(t_i) = w ∧ label_i = L)

# Total token count
N = Σ_w f(w)

# Sorting criterion
sort by: (-f(w), w)
```

### Label-Based Filtering

• Users can restrict frequency counts to specific annotation labels (e.g., TR, EN, MIXED, OTHER).  
• Label filters allow focused inspection of language-specific or structurally relevant subsets.


### Interaction and Export

• Selecting a token and double-clicking (or pressing **Enter**) sends it to the Concordance (KWIC) tool.  
• Frequency tables can be exported as `.csv` files for downstream statistical or corpus-based analyses.

This tool supports both exploratory analysis and the preparation of frequency-based datasets.





