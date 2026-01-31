## 3.2 Data basis

This section describes the dataset used for the TTRPG RAG system and the rationale for the preprocessing steps that transform raw session notes into an indexable corpus. (#citation_needed)  
Because RAG performance is constrained by the quality, consistency, and semantic “retrievability” of the indexed text, it is necessary to define the data basis precisely and document how noise and structural artifacts are handled. (#citation_needed)  
The system is designed around a local corpus of campaign notes authored during play, which introduces realistic irregularities such as inconsistent formatting, tool-specific markup, and chronological mismatches between filenames and content. (#citation_needed)

### 3.2.1 Description of dataset

The raw dataset is expected to consist of Markdown files stored under a configurable directory (default: `data/raw`) and matched by an input glob pattern (default: `Session *.md`). (#citation_needed)  
This pattern is set in `configs/base.yaml` under the `preprocess.input_pattern` key and is consumed by the preprocessing pipeline to discover input documents. (#citation_needed)  
<<(src/preprocess.py:preprocess_all)Screenshot>>

Each raw file represents a session note document from a long-running tabletop campaign and is primarily structured as human-authored narrative notes rather than machine-oriented records. (#citation_needed)  
The format is characterized by informal text, mixed levels of detail, and frequent use of bullet lists to capture sequences of events, decisions, and dialogue. (#citation_needed)  
This style is beneficial for narrative recall but creates challenges for automated retrieval because semantically important facts may be distributed across nested bullets or embedded in shorthand phrasing. (#citation_needed)

The notes are assumed to be authored in (or exported from) a personal knowledge management tool with Markdown support, such as Obsidian. (#citation_needed)  
As a result, the raw text may contain tool-specific syntax elements that are not meaningful for semantic search (e.g., internal links, aliases, or vault-specific identifiers). (#citation_needed)  
In this dataset, the dominant tool artifact is Obsidian wiki-link syntax using double brackets, such as `[[Entity]]` and `[[Entity|Alias]]`. (#citation_needed)

In addition to markup artifacts, the dataset includes a chronological naming quirk that directly affects downstream metadata. (#citation_needed)  
The preprocessing code encodes the assumption that a file named `Session N.md` often contains the summary for the previous session \(N-1\), and therefore the system maps filenames to an “output session number” by decrementing the extracted session number. (#citation_needed)  
This assumption is explicitly documented in code comments and implemented in `src/preprocess.py` to ensure that the indexed metadata reflects the narrative timeline rather than the authoring workflow. (#citation_needed)  
<<(src/preprocess.py:extract_session_number)Screenshot>>

### 3.2.2 Preprocessing of notes

Preprocessing transforms raw Markdown notes into a cleaned “processed” corpus that is suitable for chunking, embedding, and indexing. (#citation_needed)  
The output of preprocessing is written to a configurable directory (default: `data/processed`) as plain text summaries (`Session X Summary.txt`). (#citation_needed)  
The system treats this processed directory as the canonical input for subsequent chunking and indexing steps to enforce a stable, clean interface between ingestion and retrieval. (#citation_needed)

The preprocessing module (`src/preprocess.py`) implements three main operations: (1) section extraction, (2) wiki-link normalization, and (3) session-number mapping and output writing. (#citation_needed)  
These operations are intentionally applied in a fixed order so that extraction limits the text volume first, then cleaning removes syntactic noise, and finally outputs are named consistently for indexing. (#citation_needed)

#### (1) Section extraction via a header anchor

Raw session notes often include non-narrative content such as preparatory lists, reminders, meta commentary, or planning sections that are not intended to be queried as “what happened” facts. (#citation_needed)  
To reduce indexing noise and improve retrieval precision, the system extracts only the narrative portion beginning at a configured header string (default: `# Session Start`). (#citation_needed)  
This header is configured in `configs/base.yaml` as `preprocess.extract_section` and used by `extract_summary` to locate the relevant section. (#citation_needed)  
<<(src/preprocess.py:extract_summary)Screenshot>>

The extraction function uses a regular expression that captures all text after the target header until the next Markdown header of the same or higher level (or end-of-file). (#citation_needed)  
This approach is robust to varying note lengths and avoids reliance on fixed line counts or brittle heuristics. (#citation_needed)  
If the configured header is missing, the preprocessing pipeline treats the summary as empty and skips writing a processed output for that file. (#citation_needed)

#### (2) Obsidian wiki-link normalization

Obsidian wiki-links encode internal references in a form that is visually compact for the author but semantically redundant for embedding models and retrieval. (#citation_needed)  
For example, `[[Waterdeep]]` introduces bracket tokens that are not part of the underlying entity name, and `[[Lady_Silverhand|Laeral]]` contains both a link target and an alias that should be rendered as the alias in plain text. (#citation_needed)  
The preprocessing stage normalizes these links by applying two regular-expression substitutions: first replacing the alias form with the alias text, then replacing the simple link form with the link text. (#citation_needed)  
<<(src/preprocess.py:remove_obsidian_links)Screenshot>>

This normalization is intended to improve semantic retrieval by presenting entity mentions as ordinary tokens and reducing the likelihood that embeddings encode bracket syntax as meaningful signals. (#citation_needed)  
It also improves the readability of retrieved passages when they are later inserted into the LLM prompt context. (#citation_needed)

#### (3) Session-number mapping and output naming

Metadata quality is crucial for interpretability in narrative QA, because users often reason about events chronologically (“what happened in session 12?”) and because evaluation may require mapping answers back to specific sessions. (#citation_needed)  
The preprocessing pipeline therefore assigns an explicit session number to each processed output file based on parsing the input filename and then applying a dataset-specific offset. (#citation_needed)  
Concretely, `extract_session_number` parses an integer from filenames of the form `Session (\d+)`, and `preprocess_all` writes the output summary to `Session (N-1) Summary.txt` for an input `Session N.md`. (#citation_needed)  
<<(src/preprocess.py:preprocess_all)Screenshot>>

The code additionally guards against invalid or degenerate cases by skipping files where the session number cannot be extracted or where the decremented output session number would be non-positive. (#citation_needed)  
These checks prevent the creation of misleading outputs (e.g., “Session 0 Summary.txt”) and ensure that later stages can assume valid session numbering. (#citation_needed)

### 3.2.3 Implications for downstream chunking and retrieval

The chosen preprocessing steps create a clean and consistent interface for later pipeline modules, especially chunking and retrieval. (#citation_needed)  
Because chunking relies on bullet-point structure (`src/chunking.py`) and retrieval later surfaces chunk labels like `Session X - Part Y`, it is important that processed summaries preserve the bullet formatting of the extracted session section. (#citation_needed)  
At the same time, removing wiki-link syntax ensures that chunk content remains readable and semantically focused when embedded and when presented as evidence in the generated answer. (#citation_needed)

By mapping filenames to chronological session numbers during preprocessing, the system makes chunk metadata consistent with the narrative timeline, which improves user-facing interpretability of retrieval results and enables evaluation that references sessions as ground-truth units. (#citation_needed)  
This also allows Qdrant payload fields such as `session_number` and `source_file` to reflect the corrected session identity rather than the authoring filename, reducing confusion during debugging and demonstrations. (#citation_needed)

### 3.2.4 Summary

The data basis for the system is a realistic corpus of Markdown session notes with tool-specific syntax and a dataset-specific session offset that must be corrected for chronological integrity. (#citation_needed)  
The preprocessing stage extracts the intended narrative section, normalizes Obsidian links into plain text, and writes consistently named processed summaries that downstream stages can chunk, embed, index, and retrieve reliably. (#citation_needed)
