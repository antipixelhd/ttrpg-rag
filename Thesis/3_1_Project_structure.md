## 3.1 Project structure

This section documents the concrete project layout used to implement the TTRPG RAG system and explains how responsibilities are separated across modules and directories. (#citation_needed)  
The key design intent of the structure is to make each pipeline stage (preprocessing, chunking, embedding, indexing, retrieval, reranking, and response generation) independently executable and auditable. (#citation_needed)  
In practice, this means that orchestration lives in a single entry point (`main.py`), configuration is centralized in `configs/`, core functionality is packaged under `src/`, and data/artifacts are written into clearly separated runtime directories. (#citation_needed)

### 3.1.1 Repository top-level layout

At the root of the repository, the following files and directories define the operational boundary of the system. (#citation_needed)

- `main.py`: The command-line interface and orchestration layer that parses arguments, loads configuration, optionally enables run tracking, and calls into the pipeline modules. (#citation_needed)  
  `main.py` is intentionally kept “thin” in terms of business logic, delegating core functionality to `src/` so that the pipeline remains reusable outside the CLI (e.g., for notebooks or evaluation harnesses). (#citation_needed)  
  <<(main.py:main)Screenshot>>

- `src/`: The Python package containing the modular pipeline implementation. (#citation_needed)  
  The package boundary enforces a clean separation between orchestration (in `main.py`) and implementation (in `src/`), helping avoid monolithic scripts and enabling targeted testing and refactoring. (#citation_needed)  
  <<(src/config.py:get_project_root)Screenshot>>

- `configs/`: Configuration files that parameterize the pipeline, including a default configuration (`base.yaml`). (#citation_needed)  
  Secrets (API keys) are intentionally excluded from version control and loaded from `configs/secrets.yaml` at runtime to reduce the risk of accidental credential leakage. (#citation_needed)  
  <<(src/config.py:load_config)Screenshot>>

- `requirements.txt`: Dependency declarations for the Python environment. (#citation_needed)  
  The minimal dependency set reflects the core technology choices: OpenAI API client for embeddings and generation, Qdrant client for vector storage, and YAML utilities for configuration. (#citation_needed)

- `Thesis/`: Research-writing artifacts for the thesis itself, including generated chapter markdown files and the evolving continuity summary. (#citation_needed)

In addition to these primary components, the repository contains several root-level scripts (`ConvertSummariest.py`, `SummariesToChunks.py`, `QdrantSearch.py`) that represent earlier prototypes or one-off utilities. (#citation_needed)  
These scripts are not wired into the current modular pipeline (which is centered around `main.py` + `src/`) and therefore should be interpreted as experimental predecessors rather than the canonical implementation. (#citation_needed)  
Their presence is still useful for documenting the project’s evolution and design rationale, particularly when motivating later refactors into reusable modules. (#citation_needed)

### 3.1.2 `src/` package: functional decomposition of the RAG pipeline

The `src/` directory is the primary implementation surface of the system and is structured around single-responsibility modules. (#citation_needed)  
Each module corresponds to a conceptual stage of the RAG pipeline and exposes a small set of functions used by the CLI. (#citation_needed)

- `src/config.py`: Configuration and path resolution utilities. (#citation_needed)  
  It provides YAML loading (`load_yaml_file`), recursive merging (`deep_merge`), and consistent resolution of relative paths against the project root (`resolve_path`). (#citation_needed)  
  This keeps all downstream modules free of hard-coded paths and enables reproducible parameter sweeps by changing only configuration. (#citation_needed)  
  <<(src/config.py:deep_merge)Screenshot>>

- `src/preprocess.py`: The preprocessing stage that transforms raw Markdown session notes into cleaned summaries ready for chunking. (#citation_needed)  
  It encapsulates section extraction, Obsidian wikilink normalization, and dataset-specific mapping of filenames to chronological session identifiers. (#citation_needed)  
  <<(src/preprocess.py:preprocess_all)Screenshot>>

- `src/chunking.py`: The chunking stage that segments processed summaries into retrieval units. (#citation_needed)  
  The current implementation uses bullet-point boundaries as a structural prior and produces chunk dictionaries with consistent metadata fields used by later stages. (#citation_needed)  
  It also defines deterministic chunk IDs via `create_chunk_id`, which are reused as point IDs in Qdrant. (#citation_needed)  
  <<(src/chunking.py:chunk_by_bullets)Screenshot>>

- `src/embedding.py`: The embedding stage that converts each chunk’s text into a numeric vector using an embedding model configured in YAML. (#citation_needed)  
  This module centralizes all embedding calls so model choice, API key handling, and potential batching strategies are localized rather than duplicated across the codebase. (#citation_needed)  
  <<(src/embedding.py:get_embedding)Screenshot>>

- `src/indexing.py`: The indexing stage that persists embeddings in Qdrant and manages collection setup. (#citation_needed)  
  It creates a local Qdrant client, creates the collection if needed with cosine distance, checks for already-indexed chunks, and uploads only new points to support incremental updates. (#citation_needed)  
  <<(src/indexing.py:index_all)Screenshot>>

- `src/retrieval.py`: The retrieval stage that embeds the user query, performs vector similarity search in Qdrant, optionally expands queries, aggregates results, and returns a structured list for downstream use. (#citation_needed)  
  The retrieval output intentionally includes chunk labels and text content to support explainability (showing which passages were used) and to simplify prompt assembly. (#citation_needed)  
  <<(src/retrieval.py:search)Screenshot>>

- `src/reranking.py`: The optional reranking stage that applies a cross-encoder model to reorder retrieved candidates. (#citation_needed)  
  It is architected as an optional enhancement (feature flag in config) and is loaded lazily to avoid paying the import/model-load cost when reranking is disabled. (#citation_needed)  
  <<(src/reranking.py:load_reranker)Screenshot>>

- `src/response.py`: The response generation stage that converts retrieved chunks into a formatted context block and calls a chat model to produce the final answer. (#citation_needed)  
  This module is also the natural home for prompting policy, including system instructions that constrain the model to retrieved evidence and encourage explicit “unknown” responses when evidence is insufficient. (#citation_needed)  
  <<(src/response.py:format_context)Screenshot>>

- `src/run_tracker.py`: Experiment/run tracking utilities to store configs, chunks, retrieval results, and responses in timestamped run directories. (#citation_needed)  
  This subsystem supports reproducibility and later evaluation by ensuring that intermediate artifacts are not lost between experiments. (#citation_needed)  
  <<(src/run_tracker.py:save_results)Screenshot>>

This modular structure reflects a deliberate choice to treat the RAG system as a pipeline of composable transformations rather than a single opaque LLM call. (#citation_needed)  
Such decomposition makes it possible to evaluate and improve components independently (e.g., changing chunking without changing response prompts, or enabling reranking without touching indexing). (#citation_needed)

### 3.1.3 Committed code vs runtime-generated data and artifacts

The repository intentionally distinguishes between committed, reproducible inputs (code and non-secret configuration) and runtime-generated outputs (datasets, indices, run artifacts). (#citation_needed)  
This separation is enforced through `.gitignore`, which excludes `data/raw/`, `data/processed/`, `qdrant_storage/`, and `runs/` from version control. (#citation_needed)  
The intent is to keep the repository small, avoid committing potentially sensitive campaign content, and prevent large binary-like index artifacts from polluting Git history. (#citation_needed)

- `data/raw/`: Expected location for raw Markdown session notes used as input to preprocessing. (#citation_needed)  
- `data/processed/`: Output location for cleaned summary files produced by preprocessing and consumed by chunking/indexing. (#citation_needed)  
- `qdrant_storage/`: Local on-disk persistence for the Qdrant vector store. (#citation_needed)  
- `runs/`: Timestamped run directories storing experiment artifacts such as configuration snapshots and retrieval results. (#citation_needed)

Because these directories are created at runtime (e.g., via `Path.mkdir(..., exist_ok=True)` calls in the pipeline), a fresh clone of the repository can remain lightweight until the user explicitly runs preprocessing or indexing. (#citation_needed)  
This behavior supports reproducible research workflows where data and artifacts can be regenerated from the same code/config, while still keeping sensitive content out of the committed repository. (#citation_needed)

### 3.1.4 Notes on prototype scripts and evolving structure

The presence of legacy scripts at the repository root provides context on the project’s iterative development. (#citation_needed)  
For example, earlier scripts reference alternate directories such as `Source/` and `summaries/`, and some use environment variables via `.env`, whereas the modular pipeline standardizes on `data/raw`, `data/processed`, and YAML-based configuration with `configs/secrets.yaml`. (#citation_needed)  
This shift reflects a move from ad hoc experimentation to a thesis-ready system that prioritizes configurability, clarity of data lineage, and repeatable execution. (#citation_needed)

### 3.1.5 Summary

In summary, the project structure is intentionally pipeline-oriented: `main.py` orchestrates, `src/` implements each stage with single-responsibility modules, `configs/` parameterizes behavior, and runtime directories store data and artifacts outside version control. (#citation_needed)  
This organization directly supports the thesis goals of explaining and evaluating the system, since each stage can be inspected, benchmarked, and improved without destabilizing unrelated components. (#citation_needed)
