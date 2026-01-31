## 3. System Design and Implementation

This chapter describes the architecture and implementation of an LLM-powered question answering system for tabletop roleplaying game (TTRPG) session notes, built as a Retrieval-Augmented Generation (RAG) pipeline in Python. (#citation_needed)  
The system’s core goal is to answer user questions by retrieving semantically relevant excerpts from a local corpus of campaign notes and constraining the LLM to that retrieved context. (#citation_needed)  
The implementation emphasizes modularity (separate modules for preprocessing, chunking, embedding, indexing, retrieval, reranking, and response generation) to make the pipeline inspectable and maintainable. (#citation_needed)

At a high level, the system follows a staged flow: (1) preprocess raw session notes into cleaned summaries, (2) chunk summaries into semantically coherent units, (3) embed chunks using an embedding model, (4) index vectors in a Qdrant collection, (5) retrieve candidates via vector similarity search, (6) optionally rerank candidates with a cross-encoder, and (7) generate an answer using an LLM with the retrieved chunks as context. (#citation_needed)  
Each stage is exposed through a command-line interface (CLI) so that data preparation, indexing, and querying can be run independently and reproduced with consistent configuration. (#citation_needed)

### 3.0.1 Architectural overview and control flow

The system is organized around a CLI entry point in `main.py`, which routes commands to dedicated handler functions for each pipeline stage. (#citation_needed)  
This design isolates orchestration concerns (argument parsing, configuration overrides, run tracking) from the functional pipeline steps implemented in the `src/` package. (#citation_needed)  
In the current implementation, the CLI supports distinct commands for preprocessing (`preprocess`), indexing (`index`), retrieval-only search (`search`), response generation on top of retrieval (`chat`), and a full end-to-end run (`run`). (#citation_needed)

The `run` command demonstrates the intended full system behavior as a single workflow: it preprocesses raw notes (unless disabled), indexes the processed summaries into Qdrant (unless disabled), retrieves relevant chunks for the user question, and then generates a final answer based on the retrieved context. (#citation_needed)  
This end-to-end command is particularly useful for demonstration and evaluation because it produces a reproducible record of intermediate artifacts when run tracking is enabled. (#citation_needed)  
<<(main.py:cmd_run)Screenshot>>

### 3.0.2 Configuration management and reproducibility

All pipeline stages depend on a shared configuration object loaded from YAML (default: `configs/base.yaml`) using utilities in `src/config.py`. (#citation_needed)  
The configuration includes path settings (raw notes directory, processed output directory, Qdrant storage directory), model choices (embedding model, response model), and retrieval parameters (top-k, per-query limit, feature toggles such as query expansion and reranking). (#citation_needed)  
The configuration loader supports hierarchical overrides by deep-merging dictionaries, allowing CLI arguments or custom YAML files to override only specific nested settings while preserving defaults. (#citation_needed)  
<<(src/config.py:load_config)Screenshot>>

This configuration strategy supports the methodological needs of a thesis project: experiments can be repeated under controlled settings, and the exact configuration used for a run can be recorded alongside the results. (#citation_needed)  
To operationalize that reproducibility, the system optionally creates per-run artifact folders via `src/run_tracker.py`, storing configuration snapshots, retrieval results, and generated responses. (#citation_needed)  
<<(src/run_tracker.py:create_run)Screenshot>>

### 3.0.3 Data preparation: from raw notes to processed summaries

The retrieval quality of a RAG system depends heavily on the quality and consistency of the indexed text. (#citation_needed)  
In this project, the raw input consists of Markdown session note files (pattern configured as `Session *.md`) stored under a raw notes directory (default: `data/raw`). (#citation_needed)  
The preprocessing stage extracts the narrative section of each note by locating a configured header (default: `# Session Start`) and discarding surrounding metadata or unrelated sections. (#citation_needed)  
The preprocessing stage also converts Obsidian-style wikilinks (e.g., `[[Entity]]` or `[[Entity|Alias]]`) into plain text to reduce token noise and improve embedding quality. (#citation_needed)  
<<(src/preprocess.py:preprocess_file)Screenshot>>

An additional dataset-specific normalization is implemented: the preprocessing pipeline assumes a temporal offset where a file named `Session N.md` contains the summary of the previous session, and therefore writes output as `Session (N-1) Summary.txt`. (#citation_needed)  
This logic ensures that later indexing and retrieval align chunk metadata (session number) with the narrative content rather than the filename used during note-taking. (#citation_needed)  
<<(src/preprocess.py:preprocess_all)Screenshot>>

### 3.0.4 Chunking strategy and metadata design

After preprocessing, each cleaned summary file is segmented into smaller units (“chunks”) suitable for retrieval and LLM context construction. (#citation_needed)  
The implementation uses a structural chunking strategy based on top-level bullet points, reflecting the common way session notes are written as event lists and thereby preserving semantic boundaries at the “scene/event” level. (#citation_needed)  
Concretely, `src/chunking.py` scans each processed file line-by-line and starts a new chunk whenever a line begins with a top-level bullet marker (`-`), while keeping nested lines attached to their parent bullet. (#citation_needed)  
<<(src/chunking.py:chunk_by_bullets)Screenshot>>

Each chunk is represented as a Python dictionary containing metadata fields (`session_number`, `chunk_number`, `name`, `source_file`) and the chunk text content. (#citation_needed)  
The `name` field uses a human-readable convention like `Session X - Part Y`, which is later surfaced during retrieval output and used as a label in LLM context formatting. (#citation_needed)  
To ensure stable point identity in the vector database, chunk IDs are deterministically derived from session and chunk number using an integer scheme \(id = session\_number \times 1000 + chunk\_number\). (#citation_needed)  
This deterministic ID scheme supports idempotent indexing and enables efficient skipping of already-indexed chunks in later runs. (#citation_needed)  
<<(src/chunking.py:create_chunk_id)Screenshot>>

### 3.0.5 Embedding generation and vector storage in Qdrant

To enable semantic retrieval, each chunk is converted into a vector embedding using OpenAI’s embedding API through `src/embedding.py`. (#citation_needed)  
The embedding model is configured in `configs/base.yaml` (default: `text-embedding-3-large`), and the embedding client is instantiated using an API key loaded from a secrets file (`configs/secrets.yaml`). (#citation_needed)  
The embedding generation currently performs one embedding request per chunk and attaches the resulting embedding vector to the chunk object. (#citation_needed)  
<<(src/embedding.py:embed_chunks)Screenshot>>

Vector storage and similarity search are implemented using Qdrant via `qdrant-client` in `src/indexing.py`. (#citation_needed)  
The system uses Qdrant in local, file-backed mode (`QdrantClient(path=...)`) so the entire index can be stored in a project-local `qdrant_storage/` directory. (#citation_needed)  
At indexing time, the code creates a collection if it does not yet exist, configures it with cosine distance, and uploads each chunk as a point whose payload stores both metadata and the original text content for later context construction. (#citation_needed)  
<<(src/indexing.py:upload_chunks)Screenshot>>

The indexing pipeline is optimized for iterative development by detecting existing chunk IDs in the collection and embedding/uploading only those chunks that are not already present. (#citation_needed)  
This approach reduces both compute cost and latency during repeated experiments where only a subset of notes changes. (#citation_needed)  
<<(src/indexing.py:get_existing_chunk_ids)Screenshot>>

### 3.0.6 Retrieval, optional query expansion, and optional reranking

At query time, the retrieval module (`src/retrieval.py`) transforms the user question into one or more search queries and performs vector similarity search in Qdrant. (#citation_needed)  
The baseline path uses the original question only; when enabled, query expansion calls an LLM (configured model default: `gpt-4o-mini`) to generate multiple short query variants intended to mitigate vocabulary mismatch between user questions and note phrasing. (#citation_needed)  
Each query variant is embedded using the same embedding model as the indexed chunks, and Qdrant is queried for the top matches per query. (#citation_needed)  
The system aggregates results across queries by keeping the highest similarity score per unique chunk ID, then sorts candidates by score and truncates to a configurable top-k. (#citation_needed)  
<<(src/retrieval.py:search_with_multiple_queries)Screenshot>>

Reranking is implemented as an optional second-stage ranking module in `src/reranking.py` using a cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L6-v2`) from the `sentence-transformers` library. (#citation_needed)  
When enabled, the reranker scores (query, chunk) pairs and reorders the retrieved candidates using the cross-encoder scores to prioritize deeper semantic alignment beyond embedding similarity. (#citation_needed)  
To reduce repeated initialization overhead, the cross-encoder model is cached in a module-level singleton and reused across reranking calls. (#citation_needed)  
<<(src/reranking.py:rerank_results)Screenshot>>

### 3.0.7 Response generation and grounding behavior

The final answer is generated by `src/response.py`, which formats the retrieved chunks into an explicit “SESSION NOTES” context and then calls an OpenAI chat completion endpoint with a system prompt designed to discourage hallucination and to require answers to be grounded in the provided context. (#citation_needed)  
The context formatting includes chunk labels (from the `name` metadata) and uses a clear separator between chunks so the model can distinguish distinct sources. (#citation_needed)  
The response configuration uses a low temperature (default: 0.1) to increase determinism and reduce speculative completions in the absence of evidence. (#citation_needed)  
<<(src/response.py:generate_response)Screenshot>>

The system prompt instructs the model to answer using only the supplied session notes and to explicitly admit when the answer cannot be found in the retrieved materials. (#citation_needed)  
This prompt-level constraint complements retrieval quality by providing a behavioral guardrail against unsupported claims, which is critical for trustworthiness in narrative QA settings. (#citation_needed)  
<<(src/response.py:SYSTEM_PROMPT)Screenshot>>

### 3.0.8 Summary

Overall, the implementation realizes a complete local RAG pipeline with configurable preprocessing, structural chunking, OpenAI embeddings, Qdrant-backed semantic retrieval, optional query expansion and reranking, and grounded response generation. (#citation_needed)  
The modularization across `src/` and the run tracking utilities support iterative experimentation and make the system suitable for evaluation in later thesis chapters. (#citation_needed)
