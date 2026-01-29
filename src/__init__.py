# =============================================================================
# TTRPG RAG System - Source Package
# =============================================================================
# This package contains all modules for the RAG pipeline:
#   - config.py      : Configuration loading and merging
#   - preprocess.py  : Extract summaries from raw session notes
#   - chunking.py    : Split summaries into chunks
#   - embedding.py   : Generate embeddings using OpenAI
#   - indexing.py    : Upload chunks to Qdrant vector store
#   - retrieval.py   : Search the vector store
#   - reranking.py   : Rerank results with cross-encoder model
#   - response.py    : Generate LLM responses from retrieved chunks
#   - run_tracker.py : Track experiments in ./runs folder
