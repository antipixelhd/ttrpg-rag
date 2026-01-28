# TTRPG RAG System

A local Retrieval-Augmented Generation (RAG) system designed for querying and analyzing tabletop RPG (TTRPG) session summaries. This system indexes session notes, retrieves relevant context using semantic search, and generates natural language responses to queries about campaign history, characters, and events.

## Quickstart

### Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
   - **Windows (CMD)**: `.venv\Scripts\activate.bat`
   - **Linux/macOS**: `source .venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

API keys and secrets are stored in configuration files and are **NOT committed** to version control.

1. Copy the example secrets file:
```bash
cp configs/secrets.example.yaml configs/secrets.yaml
```

2. Edit `configs/secrets.yaml` and add your API keys:
   - OpenAI API key (for embeddings and generation)
   - Google Gemini API key (optional, for alternative generation)
   - Qdrant URL and API key (if using remote Qdrant instance)

3. The `configs/base.yaml` file will reference `secrets.yaml` for sensitive values (exact mechanism TBD in later implementation steps).

## Usage

### Command Line Interface

View available commands:
```bash
python src/cli.py --help
```

Initialize a new RAG pipeline run:
```bash
python src/cli.py init-run --config configs/base.yaml
```

### Run Artifacts

Each pipeline execution creates a unique run directory under `runs/<run_id>/` containing:
- Processed data and intermediate outputs
- Generated embeddings and index snapshots
- Retrieval results and evaluation metrics
- Logs and configuration snapshots

This structure enables experiment tracking, reproducibility, and comparison of different pipeline configurations.

## Project Structure

```
ttrpg-rag/
├── src/               # Source code
│   ├── cli.py         # Command-line interface
│   ├── config.py      # Configuration management
│   ├── types.py       # Type definitions
│   ├── paths.py       # Path utilities
│   ├── preprocess.py  # Data preprocessing
│   ├── chunking.py    # Text chunking strategies
│   ├── index.py       # Vector index management
│   ├── retrieval.py   # Retrieval strategies
│   ├── expansion.py   # Query expansion
│   ├── aggregation.py # Result aggregation
│   ├── reranking.py   # Reranking strategies
│   ├── prompting.py   # Prompt templates
│   ├── generation.py  # LLM generation
│   ├── evaluation.py  # Evaluation metrics
│   └── reporting.py   # Report generation
├── configs/           # Configuration files
│   ├── base.yaml      # Base configuration
│   └── secrets.example.yaml  # Template for secrets
├── data/              # Data directories (gitignored)
├── runs/              # Run artifacts (gitignored)
└── requirements.txt   # Python dependencies
```

## Development Status

This project is under active development. The scaffold is in place for incremental implementation of RAG pipeline components.
