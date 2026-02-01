# TTRPG RAG System

A Retrieval-Augmented Generation (RAG) system for querying TTRPG session summaries.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the secrets template and add your OpenAI API key:

```bash
cp configs/secrets.example.yaml configs/secrets.yaml
```

Edit `configs/secrets.yaml` and replace the placeholder with your actual API key:

### 3. Initialize the System

Run preprocessing to extract summaries:

```bash
python main.py preprocess
```

Run indexing to create embeddings and populate the vector database:

```bash
python main.py index
```

### 4. Query the System

Search for relevant chunks:

```bash
python main.py search "What happened in session 5?"
```

Get an AI-generated answer:

```bash
python main.py chat "Who is the main villain?"
```

## Available Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Extract summaries from raw session notes |
| `index` | Create embeddings and upload to Qdrant |
| `search` | Search for relevant session information |
| `chat` | Get AI-generated answers using RAG |
| `run` | Run the full pipeline (preprocess + index + search + chat) |

## Common Options

- `-c, --config` - Use a custom config file
- `-v, --verbose` - Print detailed progress
- `--evaluate` - Run evaluation on question set (for `search` and `chat`)

## Configuration

Main configuration is in `configs/base.yaml`
