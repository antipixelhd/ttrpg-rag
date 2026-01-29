# =============================================================================
# Run Tracker Module
# =============================================================================
# This module tracks experiment runs by saving configs, logs, and results
# to timestamped folders in ./runs.

import json
import logging
import yaml
from datetime import datetime
from pathlib import Path

from src.config import get_project_root


def create_run(config, run_name=None):
    """
    Create a new run folder with a timestamp.
    
    Each run gets its own folder like: runs/20260128_143022/
    This keeps all experiment data organized and reproducible.
    
    Args:
        config: The configuration dictionary used for this run
        run_name: Optional custom name to append to the folder name
        
    Returns:
        Path: The path to the newly created run folder
    """
    runs_dir = get_project_root() / 'runs'
    runs_dir.mkdir(exist_ok=True)
    
    # Create a timestamp-based folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if run_name:
        folder_name = f"{timestamp}_{run_name}"
    else:
        folder_name = timestamp
    
    run_dir = runs_dir / folder_name
    run_dir.mkdir(exist_ok=True)
    
    # Also create a results subfolder for search results
    (run_dir / 'results').mkdir(exist_ok=True)
    
    print(f"Created run folder: {run_dir}")
    
    return run_dir


def save_config(run_dir, config):
    """
    Save the configuration used for this run.
    
    This allows you to see exactly what settings were used,
    making experiments reproducible.
    
    Args:
        run_dir: Path to the run folder
        config: The configuration dictionary to save
    """
    config_path = Path(run_dir) / 'config.yaml'
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved config to: {config_path}")


def save_chunks(run_dir, chunks):
    """
    Save the generated chunks for this run.
    
    This lets you inspect what chunks were created and how the
    text was split up.
    
    Args:
        run_dir: Path to the run folder
        chunks: List of chunk dictionaries
    """
    chunks_path = Path(run_dir) / 'chunks.json'
    
    # Remove embeddings from chunks for readability (they're huge)
    chunks_to_save = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        if 'embedding' in chunk_copy:
            # Just note the embedding size, don't save the full vector
            chunk_copy['embedding_size'] = len(chunk_copy['embedding'])
            del chunk_copy['embedding']
        chunks_to_save.append(chunk_copy)
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to: {chunks_path}")


def save_embeddings(run_dir, chunks):
    """
    Save the embedding vectors for this run.
    
    Warning: This file can be large! Only save if you need to
    analyze or compare embeddings across runs.
    
    Args:
        run_dir: Path to the run folder
        chunks: List of chunk dictionaries with embeddings
    """
    embeddings_path = Path(run_dir) / 'embeddings.json'
    
    # Extract just the ID and embedding from each chunk
    embeddings_to_save = []
    for chunk in chunks:
        if 'embedding' in chunk:
            embeddings_to_save.append({
                'session_number': chunk['session_number'],
                'chunk_number': chunk['chunk_number'],
                'name': chunk['name'],
                'embedding': chunk['embedding'],
            })
    
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_to_save, f)
    
    print(f"Saved {len(embeddings_to_save)} embeddings to: {embeddings_path}")


def save_results(run_dir, query, results, query_number=None):
    """
    Save search results for a query.
    
    Results are saved to runs/TIMESTAMP/results/query_001.json
    
    Args:
        run_dir: Path to the run folder
        query: The search query string
        results: List of result dictionaries from the search function
        query_number: Optional number for ordering multiple queries
    """
    results_dir = Path(run_dir) / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename
    if query_number is not None:
        filename = f"query_{query_number:03d}.json"
    else:
        # Use timestamp if no number provided
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"query_{timestamp}.json"
    
    results_path = results_dir / filename
    
    # Save query and results together
    data = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'num_results': len(results),
        'results': results,
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved results to: {results_path}")


def save_response(run_dir, question, response, retrieved_chunks=None, metadata=None):
    """
    Save the LLM response for a query.
    
    This saves the final generated answer along with the question
    and optionally the chunks that were used as context.
    
    Args:
        run_dir: Path to the run folder
        question: The user's question
        response: The LLM's response text
        retrieved_chunks: Optional list of chunks used for context
        metadata: Optional dict with additional metadata (model, temperature, etc.)
    """
    response_path = Path(run_dir) / 'response.json'
    
    data = {
        'question': question,
        'response': response,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add metadata if provided
    if metadata:
        data['metadata'] = metadata
    
    # Add chunk references if provided (just IDs and names, not full content)
    if retrieved_chunks:
        data['context_chunks'] = [
            {
                'chunk_id': c.get('chunk_id'),
                'name': c.get('name'),
                'source': c.get('source', c.get('source_file')),
            }
            for c in retrieved_chunks
        ]
    
    with open(response_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved response to: {response_path}")


def get_logger(run_dir, name='run'):
    """
    Create a logger that writes to both console and a log file in the run folder.
    
    This captures all output from the run so you can review what happened.
    
    Args:
        run_dir: Path to the run folder
        name: Name for the logger (default: 'run')
        
    Returns:
        logging.Logger: A configured logger instance
    """
    log_path = Path(run_dir) / 'run.log'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers (in case this is called multiple times)
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - logs everything to file
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - also prints to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_path}")
    
    return logger


def list_runs():
    """
    List all existing run folders.
    
    Returns:
        list: List of run folder paths, sorted by date (newest first)
    """
    runs_dir = get_project_root() / 'runs'
    
    if not runs_dir.exists():
        return []
    
    # Find all run folders (they start with a timestamp)
    run_folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    
    # Sort by name (which includes timestamp) in reverse order
    run_folders.sort(reverse=True)
    
    return run_folders


def get_latest_run():
    """
    Get the most recent run folder.
    
    Returns:
        Path: Path to the latest run folder, or None if no runs exist
    """
    runs = list_runs()
    return runs[0] if runs else None
