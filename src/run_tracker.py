
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path

from src.config import get_project_root

def create_run(config, run_name=None):
    runs_dir = get_project_root() / 'runs'
    runs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if run_name:
        folder_name = f"{timestamp}_{run_name}"
    else:
        folder_name = timestamp
    
    run_dir = runs_dir / folder_name
    run_dir.mkdir(exist_ok=True)
    
    (run_dir / 'results').mkdir(exist_ok=True)
    
    print(f"Created run folder: {run_dir}")
    
    return run_dir

def save_config(run_dir, config):
    config_path = Path(run_dir) / 'config.yaml'
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved config to: {config_path}")

def save_chunks(run_dir, chunks):
    chunks_path = Path(run_dir) / 'chunks.json'
    
    chunks_to_save = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        if 'embedding' in chunk_copy:
            chunk_copy['embedding_size'] = len(chunk_copy['embedding'])
            del chunk_copy['embedding']
        chunks_to_save.append(chunk_copy)
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to: {chunks_path}")

def save_embeddings(run_dir, chunks):
    embeddings_path = Path(run_dir) / 'embeddings.json'
    
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
    results_dir = Path(run_dir) / 'results'
    results_dir.mkdir(exist_ok=True)
    
    if query_number is not None:
        filename = f"query_{query_number:03d}.json"
    else:
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"query_{timestamp}.json"
    
    results_path = results_dir / filename
    
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
    response_path = Path(run_dir) / 'response.json'
    
    data = {
        'question': question,
        'response': response,
        'timestamp': datetime.now().isoformat(),
    }
    
    if metadata:
        data['metadata'] = metadata
    
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
    log_path = Path(run_dir) / 'run.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_path}")
    
    return logger

def list_runs():
    runs_dir = get_project_root() / 'runs'
    
    if not runs_dir.exists():
        return []
    
    run_folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    
    run_folders.sort(reverse=True)
    
    return run_folders

def get_latest_run():
    runs = list_runs()
    return runs[0] if runs else None
