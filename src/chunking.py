
import re
from pathlib import Path
from src.config import load_config
from src.config import resolve_path

def extract_session_number(filename):
    match = re.search(r'Session (\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def chunk_by_bullets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    filename = Path(file_path).name
    session_number = extract_session_number(filename)
    
    chunks = []
    current_chunk_lines = []
    chunk_number = 0
    
    for line in lines:
        if line.startswith('-'):
            if current_chunk_lines:
                chunk_number += 1
                chunk_content = ''.join(current_chunk_lines).strip()
                
                chunks.append({
                    'session_number': session_number,
                    'chunk_number': chunk_number,
                    'content': chunk_content,
                    'name': f"Session {session_number} - Part {chunk_number}",
                    'source_file': f"Session {session_number} Summary.txt"
                })
                
                current_chunk_lines = []
            
            current_chunk_lines.append(line)
            
        elif line.strip():
            if current_chunk_lines:
                current_chunk_lines.append(line)
                
        elif current_chunk_lines:
            current_chunk_lines.append(line)
    
    if current_chunk_lines:
        chunk_number += 1
        chunk_content = ''.join(current_chunk_lines).strip()
        
        chunks.append({
            'session_number': session_number,
            'chunk_number': chunk_number,
            'content': chunk_content,
            'name': f"Session {session_number} - Part {chunk_number}",
            'source_file': f"Session {session_number} Summary.txt"
        })
    
    return chunks

def get_all_chunks(config, logger=None):
    processed_path = resolve_path(config['paths']['processed'])
    
    summary_files = list(processed_path.glob('Session * Summary.txt'))
    summary_files.sort(key=lambda x: extract_session_number(x.name))
    
    if not summary_files:
        message = f"No summary files found in {processed_path}"
        if logger:
            logger.warning(message)
        else:
            print(f"Warning: {message}")
        return []
    
    all_chunks = []
    
    for file_path in summary_files:
        chunks = chunk_by_bullets(file_path)
        all_chunks.extend(chunks)
        
        message = f"Chunked {file_path.name}: {len(chunks)} chunks"
        if logger:
            logger.info(message)
        else:
            print(message)
    
    message = f"Total chunks created: {len(all_chunks)} from {len(summary_files)} files"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return all_chunks

def create_chunk_id(chunk):
    return chunk['session_number'] * 1000 + chunk['chunk_number']
