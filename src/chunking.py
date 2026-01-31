
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
    chunk_number = 0
    chunk_lines = []
    in_chunk = False

    def add_chunk(chunks, session_number, chunk_number, lines):
        content = "".join(lines).strip()
        chunks.append({
            "session_number": session_number,
            "chunk_number": chunk_number,
            "content": content,
            "name": f"Session {session_number} - Part {chunk_number}",
            "source_file": f"Session {session_number} Summary.txt",
        })
    for line in lines:
        starts_new_chunk = line.startswith("-")

        if starts_new_chunk:
            if in_chunk:
                chunk_number += 1
                add_chunk(chunks, session_number, chunk_number, chunk_lines)
                chunk_lines = []
            in_chunk = True
            chunk_lines.append(line)
            continue

        if in_chunk:
            chunk_lines.append(line)
    # flush last chunk
    if in_chunk and chunk_lines:
        chunk_number += 1
        add_chunk(chunks, session_number, chunk_number, chunk_lines)
    
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
