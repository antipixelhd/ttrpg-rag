# =============================================================================
# Chunking Module
# =============================================================================
# This module splits processed summary files into smaller chunks.
# Each chunk becomes a separate entry in the vector database for search.

import re
from pathlib import Path

from src.config import resolve_path


def extract_session_number(filename):
    """
    Extract the session number from a filename.
    
    Args:
        filename: A filename like "Session 20 Summary.md"
        
    Returns:
        int: The session number (e.g., 20), or 0 if not found
        
    Example:
        extract_session_number("Session 20 Summary.md") -> 20
    """
    match = re.search(r'Session (\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def chunk_by_bullets(file_path):
    """
    Split a summary file into chunks based on top-level bullet points.
    
    This function reads a markdown file and creates a separate chunk for
    each top-level bullet point (lines starting with '-'). Nested content
    under each bullet is kept together with its parent.
    
    Args:
        file_path: Path to the processed summary file
        
    Returns:
        list: List of chunk dictionaries, each containing:
            - session_number: Which session this is from
            - chunk_number: Order of this chunk within the session
            - content: The actual text content
            - name: Human-readable name like "Session 5 - Part 2"
            - source_file: Original filename
            
    Example output:
        [
            {
                'session_number': 5,
                'chunk_number': 1,
                'content': '- The party arrived at the castle...',
                'name': 'Session 5 - Part 1',
                'source_file': 'Session 5 Summary.md'
            },
            ...
        ]
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Get metadata from filename
    filename = Path(file_path).name
    session_number = extract_session_number(filename)
    
    chunks = []
    current_chunk_lines = []
    chunk_number = 0
    
    for line in lines:
        # Check if this is a top-level bullet point (starts with '-')
        if line.startswith('-'):
            # If we have accumulated lines from a previous chunk, save it
            if current_chunk_lines:
                chunk_number += 1
                chunk_content = ''.join(current_chunk_lines).strip()
                
                chunks.append({
                    'session_number': session_number,
                    'chunk_number': chunk_number,
                    'content': chunk_content,
                    'name': f"Session {session_number} - Part {chunk_number}",
                    'source_file': f"Session {session_number} Summary.md"
                })
                
                # Reset for the next chunk
                current_chunk_lines = []
            
            # Start a new chunk with this bullet point
            current_chunk_lines.append(line)
            
        elif line.strip():
            # Non-empty line (nested content or continuation)
            # Add to current chunk if we're inside one
            if current_chunk_lines:
                current_chunk_lines.append(line)
                
        elif current_chunk_lines:
            # Empty line - keep it to maintain structure
            current_chunk_lines.append(line)
    
    # Don't forget the last chunk!
    if current_chunk_lines:
        chunk_number += 1
        chunk_content = ''.join(current_chunk_lines).strip()
        
        chunks.append({
            'session_number': session_number,
            'chunk_number': chunk_number,
            'content': chunk_content,
            'name': f"Session {session_number} - Part {chunk_number}",
            'source_file': f"Session {session_number} Summary.md"
        })
    
    return chunks


def get_all_chunks(config, logger=None):
    """
    Load all processed summary files and split them into chunks.
    
    This function:
    1. Finds all summary files in the processed folder
    2. Sorts them by session number
    3. Chunks each file using the configured strategy
    4. Returns all chunks with metadata
    
    Args:
        config: Configuration dictionary with paths and chunking settings
        logger: Optional logger for tracking progress
        
    Returns:
        list: All chunks from all session summaries, sorted by session number
    """
    # Get the processed summaries directory
    processed_path = resolve_path(config['paths']['processed'])
    
    # Find all summary files and sort by session number
    summary_files = list(processed_path.glob('Session * Summary.md'))
    summary_files.sort(key=lambda x: extract_session_number(x.name))
    
    if not summary_files:
        message = f"No summary files found in {processed_path}"
        if logger:
            logger.warning(message)
        else:
            print(f"Warning: {message}")
        return []
    
    # Collect chunks from all files
    all_chunks = []
    
    for file_path in summary_files:
        # For now, we only support bullet point chunking
        # This could be extended to support other strategies based on config
        chunks = chunk_by_bullets(file_path)
        all_chunks.extend(chunks)
        
        message = f"Chunked {file_path.name}: {len(chunks)} chunks"
        if logger:
            logger.info(message)
        else:
            print(message)
    
    # Summary
    message = f"Total chunks created: {len(all_chunks)} from {len(summary_files)} files"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return all_chunks


def create_chunk_id(chunk):
    """
    Create a unique ID for a chunk based on its metadata.
    
    This ID is used as the point ID in Qdrant.
    
    Args:
        chunk: A chunk dictionary with session_number and chunk_number
        
    Returns:
        int: A unique integer ID
        
    Example:
        Session 5, Part 3 -> 5003 (session * 1000 + chunk)
    """
    # This gives us IDs like 5001, 5002, 5003 for session 5
    # Allows up to 999 chunks per session
    return chunk['session_number'] * 1000 + chunk['chunk_number']
