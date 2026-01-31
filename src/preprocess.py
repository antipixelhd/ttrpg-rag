
import os
import re
from pathlib import Path

from src.config import resolve_path

def extract_session_number(filename):
    match = re.search(r'Session (\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def remove_obsidian_links(text):
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    
    return text

def extract_summary(text, section_header="# Session Start"):
    escaped_header = re.escape(section_header)
    pattern = escaped_header + r'\s*(.*?)(?=\n#|\Z)'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

def preprocess_file(input_path, section_header="# Session Start"):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    summary = extract_summary(content, section_header)
    
    cleaned = remove_obsidian_links(summary)
    
    return cleaned

def preprocess_all(config, verbose=False):
    raw_path = resolve_path(config['paths']['raw_notes'])
    processed_path = resolve_path(config['paths']['processed'])
    input_pattern = config['preprocess']['input_pattern']
    section_header = config['preprocess']['extract_section']
    
    processed_path.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(raw_path.glob(input_pattern))
    
    if not input_files:
        print(f"Warning: No files found matching '{input_pattern}' in {raw_path}")
        return []
    
    print(f"Preprocessing {len(input_files)} files...")
    
    processed_files = []
    total = len(input_files)
    
    for i, input_file in enumerate(input_files, 1):
        session_num = extract_session_number(input_file.name)
        
        if session_num is None:
            print(f"Warning: Skipping {input_file.name} - could not extract session number")
            continue
        
        output_session_num = session_num - 1
        
        if output_session_num <= 0:
            print(f"Warning: Skipping {input_file.name} - output session number would be {output_session_num}")
            continue
        
        output_name = f"Session {output_session_num} Summary.txt"
        output_path = processed_path / output_name
        
        cleaned_summary = preprocess_file(input_file, section_header)
        
        if not cleaned_summary.strip():
            print(f"Warning: Skipping {input_file.name} - no content found in '{section_header}' section")
            continue
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_summary)
        
        processed_files.append(output_path)
        
        if verbose:
            print(f"  Processed {i}/{total}: {input_file.name} -> {output_name}")
    
    print(f"Preprocessing complete: {len(processed_files)} files processed")
    
    return processed_files
