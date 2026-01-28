# =============================================================================
# Preprocessing Module
# =============================================================================
# This module extracts and cleans summaries from raw session notes.
# It handles Obsidian-specific syntax and extracts the relevant sections.

import os
import re
from pathlib import Path

from src.config import resolve_path


def remove_obsidian_links(text):
    """
    Remove Obsidian wiki-link syntax and replace with plain text.
    
    Obsidian uses [[link]] and [[link|alias]] syntax for internal links.
    This function converts them to readable text:
      - [[link|alias]] becomes "alias"
      - [[link]] becomes "link"
    
    Args:
        text: The text containing Obsidian links
        
    Returns:
        str: The text with links converted to plain text
        
    Example:
        "Met [[Bob the Wizard|Bob]] at the [[Tavern]]"
        becomes "Met Bob at the Tavern"
    """
    # First, handle links with aliases: [[link|alias]] -> alias
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    
    # Then, handle simple links: [[link]] -> link
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    
    return text


def extract_summary(text, section_header="# Session Start"):
    """
    Extract content from a specific section of a markdown file.
    
    The function finds the specified header and extracts everything until
    the next header of the same or higher level (or end of file).
    
    Args:
        text: The full markdown text
        section_header: The header to look for (default: "# Session Start")
        
    Returns:
        str: The extracted section content, or empty string if not found
    """
    # Build a regex pattern to find the section
    # This matches the header and captures everything until the next # heading or EOF
    escaped_header = re.escape(section_header)
    pattern = escaped_header + r'\s*(.*?)(?=\n#|\Z)'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""


def preprocess_file(input_path, section_header="# Session Start"):
    """
    Process a single session note file.
    
    Reads the file, extracts the summary section, and removes Obsidian links.
    
    Args:
        input_path: Path to the input markdown file
        section_header: The header marking the summary section
        
    Returns:
        str: The cleaned summary text
    """
    # Read the raw file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the summary section
    summary = extract_summary(content, section_header)
    
    # Clean up Obsidian links
    cleaned = remove_obsidian_links(summary)
    
    return cleaned


def preprocess_all(config, logger=None):
    """
    Process all session note files from raw to processed folder.
    
    This is the main function that runs the preprocessing pipeline:
    1. Finds all matching files in the raw notes folder
    2. Extracts and cleans the summary from each file
    3. Saves the cleaned summaries to the processed folder
    
    Args:
        config: Configuration dictionary with paths and settings
        logger: Optional logger for tracking progress
        
    Returns:
        list: List of processed file paths
    """
    # Get paths from config
    raw_path = resolve_path(config['paths']['raw_notes'])
    processed_path = resolve_path(config['paths']['processed'])
    input_pattern = config['preprocess']['input_pattern']
    section_header = config['preprocess']['extract_section']
    
    # Create output directory if it doesn't exist
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    input_files = sorted(raw_path.glob(input_pattern))
    
    if not input_files:
        message = f"No files found matching '{input_pattern}' in {raw_path}"
        if logger:
            logger.warning(message)
        else:
            print(f"Warning: {message}")
        return []
    
    # Process each file
    processed_files = []
    total = len(input_files)
    
    for i, input_file in enumerate(input_files, 1):
        # Extract session number from filename for the output name
        # "Session 5.md" -> "Session 5 Summary.md"
        output_name = input_file.stem + " Summary.md"
        output_path = processed_path / output_name
        
        # Process the file
        cleaned_summary = preprocess_file(input_file, section_header)
        
        # Skip empty summaries
        if not cleaned_summary.strip():
            message = f"Skipping {input_file.name} - no content found in '{section_header}' section"
            if logger:
                logger.warning(message)
            else:
                print(f"Warning: {message}")
            continue
        
        # Write the cleaned summary
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_summary)
        
        processed_files.append(output_path)
        
        # Log progress
        message = f"Processed {i}/{total}: {input_file.name} -> {output_name}"
        if logger:
            logger.info(message)
        else:
            print(message)
    
    # Summary
    message = f"Preprocessing complete: {len(processed_files)} files processed"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return processed_files
