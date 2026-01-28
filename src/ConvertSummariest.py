
import os
import re

# Remove Obsidian links and replace with aliases
def remove_obsidian_links(text):
    """
    Replace Obsidian links with their aliases or link text.
    - [[link|alias]] -> alias
    - [[link]] -> link
    """
    # Replace [[link|alias]] with alias
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    # Replace [[link]] with link
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    return text

def extract_summary(text):
    """
    Extract only the content from "# Session Start" to the next "#" heading.
    Returns the extracted section or empty string if not found.
    """
    # Find "# Session Start" section
    match = re.search(r'# Session Start\s*(.*?)(?=\n#|\Z)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""




# Process local files from Source folder
source_folder = "Source"



file_count = len(os.listdir(source_folder))

for i in range(file_count-1):
    filename = f"Session {i+1}.md"
    filepath = os.path.join(source_folder, f"Session {i+2}.md")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract only the Session Start section
    session_start = extract_summary(content)
    
    # Remove Obsidian links from the extracted section
    cleaned_summary = remove_obsidian_links(session_start)
    
    # Create output filename
    output_filename = filename.replace(".md", " Summary.md")
    output_path = os.path.join("summaries", output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_summary)
    
    print(f"Processed {i+1}/{file_count}: {filename}")