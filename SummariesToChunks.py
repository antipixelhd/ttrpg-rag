import os
import re
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from qdrant_client import models, QdrantClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


def extract_session_number(filename: str) -> int:
    """Extract session number from filename like 'Session 20 Summary.md'"""
    match = re.search(r'Session (\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def chunk_summary_file(file_path: str) -> List[Dict[str, str]]:
    """
    Chunk a summary file by top-level bullet points.
    Each chunk includes the session number and chunk number for ordering.
    
    Args:
        file_path: Path to the summary markdown file
        
    Returns:
        List of dictionaries containing chunked content with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    filename = os.path.basename(file_path)
    session_number = extract_session_number(filename)
    
    chunks = []
    current_chunk_lines = []
    chunk_number = 0
    
    for line in lines:
        # Check if this is a top-level bullet point (starts with '-' at position 0)
        if line.startswith('-'):
            # If we have accumulated lines, save the previous chunk
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
                current_chunk_lines = []
            
            # Start a new chunk with this top-level bullet
            current_chunk_lines.append(line)
        elif line.strip():  # Non-empty line (nested content or continuation)
            # Add to current chunk if we're inside one
            if current_chunk_lines:
                current_chunk_lines.append(line)
        # Empty lines are added to maintain structure if we're in a chunk
        elif current_chunk_lines:
            current_chunk_lines.append(line)
    
    # Don't forget the last chunk
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


def get_all_chunks(summaries_dir: str) -> List[Dict[str, str]]:
    """
    Load all summary files and create chunks.
    
    Args:
        summaries_dir: Directory containing summary markdown files
        
    Returns:
        List of all chunks with metadata
    """
    summaries_path = Path(summaries_dir)
    summary_files = list(summaries_path.glob('Session * Summary.md'))
    summary_files.sort(key=lambda x: extract_session_number(x.name))
    
    all_chunks = []
    for file_path in summary_files:
        chunks = chunk_summary_file(str(file_path))
        all_chunks.extend(chunks)
    
    return all_chunks

def get_embedding(text: str) -> List[float]:
        response = encoder.embeddings.create(
            input=text,
            model=embedding_model
        )
        return response.data[0].embedding

if __name__ == '__main__':
    # Setup paths
    script_dir = Path(__file__).parent.parent
    summaries_dir = script_dir / 'summaries'
    qdrant_storage = script_dir / 'qdrant_storage'
    
    print("=" * 70)
    print("RPG Session Summary Search with Qdrant")
    print("=" * 70)
    
    # 1. Initialize OpenAI client for embeddings
    print("\n1. Initializing OpenAI encoder...")
    encoder = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    embedding_model = "text-embedding-3-large"
    

    # 2. Load the dataset (RPG session summaries)
    print("2. Loading session summaries and creating chunks...")
    documents = get_all_chunks(str(summaries_dir))
    print(f"   Loaded {len(documents)} chunks from session summaries")
    
    # 3. Initialize Qdrant client with disk storage
    print("3. Connecting to Qdrant (disk storage)...")
    qdrant_storage.mkdir(exist_ok=True)
    client = QdrantClient(path=str(qdrant_storage))
    
    try:
        # 4. Create collection (or skip if it already exists)
        print("4. Setting up collection 'rpg_sessions'...")
        
        # Check if collection already exists
        try:
            collection_info = client.get_collection("rpg_sessions")
            print(f"   Collection already exists with {collection_info.points_count} points")
            print("   Skipping data upload. Delete 'qdrant_storage' folder to recreate.")
        except Exception:
            # Collection doesn't exist, create it
            print("   Creating new collection...")
            
            # Get embedding dimension from first chunk
            sample_embedding = get_embedding(documents[0]['content'])
            vector_size = len(sample_embedding)
            
            client.create_collection(
                collection_name="rpg_sessions",
                vectors_config=models.VectorParams(
                    size=vector_size,  # OpenAI text-embedding-3-small has 1536 dimensions
                    distance=models.Distance.COSINE,
                ),
            )
            
            # 5. Upload data to collection
            print(f"   Generating embeddings and uploading {len(documents)} chunks...")
            print("   (This may take a moment...)")
            
            client.upload_points(
                collection_name="rpg_sessions",
                points=[
                    models.PointStruct(
                        id=idx,
                        vector=get_embedding(doc["content"]),
                        payload=doc
                    )
                    for idx, doc in enumerate(documents)
                ],
            )
            
            print(f"   ✓ Successfully uploaded all chunks!")
        
        print("\n" + "=" * 70)
        print(f"✓ Embeddings stored in: {qdrant_storage}")
        print("=" * 70)
    
    finally:
        # Properly close the client to avoid cleanup errors
        client.close()
    
