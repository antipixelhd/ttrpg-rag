# =============================================================================
# Indexing Module
# =============================================================================
# This module handles uploading chunks with embeddings to Qdrant vector store.
# Qdrant is a vector database that allows fast similarity search.

from qdrant_client import QdrantClient, models

from src.config import resolve_path
from src.chunking import get_all_chunks, create_chunk_id
from src.embedding import embed_chunks, get_embedding_dimension


def create_qdrant_client(config):
    """
    Create a Qdrant client connected to local storage.
    
    Args:
        config: Configuration dictionary with Qdrant storage path
        
    Returns:
        QdrantClient: A client connected to the local Qdrant database
    """
    storage_path = resolve_path(config['paths']['qdrant_storage'])
    
    # Create the storage directory if it doesn't exist
    storage_path.mkdir(parents=True, exist_ok=True)
    
    return QdrantClient(path=str(storage_path))


def create_collection(client, collection_name, vector_size):
    """
    Create a new collection in Qdrant for storing vectors.
    
    A collection is like a table in a regular database. It stores vectors
    along with their metadata (payload).
    
    Args:
        client: The Qdrant client
        collection_name: Name for the new collection
        vector_size: Dimension of the embedding vectors (e.g., 1536)
        
    Returns:
        bool: True if collection was created, False if it already existed
    """
    # Check if collection already exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists with {collection_info.points_count} points")
        return False
    except Exception:
        # Collection doesn't exist, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,  # Use cosine similarity
            ),
        )
        print(f"Created new collection '{collection_name}'")
        return True


def upload_chunks(client, collection_name, chunks_with_embeddings, logger=None):
    """
    Upload chunks with their embeddings to Qdrant.
    
    Each chunk becomes a "point" in Qdrant with:
    - id: A unique identifier based on session and chunk number
    - vector: The embedding vector
    - payload: Metadata (session number, content, source file, etc.)
    
    Args:
        client: The Qdrant client
        collection_name: Name of the collection to upload to
        chunks_with_embeddings: List of chunks (each must have an 'embedding' field)
        logger: Optional logger for tracking progress
        
    Returns:
        int: Number of points uploaded
    """
    # Build the list of points to upload
    points = []
    
    for chunk in chunks_with_embeddings:
        # Create a unique ID for this chunk
        point_id = create_chunk_id(chunk)
        
        # Get the embedding vector
        vector = chunk['embedding']
        
        # Prepare the payload (metadata) - everything except the embedding
        payload = {
            'session_number': chunk['session_number'],
            'chunk_number': chunk['chunk_number'],
            'content': chunk['content'],
            'name': chunk['name'],
            'source_file': chunk['source_file'],
        }
        
        points.append(models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        ))
    
    # Upload all points at once
    client.upload_points(
        collection_name=collection_name,
        points=points,
    )
    
    message = f"Uploaded {len(points)} chunks to collection '{collection_name}'"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return len(points)


def index_all(config, logger=None):
    """
    Run the full indexing pipeline: chunk -> embed -> upload.
    
    This is the main function for indexing. It:
    1. Loads and chunks all processed summaries
    2. Generates embeddings for each chunk
    3. Uploads everything to Qdrant
    
    Args:
        config: Configuration dictionary
        logger: Optional logger for tracking progress
        
    Returns:
        dict: Summary of the indexing operation containing:
            - chunks: The list of chunks (with embeddings)
            - collection_name: Name of the Qdrant collection
            - count: Number of chunks indexed
    """
    collection_name = config['indexing']['collection_name']
    
    # Step 1: Get all chunks
    message = "Step 1: Loading and chunking summaries..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    chunks = get_all_chunks(config, logger)
    
    if not chunks:
        message = "No chunks to index. Run preprocessing first."
        if logger:
            logger.error(message)
        else:
            print(f"Error: {message}")
        return None
    
    # Step 2: Generate embeddings
    message = "Step 2: Generating embeddings..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    chunks = embed_chunks(chunks, config, logger)
    
    # Step 3: Upload to Qdrant
    message = "Step 3: Uploading to Qdrant..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    # Create client and collection
    client = create_qdrant_client(config)
    
    try:
        # Get embedding dimension from the first chunk
        vector_size = len(chunks[0]['embedding'])
        
        # Create collection (or check if it exists)
        created = create_collection(client, collection_name, vector_size)
        
        if created:
            # Only upload if we just created the collection
            upload_chunks(client, collection_name, chunks, logger)
        else:
            message = "Skipping upload - collection already has data. Delete qdrant_storage to re-index."
            if logger:
                logger.warning(message)
            else:
                print(f"Warning: {message}")
    
    finally:
        # Always close the client properly
        client.close()
    
    message = f"Indexing complete: {len(chunks)} chunks in collection '{collection_name}'"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return {
        'chunks': chunks,
        'collection_name': collection_name,
        'count': len(chunks),
    }


def delete_collection(config, logger=None):
    """
    Delete the Qdrant collection (useful for re-indexing).
    
    Args:
        config: Configuration dictionary
        logger: Optional logger for tracking progress
        
    Returns:
        bool: True if collection was deleted
    """
    collection_name = config['indexing']['collection_name']
    
    client = create_qdrant_client(config)
    
    try:
        client.delete_collection(collection_name)
        message = f"Deleted collection '{collection_name}'"
        if logger:
            logger.info(message)
        else:
            print(message)
        return True
    except Exception as e:
        message = f"Could not delete collection: {e}"
        if logger:
            logger.warning(message)
        else:
            print(f"Warning: {message}")
        return False
    finally:
        client.close()
