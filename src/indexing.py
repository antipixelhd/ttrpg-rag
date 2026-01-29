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


def collection_exists(client, collection_name):
    """
    Check if a collection exists in Qdrant.
    
    Args:
        client: The Qdrant client
        collection_name: Name of the collection to check
        
    Returns:
        bool: True if collection exists, False otherwise
    """
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def get_existing_chunk_ids(client, collection_name):
    """
    Get all chunk IDs that are already stored in the collection.
    
    This allows us to skip re-embedding chunks that are already indexed.
    
    Args:
        client: The Qdrant client
        collection_name: Name of the collection
        
    Returns:
        set: Set of chunk IDs (integers) that exist in the collection
    """
    try:
        # Scroll through all points to get their IDs
        # We use scroll instead of search to get all points efficiently
        existing_ids = set()
        offset = None
        
        while True:
            # Get a batch of points (just IDs, no vectors or payloads)
            result = client.scroll(
                collection_name=collection_name,
                limit=100,  # Process 100 at a time
                with_payload=False,  # We don't need the payload
                with_vectors=False,  # We don't need the vectors
                offset=offset,
            )
            
            points, next_offset = result
            
            # Add IDs to our set
            for point in points:
                existing_ids.add(point.id)
            
            # If no more points, we're done
            if next_offset is None:
                break
            
            offset = next_offset
        
        return existing_ids
    
    except Exception as e:
        print(f"Warning: Could not get existing chunk IDs: {e}")
        return set()


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
    if collection_exists(client, collection_name):
        collection_info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists with {collection_info.points_count} points")
        return False
    
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
    2. Checks if collection exists and what's already indexed
    3. Generates embeddings only for new chunks
    4. Uploads only new chunks to Qdrant
    
    Args:
        config: Configuration dictionary
        logger: Optional logger for tracking progress
        
    Returns:
        dict: Summary of the indexing operation containing:
            - chunks: The list of all chunks (with embeddings)
            - collection_name: Name of the Qdrant collection
            - count: Total number of chunks in collection
            - new_count: Number of new chunks added in this run
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
    
    # Step 2: Check if collection exists and what's already indexed
    message = "Step 2: Checking existing collection..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    client = create_qdrant_client(config)
    
    try:
        # Check if collection exists
        if collection_exists(client, collection_name):
            # Get existing chunk IDs
            existing_ids = get_existing_chunk_ids(client, collection_name)
            
            message = f"Collection exists with {len(existing_ids)} chunks already indexed"
            if logger:
                logger.info(message)
            else:
                print(message)
            
            # Filter out chunks that are already indexed
            new_chunks = []
            for chunk in chunks:
                chunk_id = create_chunk_id(chunk)
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
            
            message = f"Found {len(new_chunks)} new chunks to index"
            if logger:
                logger.info(message)
            else:
                print(message)
            
            # If no new chunks, we're done
            if not new_chunks:
                message = "All chunks are already indexed. Nothing to do!"
                if logger:
                    logger.info(message)
                else:
                    print(message)
                
                return {
                    'chunks': chunks,
                    'collection_name': collection_name,
                    'count': len(chunks),
                    'new_count': 0,
                }
        else:
            # Collection doesn't exist - all chunks are new
            new_chunks = chunks
            message = f"Collection does not exist. Will create and index all {len(new_chunks)} chunks"
            if logger:
                logger.info(message)
            else:
                print(message)
        
        # Step 3: Generate embeddings only for new chunks
        message = f"Step 3: Generating embeddings for {len(new_chunks)} new chunks..."
        if logger:
            logger.info(message)
        else:
            print(message)
        
        if new_chunks:
            new_chunks = embed_chunks(new_chunks, config, logger)
        
        # Step 4: Create collection if needed
        message = "Step 4: Setting up collection..."
        if logger:
            logger.info(message)
        else:
            print(message)
        
        if not collection_exists(client, collection_name):
            # Get embedding dimension from the first chunk
            vector_size = len(new_chunks[0]['embedding'])
            create_collection(client, collection_name, vector_size)
        
        # Step 5: Upload new chunks
        if new_chunks:
            message = f"Step 5: Uploading {len(new_chunks)} new chunks..."
            if logger:
                logger.info(message)
            else:
                print(message)
            
            upload_chunks(client, collection_name, new_chunks, logger)
        
    finally:
        # Always close the client properly
        client.close()
    
    message = f"Indexing complete: {len(chunks)} total chunks ({len(new_chunks)} new)"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return {
        'chunks': chunks,
        'collection_name': collection_name,
        'count': len(chunks),
        'new_count': len(new_chunks),
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
