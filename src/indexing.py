
from qdrant_client import QdrantClient, models

from src.config import resolve_path
from src.chunking import get_all_chunks, create_chunk_id
from src.embedding import embed_chunks

def create_qdrant_client(config):
    storage_path = resolve_path(config['paths']['qdrant_storage'])
    
    storage_path.mkdir(parents=True, exist_ok=True)
    
    return QdrantClient(path=str(storage_path))

def collection_exists(client, collection_name):
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

def get_existing_chunk_ids(client, collection_name):
    try:
        existing_ids = set()
        offset = None
        
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            
            points, next_offset = result
            
            for point in points:
                existing_ids.add(point.id)
            
            if next_offset is None:
                break
            
            offset = next_offset
        
        return existing_ids
    
    except Exception as e:
        print(f"Warning: Could not get existing chunk IDs: {e}")
        return set()

def create_collection(client, collection_name, vector_size, verbose=False):
    if collection_exists(client, collection_name):
        collection_info = client.get_collection(collection_name)
        if verbose:
            print(f"Collection '{collection_name}' already exists with {collection_info.points_count} points")
        return False
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    if verbose:
        print(f"Created new collection '{collection_name}'")
    return True

def upload_chunks(client, collection_name, chunks_with_embeddings, verbose=False):
    points = []
    
    for chunk in chunks_with_embeddings:
        point_id = create_chunk_id(chunk)
        
        vector = chunk['embedding']
        
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
    
    client.upload_points(
        collection_name=collection_name,
        points=points,
    )
    
    if verbose:
        print(f"Uploaded {len(points)} chunks to collection '{collection_name}'")
    
    return len(points)

def index_all(config, verbose=False):
    collection_name = config['indexing']['collection_name']
    
    print("Step 1: Loading and chunking summaries...")
    
    chunks = get_all_chunks(config, verbose)
    
    if not chunks:
        print("Error: No chunks to index. Run preprocessing first.")
        return None
    
    print("Step 2: Checking existing collection...")
    
    client = create_qdrant_client(config)
    
    try:
        if collection_exists(client, collection_name):
            existing_ids = get_existing_chunk_ids(client, collection_name)
            
            if verbose:
                print(f"Collection exists with {len(existing_ids)} chunks already indexed")
            
            new_chunks = []
            for chunk in chunks:
                chunk_id = create_chunk_id(chunk)
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
            
            if verbose:
                print(f"Found {len(new_chunks)} new chunks to index")
            
            if not new_chunks:
                print("All chunks are already indexed. Nothing to do!")
                
                return {
                    'chunks': chunks,
                    'collection_name': collection_name,
                    'count': len(chunks),
                    'new_count': 0,
                }
        else:
            new_chunks = chunks
            if verbose:
                print(f"Collection does not exist. Will create and index all {len(new_chunks)} chunks")
        
        print(f"Step 3: Generating embeddings for {len(new_chunks)} new chunks...")
        
        if new_chunks:
            new_chunks = embed_chunks(new_chunks, config, verbose)
        
        print("Step 4: Setting up collection...")
        
        if not collection_exists(client, collection_name):
            vector_size = len(new_chunks[0]['embedding'])
            create_collection(client, collection_name, vector_size, verbose)
        
        if new_chunks:
            print(f"Step 5: Uploading {len(new_chunks)} new chunks...")
            
            upload_chunks(client, collection_name, new_chunks, verbose)
        
    finally:
        client.close()
    
    print(f"Indexing complete: {len(chunks)} total chunks ({len(new_chunks)} new)")
    
    return {
        'chunks': chunks,
        'collection_name': collection_name,
        'count': len(chunks),
        'new_count': len(new_chunks),
    }

def delete_collection(config, verbose=False):
    import shutil
    
    storage_path = resolve_path(config['paths']['qdrant_storage'])
    
    try:
        if storage_path.exists():
            print(f"Deleting Qdrant storage: {storage_path}")
            
            shutil.rmtree(storage_path)
            
            print("Qdrant storage deleted successfully")
            return True
        else:
            if verbose:
                print(f"Qdrant storage does not exist: {storage_path}")
            return True
            
    except Exception as e:
        print(f"Error: Error deleting Qdrant storage: {e}")
        return False
