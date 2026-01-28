# =============================================================================
# Embedding Module
# =============================================================================
# This module generates vector embeddings for text using OpenAI's API.
# Embeddings are numerical representations that capture the meaning of text.

from openai import OpenAI

from src.config import get_secrets


def create_embedder(config):
    """
    Create an OpenAI client for generating embeddings.
    
    Args:
        config: Configuration dictionary (not currently used, but kept for consistency)
        
    Returns:
        OpenAI: An initialized OpenAI client
        
    Raises:
        FileNotFoundError: If secrets.yaml doesn't exist
    """
    secrets = get_secrets()
    api_key = secrets.get('openai_api_key')
    
    if not api_key:
        raise ValueError("OpenAI API key not found in secrets.yaml")
    
    return OpenAI(api_key=api_key)


def get_embedding(text, embedder, model):
    """
    Generate an embedding vector for a piece of text.
    
    An embedding is a list of numbers (a vector) that represents the meaning
    of the text. Similar texts will have similar embeddings.
    
    Args:
        text: The text to embed
        embedder: An OpenAI client instance
        model: The embedding model to use (e.g., "text-embedding-3-small")
        
    Returns:
        list: A list of floats representing the embedding vector
    """
    response = embedder.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def embed_chunks(chunks, config, logger=None):
    """
    Generate embeddings for a list of chunks.
    
    This function takes chunks (from the chunking module) and adds an
    'embedding' field to each one containing its vector representation.
    
    Args:
        chunks: List of chunk dictionaries (each must have a 'content' field)
        config: Configuration dictionary with embedding settings
        logger: Optional logger for tracking progress
        
    Returns:
        list: The same chunks, but with an 'embedding' field added to each
        
    Note:
        This function modifies the chunks in place and also returns them.
        It makes one API call per chunk, so it can take a while for many chunks.
    """
    # Get the embedding model from config
    model = config['embedding']['model']
    
    # Create the OpenAI client
    embedder = create_embedder(config)
    
    total = len(chunks)
    
    message = f"Generating embeddings for {total} chunks using {model}..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    for i, chunk in enumerate(chunks, 1):
        # Get the embedding for this chunk's content
        embedding = get_embedding(chunk['content'], embedder, model)
        
        # Add it to the chunk
        chunk['embedding'] = embedding
        
        # Progress update every 10 chunks
        if i % 10 == 0 or i == total:
            message = f"  Embedded {i}/{total} chunks"
            if logger:
                logger.info(message)
            else:
                print(message)
    
    message = f"Embedding complete: {total} chunks embedded"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return chunks


def get_embedding_dimension(config):
    """
    Get the dimension (size) of embeddings for the configured model.
    
    Different embedding models produce vectors of different sizes:
    - text-embedding-3-small: 1536 dimensions
    - text-embedding-3-large: 3072 dimensions
    
    Args:
        config: Configuration dictionary with embedding settings
        
    Returns:
        int: The embedding dimension for the configured model
    """
    model = config['embedding']['model']
    
    # Known dimensions for OpenAI embedding models
    dimensions = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }
    
    if model in dimensions:
        return dimensions[model]
    
    # If unknown model, we'll need to generate a sample embedding to find out
    # This is a fallback for future models
    embedder = create_embedder(config)
    sample = get_embedding("test", embedder, model)
    return len(sample)
