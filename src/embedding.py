
from openai import OpenAI

from src.config import get_secrets

def create_embedder():
    secrets = get_secrets()
    api_key = secrets.get('openai_api_key')
    
    if not api_key:
        raise ValueError("OpenAI API key not found in secrets.yaml")
    
    return OpenAI(api_key=api_key)

def get_embedding(text, embedder, model):
    response = embedder.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def embed_chunks(chunks, config, logger=None):
    model = config['embedding']['model']
    
    embedder = create_embedder(config)
    
    total = len(chunks)
    
    message = f"Generating embeddings for {total} chunks using {model}..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    for i, chunk in enumerate(chunks, 1):
        embedding = get_embedding(chunk['content'], embedder, model)
        
        chunk['embedding'] = embedding
        
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

