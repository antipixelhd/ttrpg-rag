
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

def embed_chunks(chunks, config, verbose=False):
    model = config['embedding']['model']
    
    embedder = create_embedder()
    
    total = len(chunks)
    
    print(f"Generating embeddings for {total} chunks using {model}...")
    
    for i, chunk in enumerate(chunks, 1):
        embedding = get_embedding(chunk['content'], embedder, model)
        
        chunk['embedding'] = embedding
        
        if verbose and (i % 10 == 0 or i == total):
            print(f"  Embedded {i}/{total} chunks")
    
    print(f"Embedding complete: {total} chunks embedded")
    
    return chunks
