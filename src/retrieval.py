
from openai import OpenAI

from src.config import resolve_path, get_secrets
from src.embedding import create_embedder, get_embedding
from src.indexing import create_qdrant_client

def generate_search_queries(question, config):
    if not config.get('query_expansion', {}).get('enabled', False):
        return [question]
    
    model = config['query_expansion'].get('model', 'gpt-4o-mini')
    
    secrets = get_secrets()
    client = OpenAI(api_key=secrets['openai_api_key'])
    
    prompt = f"""Given this question about a D&D campaign: "{question}"
        Generate 4 short search phrases using different word variations (past tense, synonyms, etc.), but be sure to keep the original meaning of the question!
        Return ONLY the search phrases, one per line, nothing else.
        """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    queries = response.choices[0].message.content.strip().split("\n")
    queries = [q.strip() for q in queries if q.strip()]
    queries.append(question)
    return queries

def search_qdrant(query_embedding, client, config):
    collection_name = config['indexing']['collection_name']
    limit = config['retrieval'].get('limit_per_query', 10)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit,
    )
    
    return results.points

def search_with_multiple_queries(queries, config, verbose=False):
    embedder = create_embedder()
    client = create_qdrant_client(config)
    model = config['embedding']['model']
    
    chunk_scores = {}
    chunk_hits = {}
    
    try:
        for query in queries:
            print(f"  Searching for: '{query}'")
            
            query_embedding = get_embedding(query, embedder, model)
            
            results = search_qdrant(query_embedding, client, config)
            
            for hit in results:
                chunk_id = hit.id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = []
                    chunk_hits[chunk_id] = hit
                chunk_scores[chunk_id].append(hit.score)
    
    finally:
        client.close()
    
    all_results = {}
    for chunk_id, scores in chunk_scores.items():
        avg_score = sum(scores) / len(scores)
        all_results[chunk_id] = (avg_score, chunk_hits[chunk_id])
    
    sorted_results = sorted(all_results.values(), key=lambda x: x[0], reverse=True)
    
    return sorted_results

def search(question, config, verbose=False):
    top_k = config['retrieval'].get('top_k', 5)
    reranking_enabled = config.get('reranking', {}).get('enabled', False)
    
    queries = generate_search_queries(question, config)
    
    if verbose:
        if len(queries) > 1:
            print(f"Using {len(queries)} search queries (query expansion enabled)")
        else:
            print("Using original question as search query")
    
    print("Searching vector database...")
    
    results = search_with_multiple_queries(queries, config, verbose)
    
    formatted_results = []
    
    for score, hit in results:
        formatted_results.append({
            'name': hit.payload['name'],
            'source': hit.payload.get('source_file', 'N/A'),
            'content': hit.payload['content'],
            'score': score,
            'chunk_id': hit.id,
            'session_number': hit.payload.get('session_number'),
            'chunk_number': hit.payload.get('chunk_number'),
        })
    
    if reranking_enabled:
        print("Reranking results with cross-encoder...")
        
        from src.reranking import rerank_results
        
        formatted_results = rerank_results(question, formatted_results, config, verbose)
        
    formatted_results = formatted_results[:top_k]
    
    if verbose:
        print(f"\nTop {len(formatted_results)} results:")
        for i, result in enumerate(formatted_results, 1):
            score_str = f"score: {result['score']:.4f}"
            if 'rerank_score' in result:
                score_str += f", rerank: {result['rerank_score']:.4f}"
            
            print(f"\n  {i}. {result['name']} ({score_str})")
            print(f"     Source: {result['source']}")
            print(f"     Chunk ID: {result['chunk_id']}")
            print(f"     Content preview: {result['content'][:200]}...")
    
    print(f"\nSearch complete: {len(formatted_results)} results returned")
    
    return formatted_results
