# =============================================================================
# Retrieval Module
# =============================================================================
# This module handles searching the Qdrant vector store.
# It supports optional query expansion and reranking to improve search results.

from openai import OpenAI

from src.config import resolve_path, get_secrets
from src.embedding import create_embedder, get_embedding
from src.indexing import create_qdrant_client
from src.reranking import rerank_results


def generate_search_queries(question, config):
    """
    Generate multiple search queries from a single question.
    
    This is called "query expansion" - it creates variations of the question
    to improve search coverage. For example, "Who did Selene kill?" might
    generate: "Selene killed", "Selene slayed", "Selene murdered", etc.
    
    Args:
        question: The original user question
        config: Configuration dictionary with query expansion settings
        
    Returns:
        list: List of search queries (including the original)
    """
    # Check if query expansion is enabled
    if not config.get('query_expansion', {}).get('enabled', False):
        # Query expansion disabled - just return the original question
        return [question]
    
    # Get the model to use for query expansion
    model = config['query_expansion'].get('model', 'gpt-4o-mini')
    
    # Create OpenAI client
    secrets = get_secrets()
    client = OpenAI(api_key=secrets['openai_api_key'])
    
    # Prompt to generate search variations
    prompt = f"""Given this question about a D&D campaign: "{question}"

Generate 5 short search phrases to find relevant information in session summaries.
Use different word variations (past tense, synonyms, etc.).
Return ONLY the search phrases, one per line, nothing else.
One of them has to be the original question!

Example for "Who did Selene kill?":
Who did Selene kill?
Selene killed
Selene slayed
Selene murdered
Selene executed"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Split response into individual queries and clean them up
    queries = response.choices[0].message.content.strip().split("\n")
    queries = [q.strip() for q in queries if q.strip()]
    
    return queries


def search_qdrant(query_embedding, client, config):
    """
    Search the Qdrant database with a query embedding.
    
    Args:
        query_embedding: The embedding vector for the search query
        client: The Qdrant client
        config: Configuration dictionary with search settings
        
    Returns:
        list: List of search results (points with scores)
    """
    collection_name = config['indexing']['collection_name']
    limit = config['retrieval'].get('limit_per_query', 10)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit,
    )
    
    return results.points


def search_with_multiple_queries(queries, config, logger=None):
    """
    Search with multiple queries and combine results.
    
    When using query expansion, this function searches with each generated
    query and combines the results, keeping the highest score for each
    unique chunk.
    
    Args:
        queries: List of search query strings
        config: Configuration dictionary
        logger: Optional logger for tracking progress
        
    Returns:
        list: Combined and sorted search results (highest score first)
    """
    embedder = create_embedder(config)
    client = create_qdrant_client(config)
    model = config['embedding']['model']
    
    # Track best results: chunk_id -> (score, hit)
    all_results = {}
    
    try:
        for query in queries:
            message = f"  Searching for: '{query}'"
            if logger:
                logger.info(message)
            else:
                print(message)
            
            # Get embedding for this query
            query_embedding = get_embedding(query, embedder, model)
            
            # Search Qdrant
            results = search_qdrant(query_embedding, client, config)
            
            # Merge results, keeping highest score per chunk
            for hit in results:
                chunk_id = hit.id
                if chunk_id not in all_results or hit.score > all_results[chunk_id][0]:
                    all_results[chunk_id] = (hit.score, hit)
    
    finally:
        client.close()
    
    # Sort by score (highest first)
    sorted_results = sorted(all_results.values(), key=lambda x: x[0], reverse=True)
    
    return sorted_results


def search(question, config, logger=None):
    """
    Main search function - find relevant chunks for a question.
    
    This is the primary interface for searching. It:
    1. Optionally generates multiple search queries (query expansion)
    2. Searches with each query
    3. Optionally reranks results using a cross-encoder model
    4. Returns the top results
    
    Args:
        question: The user's question
        config: Configuration dictionary
        logger: Optional logger for tracking progress
        
    Returns:
        list: List of result dictionaries, each containing:
            - name: Human-readable chunk name
            - source: Source file name
            - content: The chunk text
            - score: Similarity score (0-1, higher is better)
            - chunk_id: The unique chunk identifier
            - rerank_score: (if reranking enabled) Cross-encoder score
    """
    top_k = config['retrieval'].get('top_k', 5)
    reranking_enabled = config.get('reranking', {}).get('enabled', False)
    
    message = f"Searching for: '{question}'"
    if logger:
        logger.info(message)
    else:
        print(f"\n{'=' * 70}")
        print(f"Question: '{question}'")
        print('=' * 70)
    
    # Step 1: Generate search queries (or just use the original)
    message = "Generating search queries..."
    if logger:
        logger.info(message)
    else:
        print(f"\n{message}")
    
    queries = generate_search_queries(question, config)
    
    if len(queries) > 1:
        message = f"Using {len(queries)} search queries (query expansion enabled)"
    else:
        message = "Using original question as search query"
    
    if logger:
        logger.info(message)
    else:
        print(message)
    
    # Step 2: Search with all queries
    message = "Searching vector database..."
    if logger:
        logger.info(message)
    else:
        print(f"\n{message}")
    
    results = search_with_multiple_queries(queries, config, logger)
    
    # Step 3: Get candidates for reranking
    # If reranking is enabled, get more results to rerank from
    # This allows the reranker to potentially find better results
    if reranking_enabled:
        # Get 3x more candidates for reranking
        num_candidates = min(top_k * 3, len(results))
        candidates = results[:num_candidates]
    else:
        candidates = results[:top_k]
    
    # Format results nicely
    formatted_results = []
    
    for score, hit in candidates:
        formatted_results.append({
            'name': hit.payload['name'],
            'source': hit.payload.get('source_file', 'N/A'),
            'content': hit.payload['content'],
            'score': score,
            'chunk_id': hit.id,
            'session_number': hit.payload.get('session_number'),
            'chunk_number': hit.payload.get('chunk_number'),
        })
    
    # Step 4: Rerank results if enabled
    if reranking_enabled:
        message = "Reranking results with cross-encoder..."
        if logger:
            logger.info(message)
        else:
            print(f"\n{message}")
        
        formatted_results = rerank_results(question, formatted_results, config, logger)
        
        # Take only top_k after reranking
        formatted_results = formatted_results[:top_k]
    
    # Print results if no logger (interactive mode)
    if not logger:
        print(f"\n{'=' * 70}")
        print(f"Top {len(formatted_results)} results:")
        print('=' * 70)
        
        for i, result in enumerate(formatted_results, 1):
            score_str = f"score: {result['score']:.4f}"
            if 'rerank_score' in result:
                score_str += f", rerank: {result['rerank_score']:.4f}"
            
            print(f"\n{i}. {result['name']} ({score_str})")
            print(f"   Source: {result['source']}")
            print(f"   Content:\n   {result['content'][:200]}...")
    
    return formatted_results
