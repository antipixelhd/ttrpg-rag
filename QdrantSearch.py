import os
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Setup paths - find where the qdrant storage folder is
script_dir = Path(__file__).parent.parent
qdrant_storage = script_dir / 'qdrant_storage'

# Create OpenAI client for generating embeddings and query variations
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Create Qdrant client for searching the database
qdrant_client = QdrantClient(path=str(qdrant_storage))


def get_embedding(text):
    """
    Turn text into numbers (embeddings) that the computer can compare.
    """
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def generate_search_queries(question):
    """
    Use GPT-4o-mini to generate multiple search queries from a question.
    This is what OpenAI Vector Store does behind the scenes.
    
    For example, "Who did Selene kill?" becomes:
    - "Selene kill"
    - "Selene killed"
    - "Selene slayed"
    - "Selene murdered"
    - "Selene executed"
    """
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

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Split response into individual search queries
    queries = response.choices[0].message.content.strip().split("\n")
    
    # Clean up the queries (remove empty lines, extra spaces)
    queries = [q.strip() for q in queries if q.strip()]
    queries = [question]
    return queries


def search_with_queries(queries, limit_per_query=10):
    """
    Search Qdrant with multiple queries and combine results.
    Keeps the best score for each unique chunk.
    """
    all_results = {}  # key = chunk id, value = (score, hit)
    
    for query in queries:
        # Get embedding for this query
        query_embedding = get_embedding(query)
        
        # Search Qdrant
        results = qdrant_client.query_points(
            collection_name="rpg_sessions",
            query=query_embedding,
            limit=limit_per_query,
        )
        
        # Add results (keep highest score if we see same chunk twice)
        for hit in results.points:
            chunk_id = hit.id
            if chunk_id not in all_results or hit.score > all_results[chunk_id][0]:
                all_results[chunk_id] = (hit.score, hit)
    
    # Sort by score (highest first)
    sorted_results = sorted(all_results.values(), key=lambda x: x[0], reverse=True)
    
    return sorted_results


def search(question, top_k=5):
    """
    Main search function - mimics OpenAI Vector Store behavior.
    
    1. Generate multiple search queries from the question
    2. Search with each query
    3. Return top results
    
    Args:
        question: The user's question
        top_k: How many results to return (default 5)
    
    Returns:
        List of chunks with their content and metadata
    """
    print("\n" + "=" * 70)
    print(f"🔍 Question: '{question}'")
    print("=" * 70)
    
    # Step 1: Generate search queries using GPT-4o-mini
    print("\n📝 Generating search queries...")
    queries = generate_search_queries(question)
    print(f"   Search queries: {queries}")
    
    # Step 2: Search with all queries
    print("\n🔎 Searching with each query...")
    results = search_with_queries(queries)
    
    # Step 3: Return top results
    top_results = results[:top_k]
    
    print(f"\n{'=' * 70}")
    print(f"📚 Top {len(top_results)} results:")
    print('=' * 70)
    
    # Print and collect results
    retrieved_chunks = []
    
    for i in range(len(top_results)):
        score, hit = top_results[i]
        
        name = hit.payload['name']
        source = hit.payload.get('source_file', 'N/A')
        content = hit.payload['content']
        
        print(f"\n{i + 1}. {name} (score: {score:.4f})")
        print(f"   Source: {source}")
        print(f"   Content:\n   {content}")
        
        # Add to our list of retrieved chunks
        retrieved_chunks.append({
            'name': name,
            'source': source,
            'content': content,
            'score': score
        })
    
    return retrieved_chunks


# This runs when you execute the script directly
if __name__ == '__main__':
    try:
        # Test the search
        results = search("Who did Selene kill?", top_k=10)
        
        # You can also try other questions:
        # results = search("What happened with the dragon?")
        # results = search("Tell me about treasure")
        
    finally:
        # Clean up - close the database connection properly
        qdrant_client.close()