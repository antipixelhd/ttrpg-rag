
from sentence_transformers import CrossEncoder

_reranker_model = None
_reranker_model_name = None

def load_reranker(config, verbose=False):
    global _reranker_model, _reranker_model_name
    
    model_name = config['reranking']['model']
    
    if _reranker_model is not None and _reranker_model_name == model_name:
        return _reranker_model
    
    if verbose:
        print(f"Loading reranker model: {model_name}")
    _reranker_model = CrossEncoder(model_name)
    _reranker_model_name = model_name
    
    return _reranker_model

def rerank_results(question, results, config, verbose=False):
    if not results:
        return results
    if verbose:
        print(f"Reranking {len(results)} results...")

    model = load_reranker(config, verbose)
    pairs = [(question, result['content']) for result in results]
    
    scores = model.predict(pairs)
    
    for i, result in enumerate(results):
        result['rerank_score'] = float(scores[i])
    
    reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    if verbose:
        print("Reranking complete")
    
    return reranked_results
