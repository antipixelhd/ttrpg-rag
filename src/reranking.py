
from sentence_transformers import CrossEncoder

_reranker_model = None
_reranker_model_name = None

def load_reranker(config):
    global _reranker_model, _reranker_model_name
    
    model_name = config['reranking']['model']
    device = config['reranking'].get('device', 'cpu')
    
    if _reranker_model is not None and _reranker_model_name == model_name:
        return _reranker_model
    
    print(f"Loading reranker model: {model_name} (device: {device})")
    _reranker_model = CrossEncoder(model_name, device=device)
    _reranker_model_name = model_name
    
    return _reranker_model

def rerank_results(question, results, config, logger=None):
    if not results:
        return results
    
    if not config.get('reranking', {}).get('enabled', False):
        message = "Reranking is disabled"
        if logger:
            logger.info(message)
        return results
    
    message = f"Reranking {len(results)} results..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    model = load_reranker(config)
    
    pairs = [(question, result['content']) for result in results]
    
    scores = model.predict(pairs)
    
    for i, result in enumerate(results):
        result['rerank_score'] = float(scores[i])
    
    reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    message = "Reranking complete"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return reranked_results

def rerank_with_details(question, results, config, logger=None):
    original_order = [r.get('chunk_id', r.get('name', i)) for i, r in enumerate(results)]
    original_scores = [r.get('score', 0) for r in results]
    
    reranked = rerank_results(question, results, config, logger)
    
    new_order = [r.get('chunk_id', r.get('name', i)) for i, r in enumerate(reranked)]
    rerank_scores = [r.get('rerank_score', 0) for r in reranked]
    
    rank_changes = []
    for i, chunk_id in enumerate(new_order):
        old_rank = original_order.index(chunk_id) if chunk_id in original_order else -1
        new_rank = i
        rank_changes.append({
            'chunk_id': chunk_id,
            'old_rank': old_rank + 1,
            'new_rank': new_rank + 1,
            'change': old_rank - new_rank,
        })
    
    details = {
        'enabled': config.get('reranking', {}).get('enabled', False),
        'model': config.get('reranking', {}).get('model', 'N/A'),
        'original_order': original_order,
        'new_order': new_order,
        'original_scores': original_scores,
        'rerank_scores': rerank_scores,
        'rank_changes': rank_changes,
    }
    
    return reranked, details
