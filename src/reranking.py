# =============================================================================
# Reranking Module
# =============================================================================
# This module provides reranking functionality using cross-encoder models.
# Reranking takes the initial retrieval results and re-scores them using
# a more powerful model that looks at both query and document together.

from sentence_transformers import CrossEncoder


# Global variable to cache the model (loading is expensive)
_reranker_model = None
_reranker_model_name = None


def load_reranker(config):
    """
    Load the cross-encoder model for reranking.
    
    The model is cached globally to avoid reloading it on every search.
    This significantly speeds up repeated searches.
    
    Args:
        config: Configuration dictionary with reranking settings
        
    Returns:
        CrossEncoder: The loaded cross-encoder model
    """
    global _reranker_model, _reranker_model_name
    
    model_name = config['reranking']['model']
    device = config['reranking'].get('device', 'cpu')
    
    # Check if we already have the model loaded
    if _reranker_model is not None and _reranker_model_name == model_name:
        return _reranker_model
    
    # Load the model
    print(f"Loading reranker model: {model_name} (device: {device})")
    _reranker_model = CrossEncoder(model_name, device=device)
    _reranker_model_name = model_name
    
    return _reranker_model


def rerank_results(question, results, config, logger=None):
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        question: The user's question
        results: List of result dictionaries from initial retrieval
                 Each must have a 'content' field
        config: Configuration dictionary with reranking settings
        logger: Optional logger for tracking progress
        
    Returns:
        list: The same results, but reordered by reranking score.
              Each result gets a 'rerank_score' field added.
    """
    if not results:
        return results
    
    # Check if reranking is enabled
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
    
    # Load the model
    model = load_reranker(config)
    
    # Prepare query-document pairs for the cross-encoder
    # The cross-encoder takes (query, document) pairs and scores them
    pairs = [(question, result['content']) for result in results]
    
    # Get reranking scores
    scores = model.predict(pairs)
    
    # Add scores to results
    for i, result in enumerate(results):
        result['rerank_score'] = float(scores[i])
    
    # Sort by reranking score (highest first)
    reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    message = "Reranking complete"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return reranked_results


def rerank_with_details(question, results, config, logger=None):
    """
    Rerank results and return additional details about the reranking.
    
    This is useful for analysis and debugging - it shows how much
    the rankings changed after reranking.
    
    Args:
        question: The user's question
        results: List of result dictionaries
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        tuple: (reranked_results, details_dict)
            - reranked_results: The reordered results
            - details_dict: Information about the reranking process
    """
    # Store original order
    original_order = [r.get('chunk_id', r.get('name', i)) for i, r in enumerate(results)]
    original_scores = [r.get('score', 0) for r in results]
    
    # Perform reranking
    reranked = rerank_results(question, results, config, logger)
    
    # Get new order
    new_order = [r.get('chunk_id', r.get('name', i)) for i, r in enumerate(reranked)]
    rerank_scores = [r.get('rerank_score', 0) for r in reranked]
    
    # Calculate how much rankings changed
    rank_changes = []
    for i, chunk_id in enumerate(new_order):
        old_rank = original_order.index(chunk_id) if chunk_id in original_order else -1
        new_rank = i
        rank_changes.append({
            'chunk_id': chunk_id,
            'old_rank': old_rank + 1,  # 1-indexed for readability
            'new_rank': new_rank + 1,
            'change': old_rank - new_rank,  # Positive = moved up
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
