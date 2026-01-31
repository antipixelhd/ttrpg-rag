
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List
from datetime import datetime

from src.config import resolve_path

@dataclass
class EvaluationQuestion:
    qid: int
    question: str
    answerable: bool
    goldAnswer: str
    goldChunks: List[int]

def load_questions(config) -> List[EvaluationQuestion]:
    questions_path = resolve_path(config.get('evaluation', {}).get('questions_file', 'data/questions.json'))
    
    if not questions_path.exists():
        print(f"Warning: Questions file not found: {questions_path}")
        return []
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        questions.append(EvaluationQuestion(
            qid=item['qid'],
            question=item['question'],
            answerable=item['answerable'],
            goldAnswer=item['goldAnswer'],
            goldChunks=item['goldChunks'],
        ))
    
    return questions

def compute_metrics_at_k(retrieved_ids: List[int], gold_ids: set, k: int) -> tuple:
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_at_k) & gold_ids
    
    precision = len(relevant_retrieved) / k if k > 0 else 0.0
    recall = len(relevant_retrieved) / len(gold_ids) if gold_ids else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def compute_mrr(retrieved_ids: List[int], gold_ids: set) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(config, verbose: bool = False) -> dict:
    from src.retrieval import search
    
    questions = load_questions(config)
    
    questions_with_gold = [q for q in questions if q.goldChunks]
    
    if not questions_with_gold:
        print("Error: No evaluation questions with goldChunks found")
        return {'error': 'No questions with gold chunks'}
    
    print(f"Running retrieval evaluation on {len(questions_with_gold)} questions...")
    
    all_precision_5 = []
    all_recall_5 = []
    all_f1_5 = []
    all_precision_10 = []
    all_recall_10 = []
    all_f1_10 = []
    all_mrr = []
    
    individual_results = []
    
    for i, q in enumerate(questions_with_gold, 1):
        if verbose:
            print(f"  [{i}/{len(questions_with_gold)}] Evaluating: {q.question[:50]}...")
        
        retrieved = search(q.question, config, verbose=False)
        retrieved_ids = [chunk.get('chunk_id') for chunk in retrieved]
        gold_ids = set(q.goldChunks)
        
        p5, r5, f1_5 = compute_metrics_at_k(retrieved_ids, gold_ids, 5)
        p10, r10, f1_10 = compute_metrics_at_k(retrieved_ids, gold_ids, 10)
        mrr = compute_mrr(retrieved_ids, gold_ids)
        
        all_precision_5.append(p5)
        all_recall_5.append(r5)
        all_f1_5.append(f1_5)
        all_precision_10.append(p10)
        all_recall_10.append(r10)
        all_f1_10.append(f1_10)
        all_mrr.append(mrr)
        
        individual_results.append({
            'qid': q.qid,
            'question': q.question,
            'gold_chunks': list(gold_ids),
            'retrieved_chunks': retrieved_ids[:10],
            'precision_at_5': p5,
            'recall_at_5': r5,
            'f1_at_5': f1_5,
            'precision_at_10': p10,
            'recall_at_10': r10,
            'f1_at_10': f1_10,
            'mrr': mrr,
        })
    
    n = len(questions_with_gold)
    results = {
        'num_questions': n,
        'metrics': {
            'precision_at_5': sum(all_precision_5) / n,
            'recall_at_5': sum(all_recall_5) / n,
            'f1_at_5': sum(all_f1_5) / n,
            'precision_at_10': sum(all_precision_10) / n,
            'recall_at_10': sum(all_recall_10) / n,
            'f1_at_10': sum(all_f1_10) / n,
            'mrr': sum(all_mrr) / n,
        },
        'individual_results': individual_results,
    }
    
    print(f"Retrieval evaluation complete: {n} questions processed")
    
    return results

def evaluate_response(config, verbose: bool = False) -> dict:
    print("Response evaluation not yet implemented")
    return {
        'num_questions': 0,
        'metrics': {},
        'individual_results': [],
        'status': 'not_implemented',
    }

def save_retrieval_results(results: dict, config) -> Path:
    output_dir = resolve_path('data/evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'retrievalEvaluation.json'
    
    existing = []
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'num_questions': results['num_questions'],
        'metrics': results['metrics'],
        'config': {
            'embedding_model': config['embedding']['model'],
            'top_k': config['retrieval']['top_k'],
            'reranking_enabled': config.get('reranking', {}).get('enabled', False),
            'query_expansion_enabled': config.get('query_expansion', {}).get('enabled', False),
        }
    }
    
    existing.append(results_to_save)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    
    return output_file
