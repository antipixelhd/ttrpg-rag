
import argparse
import sys

from src.config import load_config, print_config
from src.preprocess import preprocess_all
from src.indexing import index_all, delete_collection
from src.retrieval import search
from src.run_tracker import (
    create_run,
    save_config,
    save_results,
    save_response,
)

def cmd_preprocess(args):
    print("=" * 70)
    print("TTRPG RAG - Preprocessing")
    print("=" * 70)
    
    config = load_config(args.config)
    verbose = args.verbose or config.get('output', {}).get('verbose', False)
    
    if verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    processed_files = preprocess_all(config, verbose)
    
    print(f"\nDone! Processed {len(processed_files)} files.")
    
    return 0

def cmd_index(args):
    print("=" * 70)
    print("TTRPG RAG - Indexing")
    print("=" * 70)
    
    config = load_config(args.config)
    verbose = args.verbose or config.get('output', {}).get('verbose', False)
    
    if verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.delete:
        print("\nDeleting Qdrant storage...")
        success = delete_collection(config, verbose)
        if not success:
            print("Failed to delete storage. Aborting.")
            return 1
    
    result = index_all(config, verbose)
    
    if result:
        total = result['count']
        new = result.get('new_count', 0)
        if new > 0:
            print(f"\nDone! Indexed {new} new chunks (total: {total} chunks).")
        else:
            print(f"\nDone! All {total} chunks already indexed.")
    else:
        print("\nIndexing failed or no data to index.")
        return 1
    
    return 0

def cmd_search(args):
    print("=" * 70)
    print("TTRPG RAG - Search")
    print("=" * 70)
    
    cli_overrides = {}
    if args.top_k:
        cli_overrides['retrieval'] = {'top_k': args.top_k}
    if args.expand:
        cli_overrides['query_expansion'] = {'enabled': True}
    if args.rerank:
        cli_overrides['reranking'] = {'enabled': True}
    
    config = load_config(args.config, cli_overrides)
    verbose = args.verbose or config.get('output', {}).get('verbose', False)
    
    if verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.evaluate:
        from src.evaluation import evaluate_retrieval, save_retrieval_results
        results = evaluate_retrieval(config, verbose=verbose)
        
        if 'error' not in results:
            output_file = save_retrieval_results(results, config)
            
            print("\nRetrieval Evaluation Results:")
            print(f"  Questions evaluated: {results['num_questions']}")
            print(f"  Precision@5:  {results['metrics']['precision_at_5']:.4f}")
            print(f"  Recall@5:     {results['metrics']['recall_at_5']:.4f}")
            print(f"  F1@5:         {results['metrics']['f1_at_5']:.4f}")
            print(f"  Precision@10: {results['metrics']['precision_at_10']:.4f}")
            print(f"  Recall@10:    {results['metrics']['recall_at_10']:.4f}")
            print(f"  F1@10:        {results['metrics']['f1_at_10']:.4f}")
            print(f"  MRR:          {results['metrics']['mrr']:.4f}")
            print(f"\nResults saved to: {output_file}")
        return 0
    
    if not args.question:
        print("Error: Question is required (or use --evaluate)")
        return 1
    
    if args.track:
        run_dir = create_run(config, "search")
        save_config(run_dir, config)
    else:
        run_dir = None
    
    results = search(args.question, config, verbose)
    
    if run_dir:
        save_results(run_dir, args.question, results, query_number=1)
    
    return 0

def cmd_chat(args):
    print("=" * 70)
    print("TTRPG RAG - Chat")
    print("=" * 70)
    
    cli_overrides = {}
    if args.top_k:
        cli_overrides['retrieval'] = {'top_k': args.top_k}
    if args.expand:
        cli_overrides['query_expansion'] = {'enabled': True}
    if args.rerank:
        cli_overrides['reranking'] = {'enabled': True}
    
    config = load_config(args.config, cli_overrides)
    verbose = args.verbose or config.get('output', {}).get('verbose', False)
    
    if verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.evaluate:
        from src.evaluation import evaluate_response, save_response_results
        results = evaluate_response(config, verbose=verbose)
        
        if 'error' not in results:
            output_file = save_response_results(results, config)
            
            print("\nResponse Evaluation Results:")
            print(f"  Questions evaluated: {results['num_questions']}")
            print(f"  Avg Groundedness: {results['metrics']['avg_groundedness']:.2f}/5")
            print(f"  Avg Correctness:  {results['metrics']['avg_correctness']:.2f}/5")
            print(f"  Response Model: {results['model']}")
            print(f"  Eval Model: {results['eval_model']}")
            print(f"  Generation tokens: {results['token_usage']['generation_input_tokens']} in / {results['token_usage']['generation_output_tokens']} out")
            print(f"  Evaluation tokens: {results['token_usage']['evaluation_input_tokens']} in / {results['token_usage']['evaluation_output_tokens']} out")
            print(f"\nResults saved to: {output_file}")
        return 0
    
    if not args.question:
        print("Error: Question is required (or use --evaluate)")
        return 1
    
    if args.track:
        run_dir = create_run(config, "chat")
        save_config(run_dir, config)
    else:
        run_dir = None
    
    print(f"\nQuestion: {args.question}")
    print("-" * 70)
    
    results = search(args.question, config, verbose)
    
    if run_dir:
        save_results(run_dir, args.question, results, query_number=1)
    
    from src.response import generate_response
    
    response_data = generate_response(args.question, results, config, verbose)
    response = response_data['answer']
    
    if run_dir:
        save_response(run_dir, args.question, response, results)
    
    print("\n" + "=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(response)
    print("=" * 70)
    
    if args.show_sources:
        print("\nSources used:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['name']} (score: {result['score']:.4f})")
    
    return 0

def cmd_run(args):
    print("=" * 70)
    print("TTRPG RAG - Full Pipeline Run")
    print("=" * 70)
    
    cli_overrides = {}
    if args.top_k:
        cli_overrides['retrieval'] = {'top_k': args.top_k}
    if args.expand:
        cli_overrides['query_expansion'] = {'enabled': True}
    if args.rerank:
        cli_overrides['reranking'] = {'enabled': True}
    
    config = load_config(args.config, cli_overrides)
    verbose = args.verbose or config.get('output', {}).get('verbose', False)
    
    if verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    run_dir = create_run(config, "full_run")
    save_config(run_dir, config)
    
    print("=" * 50)
    print("STEP 1: Preprocessing")
    print("=" * 50)
    
    if not args.skip_preprocess:
        preprocess_all(config, verbose)
    else:
        print("Skipping preprocessing (--skip-preprocess flag)")
    
    print()
    print("=" * 50)
    print("STEP 2: Indexing")
    print("=" * 50)
    
    if not args.skip_index:
        if args.delete:
            print("Deleting Qdrant storage...")
            success = delete_collection(config, verbose)
            if not success:
                print("Failed to delete storage. Aborting.")
                return 1
        
        result = index_all(config, verbose)
    else:
        print("Skipping indexing (--skip-index flag)")
    
    print()
    print("=" * 50)
    print("STEP 3: Searching")
    print("=" * 50)
    
    results = search(args.question, config, verbose)
    save_results(run_dir, args.question, results, query_number=1)
    
    print()
    print("=" * 50)
    print("STEP 4: Generating Response")
    print("=" * 50)
    
    from src.response import generate_response
    
    response_data = generate_response(args.question, results, config, verbose)
    response = response_data['answer']
    save_response(run_dir, args.question, response, results)
    
    print()
    print("-" * 50)
    print("ANSWER:")
    print("-" * 50)
    print(response)
    
    print()
    print("=" * 50)
    print(f"Pipeline complete! Results saved to: {run_dir}")
    print("=" * 50)
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description='TTRPG RAG System - Search your campaign session notes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py preprocess
  python main.py index
  python main.py search "Who is the villain?"
  python main.py run "What happened last?"
  
  python main.py search "query" --top-k 10
  python main.py search "query" --expand
  python main.py index --delete"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Extract and clean summaries from raw session notes'
    )
    preprocess_parser.add_argument(
        '--config', '-c',
        help='Path to custom config YAML file'
    )
    preprocess_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    
    index_parser = subparsers.add_parser(
        'index',
        help='Create embeddings and upload chunks to Qdrant'
    )
    index_parser.add_argument(
        '--config', '-c',
        help='Path to custom config YAML file'
    )
    index_parser.add_argument(
        '--delete', '-d',
        action='store_true',
        help='Delete entire Qdrant storage before indexing'
    )
    index_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    
    search_parser = subparsers.add_parser(
        'search',
        help='Search for relevant session information'
    )
    search_parser.add_argument(
        'question',
        nargs='?',
        default=None,
        help='The question to search for (not required with --evaluate)'
    )
    search_parser.add_argument(
        '--config', '-c',
        help='Path to custom config YAML file'
    )
    search_parser.add_argument(
        '--top-k', '-k',
        type=int,
        help='Number of results to return'
    )
    search_parser.add_argument(
        '--expand', '-e',
        action='store_true',
        help='Enable query expansion for better results'
    )
    search_parser.add_argument(
        '--rerank', '-r',
        action='store_true',
        help='Enable reranking with cross-encoder model'
    )
    search_parser.add_argument(
        '--track', '-t',
        action='store_true',
        help='Create a run folder to track this operation'
    )
    search_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    search_parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation on all questions from data/questions.json'
    )
    
    chat_parser = subparsers.add_parser(
        'chat',
        help='Ask a question and get an AI-generated answer'
    )
    chat_parser.add_argument(
        'question',
        nargs='?',
        default=None,
        help='Your question about the campaign (not required with --evaluate)'
    )
    chat_parser.add_argument(
        '--config', '-c',
        help='Path to custom config YAML file'
    )
    chat_parser.add_argument(
        '--top-k', '-k',
        type=int,
        help='Number of chunks to retrieve for context'
    )
    chat_parser.add_argument(
        '--expand', '-e',
        action='store_true',
        help='Enable query expansion for better results'
    )
    chat_parser.add_argument(
        '--rerank', '-r',
        action='store_true',
        help='Enable reranking with cross-encoder model'
    )
    chat_parser.add_argument(
        '--show-sources', '-s',
        action='store_true',
        help='Show the sources used to generate the answer'
    )
    chat_parser.add_argument(
        '--track', '-t',
        action='store_true',
        help='Create a run folder to track this query'
    )
    chat_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    chat_parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation on all questions from data/questions.json'
    )
    
    run_parser = subparsers.add_parser(
        'run',
        help='Run the full pipeline (preprocess -> index -> search -> respond)'
    )
    run_parser.add_argument(
        'question',
        help='The question to search for'
    )
    run_parser.add_argument(
        '--config', '-c',
        help='Path to custom config YAML file'
    )
    run_parser.add_argument(
        '--top-k', '-k',
        type=int,
        help='Number of results to return'
    )
    run_parser.add_argument(
        '--expand', '-e',
        action='store_true',
        help='Enable query expansion for better results'
    )
    run_parser.add_argument(
        '--rerank', '-r',
        action='store_true',
        help='Enable reranking with cross-encoder model'
    )
    run_parser.add_argument(
        '--delete', '-d',
        action='store_true',
        help='Delete entire Qdrant storage before indexing'
    )
    run_parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip the preprocessing step'
    )
    run_parser.add_argument(
        '--skip-index',
        action='store_true',
        help='Skip the indexing step'
    )
    run_parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save embedding vectors to run folder (large file!)'
    )
    run_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'preprocess':
        return cmd_preprocess(args)
    elif args.command == 'index':
        return cmd_index(args)
    elif args.command == 'search':
        return cmd_search(args)
    elif args.command == 'chat':
        return cmd_chat(args)
    elif args.command == 'run':
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
