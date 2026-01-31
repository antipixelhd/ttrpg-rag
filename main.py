
import argparse
import sys

from src.config import load_config, print_config
from src.preprocess import preprocess_all
from src.indexing import index_all, delete_collection
from src.retrieval import search
from src.run_tracker import (
    create_run,
    save_config,
    save_chunks,
    save_embeddings,
    save_results,
    save_response,
    get_logger,
)

def cmd_preprocess(args):
    print("=" * 70)
    print("TTRPG RAG - Preprocessing")
    print("=" * 70)
    
    config = load_config(args.config)
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.track:
        run_dir = create_run(config, "preprocess")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        logger = None
    
    processed_files = preprocess_all(config, logger)
    
    print(f"\nDone! Processed {len(processed_files)} files.")
    
    return 0

def cmd_index(args):
    print("=" * 70)
    print("TTRPG RAG - Indexing")
    print("=" * 70)
    
    config = load_config(args.config)
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.delete:
        print("\nDeleting Qdrant storage...")
        success = delete_collection(config)
        if not success:
            print("Failed to delete storage. Aborting.")
            return 1
    
    if args.track:
        run_dir = create_run(config, "index")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        run_dir = None
        logger = None
    
    result = index_all(config, logger)

    if result and run_dir:
        save_chunks(run_dir, result['chunks'])
        if args.save_embeddings:
            save_embeddings(run_dir, result['chunks'])
    
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
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.track:
        run_dir = create_run(config, "search")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        run_dir = None
        logger = None
    
    results = search(args.question, config, logger)
    
    if run_dir:
        save_results(run_dir, args.question, results, query_number=1)
    
    print(f"\nFound {len(results)} results.")
    
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
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    if args.track:
        run_dir = create_run(config, "chat")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        run_dir = None
        logger = None
    
    print(f"\nQuestion: {args.question}")
    print("-" * 70)
    
    results = search(args.question, config, logger)
    
    if run_dir:
        save_results(run_dir, args.question, results, query_number=1)
    
    from src.response import generate_response
    
    response = generate_response(args.question, results, config, logger)
    
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
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    run_dir = create_run(config, "full_run")
    logger = get_logger(run_dir)
    save_config(run_dir, config)
    
    logger.info("=" * 50)
    logger.info("STEP 1: Preprocessing")
    logger.info("=" * 50)
    
    if not args.skip_preprocess:
        preprocess_all(config, logger)
    else:
        logger.info("Skipping preprocessing (--skip-preprocess flag)")
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 2: Indexing")
    logger.info("=" * 50)
    
    if not args.skip_index:
        if args.delete:
            logger.info("Deleting Qdrant storage...")
            success = delete_collection(config, logger)
            if not success:
                logger.error("Failed to delete storage. Aborting.")
                return 1
        
        result = index_all(config, logger)
        if result:
            save_chunks(run_dir, result['chunks'])
            if args.save_embeddings:
                save_embeddings(run_dir, result['chunks'])
    else:
        logger.info("Skipping indexing (--skip-index flag)")
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 3: Searching")
    logger.info("=" * 50)
    
    results = search(args.question, config, logger)
    save_results(run_dir, args.question, results, query_number=1)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 4: Generating Response")
    logger.info("=" * 50)
    
    from src.response import generate_response
    
    response = generate_response(args.question, results, config, logger)
    save_response(run_dir, args.question, response, results)
    
    logger.info("")
    logger.info("-" * 50)
    logger.info("ANSWER:")
    logger.info("-" * 50)
    logger.info(response)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Pipeline complete! Results saved to: {run_dir}")
    logger.info("=" * 50)
    
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
        '--track', '-t',
        action='store_true',
        help='Create a run folder to track this operation'
    )
    preprocess_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed configuration'
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
        '--track', '-t',
        action='store_true',
        help='Create a run folder to track this operation'
    )
    index_parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save embedding vectors to run folder (large file!)'
    )
    index_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed configuration'
    )
    
    search_parser = subparsers.add_parser(
        'search',
        help='Search for relevant session information'
    )
    search_parser.add_argument(
        'question',
        help='The question to search for'
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
        help='Print detailed configuration'
    )
    
    chat_parser = subparsers.add_parser(
        'chat',
        help='Ask a question and get an AI-generated answer'
    )
    chat_parser.add_argument(
        'question',
        help='Your question about the campaign'
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
        help='Print detailed configuration'
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
        help='Print detailed configuration'
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