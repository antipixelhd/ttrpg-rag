# =============================================================================
# TTRPG RAG System - Main CLI Entry Point
# =============================================================================
# This is the main command-line interface for the RAG system.
# It provides subcommands for each pipeline step plus a full pipeline runner.
#
# Usage:
#   python main.py preprocess              # Extract summaries from raw notes
#   python main.py index                   # Create embeddings and upload to Qdrant
#   python main.py search "your question"  # Search the vector store
#   python main.py run "your question"     # Run the full pipeline
#
# All commands support:
#   --config FILE    Load a custom config file
#   --top-k N        Override the number of results to return

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
    get_logger,
)

# =============================================================================
# Command Handlers
# =============================================================================

def cmd_preprocess(args):
    """
    Handle the 'preprocess' command.
    
    Extracts summaries from raw session notes in data/raw/
    and saves cleaned versions to data/processed/
    """
    print("=" * 70)
    print("TTRPG RAG - Preprocessing")
    print("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    # Create a run folder if tracking is enabled
    if args.track:
        run_dir = create_run(config, "preprocess")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        logger = None
    
    # Run preprocessing
    processed_files = preprocess_all(config, logger)
    
    print(f"\nDone! Processed {len(processed_files)} files.")
    
    return 0


def cmd_index(args):
    """
    Handle the 'index' command.
    
    Chunks the processed summaries, generates embeddings,
    and uploads everything to Qdrant.
    """
    print("=" * 70)
    print("TTRPG RAG - Indexing")
    print("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.verbose:
        print("\nConfiguration:")
        print_config(config)
        print()
    
    # Handle --delete flag to clear existing storage
    if args.delete:
        print("\nDeleting Qdrant storage...")
        success = delete_collection(config)
        if not success:
            print("Failed to delete storage. Aborting.")
            return 1
    
    # Create a run folder if tracking is enabled
    if args.track:
        run_dir = create_run(config, "index")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        run_dir = None
        logger = None
    
    # Run indexing
    result = index_all(config, logger)
    
    if result and run_dir:
        # Save chunks and embeddings to run folder
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
    """
    Handle the 'search' command.
    
    Searches the Qdrant vector store for chunks relevant to the question.
    """
    print("=" * 70)
    print("TTRPG RAG - Search")
    print("=" * 70)
    
    # Load configuration with CLI overrides
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
    
    # Create a run folder if tracking is enabled
    if args.track:
        run_dir = create_run(config, "search")
        logger = get_logger(run_dir)
        save_config(run_dir, config)
    else:
        run_dir = None
        logger = None
    
    # Run search
    results = search(args.question, config, logger)
    
    # Save results if tracking
    if run_dir:
        save_results(run_dir, args.question, results, query_number=1)
    
    print(f"\nFound {len(results)} results.")
    
    return 0


def cmd_run(args):
    """
    Handle the 'run' command.
    
    Runs the full pipeline: preprocess -> index -> search
    This is useful for testing the complete system.
    """
    print("=" * 70)
    print("TTRPG RAG - Full Pipeline Run")
    print("=" * 70)
    
    # Load configuration with CLI overrides
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
    
    # Always create a run folder for full pipeline runs
    run_dir = create_run(config, "full_run")
    logger = get_logger(run_dir)
    save_config(run_dir, config)
    
    # Step 1: Preprocess
    logger.info("=" * 50)
    logger.info("STEP 1: Preprocessing")
    logger.info("=" * 50)
    
    if not args.skip_preprocess:
        preprocess_all(config, logger)
    else:
        logger.info("Skipping preprocessing (--skip-preprocess flag)")
    
    # Step 2: Index
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
    
    # Step 3: Search
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 3: Searching")
    logger.info("=" * 50)
    
    results = search(args.question, config, logger)
    save_results(run_dir, args.question, results, query_number=1)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Pipeline complete! Results saved to: {run_dir}")
    logger.info("=" * 50)
    
    return 0



# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point - parse arguments and run the appropriate command.
    """
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='TTRPG RAG System - Search your campaign session notes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess                    # Process raw session notes
  python main.py index                         # Create vector embeddings
  python main.py search "Who is the villain?"  # Search for relevant info
  python main.py run "What happened last?"     # Run full pipeline
  
  python main.py search "query" --top-k 10     # Get more results
  python main.py search "query" --expand       # Enable query expansion
  python main.py index --delete                # Re-index from scratch
        """
    )
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # -------------------------------------------------------------------------
    # Preprocess command
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Index command
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Search command
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Run command (full pipeline)
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        'run',
        help='Run the full pipeline (preprocess -> index -> search)'
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, print help
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to the appropriate command handler
    if args.command == 'preprocess':
        return cmd_preprocess(args)
    elif args.command == 'index':
        return cmd_index(args)
    elif args.command == 'search':
        return cmd_search(args)
    elif args.command == 'run':
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
