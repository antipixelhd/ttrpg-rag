
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