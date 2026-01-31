
from src.config import load_config
from src.chunking import get_all_chunks, create_chunk_id

def main():
    config = load_config()
    
    print("Loading and chunking all session summaries...")
    chunks = get_all_chunks(config, verbose=False)
    
    if not chunks:
        print("No chunks found. Run preprocessing first.")
        return
    
    print(f"Found {len(chunks)} chunks. Writing to allChunks.txt...")
    
    with open('allChunks.txt', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            chunk_id = create_chunk_id(chunk)
            f.write(f"chunk id: {chunk_id}\n")
            f.write(f"Content:\n{chunk['content']}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"Successfully wrote {len(chunks)} chunks to allChunks.txt")

if __name__ == '__main__':
    main()
