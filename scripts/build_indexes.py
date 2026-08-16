import os
import json
import time
from typing import List
from app.retrieval.models import IndexedChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.bm25_index import BM25Index
from app.retrieval.persistence import MetadataStore

def build_indexes():
    lang = os.getenv("INDEX_LANGUAGE", "hi").lower()
    if lang not in ["en", "hi"]:
        raise ValueError("INDEX_LANGUAGE must be 'en' or 'hi'")
        
    print(f"=== Building Indexes for '{lang}' ===")
    
    # 1. Load chunks
    chunk_file = "english_chunks.jsonl" if lang == "en" else "chunks.jsonl"
    chunk_path = os.path.join(os.path.dirname(__file__), f"../data/processed/{chunk_file}")
    
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        
    chunks: List[IndexedChunk] = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(IndexedChunk.model_validate_json(line))
            
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")
    
    # 2. Init Embedding Service
    print("Initializing embedding service...")
    embedder = EmbeddingService()
    print(f"Using model: {embedder.model_name} on {embedder.device}")
    
    # 3. Generate embeddings
    print("Generating embeddings...")
    start_time = time.time()
    
    # Batch extraction for progress tracking
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = [c.text for c in chunks[i:i+batch_size]]
        emb = embedder.encode_documents(batch)
        all_embeddings.extend(emb)
        print(f"  Encoded {min(i+batch_size, len(chunks))}/{len(chunks)} chunks...", end='\r')
        
    print()
    emb_time = time.time() - start_time
    print(f"Generated {len(all_embeddings)} embeddings in {emb_time:.2f}s")
    
    # 4. Build FAISS
    import numpy as np
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    dim = embeddings_array.shape[1]
    
    print(f"Building FAISS index (dimension {dim})...")
    faiss_index = FaissIndex(dimension=dim)
    faiss_index.add_embeddings(embeddings_array)
    
    # 5. Build BM25
    print("Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build_index([c.text for c in chunks])
    
    # 6. Metadata Store
    store = MetadataStore()
    store.add_chunks(chunks)
    
    # 7. Persist
    out_dir = os.path.join(os.path.dirname(__file__), f"../data/indexes/{'english' if lang == 'en' else 'hindi'}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving indexes to {out_dir}...")
    faiss_index.save(os.path.join(out_dir, "faiss.index"))
    bm25_index.save(os.path.join(out_dir, "bm25.pkl"))
    store.save(os.path.join(out_dir, "metadata.jsonl"))
    
    # 8. Report
    report = {
        "language": lang,
        "chunks_indexed": len(chunks),
        "embedding_model": embedder.model_name,
        "dimension": dim,
        "device": embedder.device,
        "faiss": {
            "type": "IndexFlatIP",
            "count": len(chunks)
        },
        "bm25": {
            "type": "BM25Okapi",
            "count": len(chunks)
        },
        "build_time_seconds": emb_time
    }
    
    with open(os.path.join(out_dir, "index_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    build_indexes()
