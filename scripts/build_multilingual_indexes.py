import os
import sys
import json
import time
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app.retrieval.models import IndexedChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.bm25_index import BM25Index
from app.retrieval.persistence import MetadataStore

def process_language(lang_name: str):
    print(f"\n=== Building Indexes for {lang_name} ===")
    
    chunk_path = os.path.join(os.path.dirname(__file__), f"../data/processed/chunks_{lang_name}.jsonl")
    if not os.path.exists(chunk_path):
        print(f"Chunk file not found: {chunk_path}")
        return
        
    chunks: List[IndexedChunk] = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunks.append(IndexedChunk.model_validate_json(line))
            
    print(f"Loaded {len(chunks)} chunks.")
    
    embedder = EmbeddingService()
    
    all_embeddings = []
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = [c.text for c in chunks[i:i+batch_size]]
        emb = embedder.encode_documents(batch)
        all_embeddings.extend(emb)
        
    import numpy as np
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    dim = embeddings_array.shape[1]
    
    faiss_index = FaissIndex(dimension=dim)
    faiss_index.add_embeddings(embeddings_array)
    
    bm25_index = BM25Index()
    bm25_index.build_index([c.text for c in chunks])
    
    store = MetadataStore()
    store.add_chunks(chunks)
    
    out_dir = os.path.join(os.path.dirname(__file__), f"../data/indexes/{lang_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    faiss_index.save(os.path.join(out_dir, "faiss.index"))
    bm25_index.save(os.path.join(out_dir, "bm25.pkl"))
    store.save(os.path.join(out_dir, "metadata.jsonl"))
    print(f"Indexes built successfully in {out_dir}")

def main():
    languages = ["tamil", "telugu", "malayalam"]
    for lang in languages:
        process_language(lang)

if __name__ == "__main__":
    main()
