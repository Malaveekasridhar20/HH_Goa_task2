import os
import json
import time
import pickle
from typing import List
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.retrieval.models import IndexedChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.bm25_index import BM25Index
from app.retrieval.persistence import MetadataStore
from app.generation.extractive_generator import ExtractiveAnswerGenerator

def get_chunk_file(lang: str) -> str:
    if lang == "en": return "english_chunks.jsonl"
    if lang == "hi": return "chunks.jsonl"
    if lang == "ta": return "chunks_tamil.jsonl"
    if lang == "te": return "chunks_telugu.jsonl"
    if lang == "ml": return "chunks_malayalam.jsonl"
    raise ValueError(f"Unknown lang {lang}")

def get_index_dir(lang: str) -> str:
    names = {"en": "english", "hi": "hindi", "ta": "tamil", "te": "telugu", "ml": "malayalam"}
    return names[lang]

def process_language(lang: str, embedder: EmbeddingService, generator: ExtractiveAnswerGenerator):
    print(f"\n=== Building Indexes for '{lang}' ===")
    
    chunk_file = get_chunk_file(lang)
    chunk_path = os.path.join(os.path.dirname(__file__), f"../data/processed/{chunk_file}")
    
    if not os.path.exists(chunk_path):
        print(f"ERROR: Chunk file not found: {chunk_path}")
        return
        
    chunks: List[IndexedChunk] = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunks.append(IndexedChunk.model_validate_json(line))
            
    print(f"Loaded {len(chunks)} chunks.")
    
    start_time = time.time()
    
    print("Generating chunk embeddings for FAISS...")
    chunk_embeddings = []
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = [c.text for c in chunks[i:i+batch_size]]
        emb = embedder.encode_documents(batch)
        chunk_embeddings.extend(emb)
    
    print("Generating sentence embeddings for Extractive Generator...")
    sentence_store = {}
    total_sentences = 0
    all_sentences_batch = []
    all_keys = []
    
    for chunk in chunks:
        sents = generator._split_sentences(chunk.text)
        for idx, s in enumerate(sents):
            s = s.strip()
            if len(s) > 5:
                all_sentences_batch.append(s)
                all_keys.append((chunk.chunk_id, idx, s))
                total_sentences += 1
                
    sent_embeddings = []
    for i in range(0, len(all_sentences_batch), batch_size):
        batch = all_sentences_batch[i:i+batch_size]
        emb = embedder.encode_documents(batch)
        sent_embeddings.extend(emb)
        
    for k, emb in zip(all_keys, sent_embeddings):
        sentence_store[k] = emb
        
    emb_time = time.time() - start_time
    print(f"Generated {len(chunk_embeddings)} chunk embeddings and {total_sentences} sentence embeddings in {emb_time:.2f}s")
    
    embeddings_array = np.array(chunk_embeddings, dtype=np.float32)
    dim = embeddings_array.shape[1]
    
    faiss_index = FaissIndex(dimension=dim)
    faiss_index.add_embeddings(embeddings_array)
    
    bm25_index = BM25Index()
    bm25_index.build_index([c.text for c in chunks])
    
    store = MetadataStore()
    store.add_chunks(chunks)
    
    out_dir = os.path.join(os.path.dirname(__file__), f"../data/indexes/{get_index_dir(lang)}")
    os.makedirs(out_dir, exist_ok=True)
    
    faiss_index.save(os.path.join(out_dir, "faiss.index"))
    bm25_index.save(os.path.join(out_dir, "bm25.pkl"))
    store.save(os.path.join(out_dir, "metadata.jsonl"))
    
    with open(os.path.join(out_dir, "sentence_embeddings.pkl"), "wb") as f:
        pickle.dump(sentence_store, f)
        
    report = {
        "language": lang,
        "chunks_indexed": len(chunks),
        "sentences_indexed": total_sentences,
        "embedding_model": embedder.model_name,
        "dimension": dim,
        "device": embedder.device,
        "build_time_seconds": emb_time
    }
    with open(os.path.join(out_dir, "index_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Indexes built successfully in {out_dir}")

def main():
    languages = ["en", "hi", "ta", "te", "ml"]
    print("Initializing embedding service (loading model)...")
    embedder = EmbeddingService()
    generator = ExtractiveAnswerGenerator(embedding_service=embedder)
    
    for lang in languages:
        process_language(lang, embedder, generator)

if __name__ == "__main__":
    main()
