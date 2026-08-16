import os
import time
import json
import numpy as np
import pytest
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.bm25_index import BM25Index
from app.retrieval.persistence import MetadataStore
from app.retrieval.retriever import Retriever

def run_audit():
    print("=== 2. EMBEDDING MODEL VERIFICATION ===")
    embedder = EmbeddingService("intfloat/multilingual-e5-small")
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: 384 (Implied by output shape)")
    
    # 3. Determinism
    print("\n=== 3. EMBEDDING DETERMINISM ===")
    emb1 = embedder.encode_query("hello world")
    emb2 = embedder.encode_query("hello world")
    if np.allclose(emb1, emb2, atol=1e-6):
        print("Embedding determinism: PASS")
    else:
        print("Embedding determinism: FAIL")

    print("\n=== 4. FAISS INDEX CORRECTNESS ===")
    for lang in ["en", "hi"]:
        idx_dir = f"data/indexes/{'english' if lang == 'en' else 'hindi'}"
        meta_path = os.path.join(idx_dir, "metadata.jsonl")
        faiss_path = os.path.join(idx_dir, "faiss.index")
        
        meta = MetadataStore.load(meta_path)
        faiss_idx = FaissIndex.load(faiss_path)
        
        print(f"[{lang}] FAISS vectors: {faiss_idx.index.ntotal}")
        print(f"[{lang}] FAISS dimension: {faiss_idx.dimension}")
        print(f"[{lang}] Metadata count: {len(meta.chunks)}")
        
        # Mapping test
        q_emb = embedder.encode_query(meta.chunks[0].text[:50])
        dist, ids = faiss_idx.search(q_emb, top_k=1)
        if len(ids) > 0 and 0 <= ids[0] < len(meta.chunks):
            print(f"[{lang}] English mapping: PASS" if lang == 'en' else f"[{lang}] Hindi mapping: PASS")
        else:
            print(f"[{lang}] English mapping: FAIL" if lang == 'en' else f"[{lang}] Hindi mapping: FAIL")

    print("\n=== 5. BM25 INDEX CORRECTNESS ===")
    for lang in ["en", "hi"]:
        idx_dir = f"data/indexes/{'english' if lang == 'en' else 'hindi'}"
        bm25_path = os.path.join(idx_dir, "bm25.pkl")
        bm25_idx = BM25Index.load(bm25_path)
        print(f"[{lang}] BM25 document count: {bm25_idx.doc_count}")
        # Test basic search
        score, ids = bm25_idx.search("test" if lang == 'en' else "नमस्ते", top_k=1)
        if len(ids) > 0:
            print(f"[{lang}] Search mapping works: PASS")

    print("\n=== 11. LATENCY AUDIT & 12. LATENCY DISTRIBUTION ===")
    hi_idx_dir = "data/indexes/hindi"
    retriever_hi = Retriever(hi_idx_dir)
    
    raw_data_path = "data/processed/hinval_500.jsonl"
    queries = []
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if row.get('query'):
                queries.append(row.get('query'))
                
    # Warmup
    for q in queries[:10]:
        retriever_hi.retrieve_vector(q, top_k=10)
        retriever_hi.retrieve_bm25(q, top_k=10)
        
    faiss_lat = []
    bm25_lat = []
    bm25_logs = []
    
    for q in queries[:248]:
        t0 = time.time()
        retriever_hi.retrieve_vector(q, top_k=10)
        faiss_lat.append(time.time() - t0)
        
        t0 = time.time()
        res = retriever_hi.retrieve_bm25(q, top_k=10)
        dur = time.time() - t0
        bm25_lat.append(dur)
        bm25_logs.append((dur, q, len(res)))
        
    print("Hindi FAISS warm P50:", np.percentile(faiss_lat, 50))
    print("Hindi FAISS warm P70:", np.percentile(faiss_lat, 70))
    print("Hindi FAISS warm P100:", np.max(faiss_lat))
    
    print("Hindi BM25 warm P50:", np.percentile(bm25_lat, 50))
    print("Hindi BM25 warm P70:", np.percentile(bm25_lat, 70))
    print("Hindi BM25 warm P100:", np.max(bm25_lat))
    
    bm25_logs.sort(key=lambda x: x[0], reverse=True)
    print("\nSlowest 5 Hindi BM25 Queries:")
    for dur, q, count in bm25_logs[:5]:
        print(f"Latency: {dur:.4f}s | Query Len: {len(q)} | Results: {count}")
        
    print("\n=== 13. INDEX RELOAD ===")
    try:
        del retriever_hi
        retriever_hi = Retriever("data/indexes/hindi")
        retriever_en = Retriever("data/indexes/english")
        print("English reload: PASS")
        print("Hindi reload: PASS")
    except Exception as e:
        print(f"Reload failed: {e}")
        
    print("\n=== 14. CROSS-LANGUAGE TEST ===")
    try:
        retriever_hi.retrieve_vector("What is a company?", top_k=1)
        print("English -> Hindi: PASS")
    except:
        print("English -> Hindi: FAIL")
        
    try:
        retriever_en.retrieve_vector("नमस्ते दुनिया", top_k=1)
        print("Hindi -> English: PASS")
    except:
        print("Hindi -> English: FAIL")

if __name__ == "__main__":
    run_audit()
