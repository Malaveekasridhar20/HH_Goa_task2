import pytest
import os
from app.retrieval.bm25_index import BM25Index

def test_bm25_index_basic(tmp_path):
    index = BM25Index()
    
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog outpaces a fast fox.",
        "नमस्ते दुनिया", # Hindi text
        "Hello world in Hindi is नमस्ते दुनिया",
        "Just another document to adjust IDF."
    ]
    
    index.build_index(docs)
    assert index.doc_count == 5
    
    # English search
    scores, indices = index.search("quick fox", top_k=2)
    assert indices[0] in [0, 1]
    
    # Hindi search
    scores_hi, indices_hi = index.search("नमस्ते", top_k=2)
    assert indices_hi[0] in [2, 3]
    
    # Save/load
    path = os.path.join(tmp_path, "bm25.pkl")
    index.save(path)
    
    loaded = BM25Index.load(path)
    assert loaded.doc_count == 5
    s2, i2 = loaded.search("नमस्ते", top_k=2)
    assert list(indices_hi) == list(i2)
