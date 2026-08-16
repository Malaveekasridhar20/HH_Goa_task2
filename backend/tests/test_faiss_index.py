import pytest
import numpy as np
import os
from app.retrieval.faiss_index import FaissIndex

def test_faiss_index_basic(tmp_path):
    dim = 4
    index = FaissIndex(dimension=dim)
    
    # 3 embeddings of dim 4
    # Normalize them to simulate cosine similarity
    emb = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    index.add_embeddings(emb)
    
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    dist, idx = index.search(q, top_k=2)
    
    assert idx[0] == 0
    assert np.isclose(dist[0], 1.0)
    
    # Test save/load
    path = os.path.join(tmp_path, "test.faiss")
    index.save(path)
    
    loaded = FaissIndex.load(path)
    dist2, idx2 = loaded.search(q, top_k=2)
    assert list(idx) == list(idx2)
