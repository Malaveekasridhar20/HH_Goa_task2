import pytest
import numpy as np
from app.retrieval.embeddings import EmbeddingService

def test_embedding_service_e5_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    service = EmbeddingService()
    
    assert service.is_e5 == True
    
    # We can't easily mock the internal sentence transformer without complexity,
    # but we can test that it returns valid shapes.
    docs = ["This is a test passage."]
    doc_emb = service.encode_documents(docs)
    assert doc_emb.shape == (1, 384)
    assert doc_emb.dtype == np.float32
    
    q_emb = service.encode_query("This is a query.")
    assert q_emb.shape == (384,)
    assert q_emb.dtype == np.float32
