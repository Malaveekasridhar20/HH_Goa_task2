import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import IndexedChunk

@pytest.fixture
def mock_hybrid_retriever():
    with patch('app.retrieval.retriever.MetadataStore.load') as mock_meta, \
         patch('app.retrieval.retriever.FaissIndex.load') as mock_faiss, \
         patch('app.retrieval.retriever.BM25Index.load') as mock_bm25, \
         patch('app.retrieval.retriever.EmbeddingService') as mock_emb:
        
        meta_instance = MagicMock()
        # Mock chunks A, B, C, D
        def get_chunk(idx):
            chunk_id = ["A", "B", "C", "D"][idx]
            return IndexedChunk(
                chunk_id=chunk_id, text=f"text {chunk_id}", query_id="q1",
                source_lang="en", target_lang="hi", query_type="DESC",
                passage_index=0, chunk_index=0, is_selected=True,
                strategy="whole", strategy_reason="", start_position=0, end_position=10
            )
        meta_instance.get_chunk.side_effect = get_chunk
        mock_meta.return_value = meta_instance
        
        # FAISS returns A(0), B(1), C(2)
        faiss_instance = MagicMock()
        faiss_instance.search.return_value = ([0.9, 0.8, 0.7], [0, 1, 2])
        mock_faiss.return_value = faiss_instance
        
        # BM25 returns B(1), C(2), D(3)
        bm25_instance = MagicMock()
        bm25_instance.search.return_value = ([2.0, 1.5, 1.0], [1, 2, 3])
        mock_bm25.return_value = bm25_instance
        
        emb_instance = MagicMock()
        emb_instance.encode_query.return_value = [0.1]
        mock_emb.return_value = emb_instance
        
        retriever = HybridRetriever("dummy/dir")
        yield retriever

def test_score_normalization(mock_hybrid_retriever):
    scores = [0.9, 0.8, 0.7]
    norm = mock_hybrid_retriever._normalize_scores(scores)
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(0.5)
    assert norm[2] == pytest.approx(0.0)

def test_equal_score_normalization(mock_hybrid_retriever):
    scores = [1.5, 1.5, 1.5]
    norm = mock_hybrid_retriever._normalize_scores(scores)
    assert norm == [1.0, 1.0, 1.0]

def test_candidate_fusion_and_dedup(mock_hybrid_retriever):
    results = mock_hybrid_retriever.retrieve_hybrid("test", alpha=0.5, top_k=4)
    # Expected chunks: A, B, C, D
    assert len(results) == 4
    ids = [r.chunk_id for r in results]
    assert sorted(ids) == ["A", "B", "C", "D"]
    
    # Check normalized scores
    # FAISS: A=1.0, B=0.5, C=0.0
    # BM25: B=1.0, C=0.5, D=0.0
    # Fused (alpha=0.5):
    # A = 0.5*1.0 + 0.5*0.0 = 0.5
    # B = 0.5*0.5 + 0.5*1.0 = 0.75
    # C = 0.5*0.0 + 0.5*0.5 = 0.25
    # D = 0.5*0.0 + 0.5*0.0 = 0.0
    # Ranked: B(0.75), A(0.5), C(0.25), D(0.0)
    
    assert ids[0] == "B"
    assert results[0].hybrid_score == pytest.approx(0.75)
    
    assert ids[1] == "A"
    assert results[1].hybrid_score == pytest.approx(0.5)
    
    assert ids[2] == "C"
    assert results[2].hybrid_score == pytest.approx(0.25)
    
    assert ids[3] == "D"
    assert results[3].hybrid_score == pytest.approx(0.0)

def test_alpha_0_mimics_bm25(mock_hybrid_retriever):
    # Alpha = 0.0 means only BM25 score matters
    results = mock_hybrid_retriever.retrieve_hybrid("test", alpha=0.0, top_k=4)
    # BM25 norm: B=1.0, C=0.5, D=0.0, A=0.0
    ids = [r.chunk_id for r in results]
    assert ids[0] == "B"
    assert results[0].hybrid_score == 1.0
    
def test_alpha_1_mimics_faiss(mock_hybrid_retriever):
    # Alpha = 1.0 means only FAISS score matters
    results = mock_hybrid_retriever.retrieve_hybrid("test", alpha=1.0, top_k=4)
    # FAISS norm: A=1.0, B=0.5, C=0.0, D=0.0
    ids = [r.chunk_id for r in results]
    assert ids[0] == "A"
    assert results[0].hybrid_score == 1.0

def test_empty_results(mock_hybrid_retriever):
    mock_hybrid_retriever.retriever.faiss_index.search.return_value = ([], [])
    mock_hybrid_retriever.retriever.bm25_index.search.return_value = ([], [])
    results = mock_hybrid_retriever.retrieve_hybrid("test", top_k=10)
    assert len(results) == 0
