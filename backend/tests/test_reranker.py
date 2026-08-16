import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.reranker import Reranker
from app.retrieval.models import RetrievalResult

@pytest.fixture
def mock_reranker():
    with patch('app.retrieval.reranker.CrossEncoder') as mock_ce:
        ce_instance = MagicMock()
        # Mock prediction returning scores that match the order of pairs
        # Let's say it returns ascending scores so the last input is ranked first
        ce_instance.predict.side_effect = lambda pairs, **kwargs: [float(i) for i in range(len(pairs))]
        mock_ce.return_value = ce_instance
        
        reranker = Reranker("dummy/model")
        yield reranker

def test_reranker_basic(mock_reranker):
    candidates = [
        RetrievalResult(chunk_id="A", text="text A", score=0.9, rank=1, source_lang="en", is_selected=True),
        RetrievalResult(chunk_id="B", text="text B", score=0.8, rank=2, source_lang="en", is_selected=False),
        RetrievalResult(chunk_id="C", text="text C", score=0.7, rank=3, source_lang="en", is_selected=True),
    ]
    
    # Since predict returns [0.0, 1.0, 2.0], C gets highest score (2.0)
    results = mock_reranker.rerank("query", candidates, top_k=2)
    
    assert len(results) == 2
    
    assert results[0].chunk_id == "C"
    assert results[0].original_rank == 3
    assert results[0].final_rank == 1
    assert results[0].rerank_score == 2.0
    assert results[0].is_selected is True
    
    assert results[1].chunk_id == "B"
    assert results[1].original_rank == 2
    assert results[1].final_rank == 2
    assert results[1].rerank_score == 1.0

def test_reranker_empty(mock_reranker):
    assert mock_reranker.rerank("query", [], top_k=10) == []
