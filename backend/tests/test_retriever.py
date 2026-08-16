import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.retriever import Retriever
from app.retrieval.models import IndexedChunk

@pytest.fixture
def mock_retriever():
    with patch('app.retrieval.retriever.MetadataStore.load') as mock_meta, \
         patch('app.retrieval.retriever.FaissIndex.load') as mock_faiss, \
         patch('app.retrieval.retriever.BM25Index.load') as mock_bm25, \
         patch('app.retrieval.retriever.EmbeddingService') as mock_emb:
        
        # Setup metadata mock
        meta_instance = MagicMock()
        meta_instance.get_chunk.return_value = IndexedChunk(
            chunk_id="chunk1",
            text="Hello world",
            query_id="q1",
            query="test",
            eng_query="test eng",
            source_lang="en",
            target_lang="hi",
            query_type="DESC",
            passage_index=0,
            chunk_index=0,
            is_selected=True,
            strategy="whole_passage",
            strategy_reason="",
            start_position=0,
            end_position=10
        )
        mock_meta.return_value = meta_instance
        
        # Setup FAISS mock
        faiss_instance = MagicMock()
        faiss_instance.search.return_value = ([0.99], [0])
        mock_faiss.return_value = faiss_instance
        
        # Setup BM25 mock
        bm25_instance = MagicMock()
        bm25_instance.search.return_value = ([1.5], [0])
        mock_bm25.return_value = bm25_instance
        
        # Setup Embedding mock
        emb_instance = MagicMock()
        emb_instance.encode_query.return_value = [0.1, 0.2]
        mock_emb.return_value = emb_instance
        
        retriever = Retriever("dummy/dir")
        yield retriever

def test_retrieve_vector(mock_retriever):
    results = mock_retriever.retrieve_vector("test query", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk1"
    assert results[0].score == 0.99
    assert results[0].rank == 1

def test_retrieve_bm25(mock_retriever):
    results = mock_retriever.retrieve_bm25("test query", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk1"
    assert results[0].score == 1.5
    assert results[0].rank == 1
