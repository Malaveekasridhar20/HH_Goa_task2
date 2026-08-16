from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class IndexedChunk(BaseModel):
    """
    Structured model representing an indexed chunk and its metadata.
    This preserves the original Phase 2 chunk structure.
    """
    chunk_id: str
    text: str
    query_id: str
    query: Optional[str] = None
    eng_query: Optional[str] = None
    source_lang: str
    target_lang: str
    query_type: str
    passage_index: int
    chunk_index: int
    is_selected: bool
    strategy: str
    strategy_reason: str
    start_position: int
    end_position: int
    
    # Optional dynamic metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    """
    Structured result returned by the retriever.
    """
    chunk_id: str
    text: str
    score: float
    rank: int
    source_lang: str
    is_selected: bool

class HybridRetrievalResult(BaseModel):
    """
    Structured result returned by the hybrid retriever.
    """
    chunk_id: str
    text: str
    hybrid_score: float
    vector_score: float
    bm25_score: float
    normalized_vector_score: float
    normalized_bm25_score: float
    rank: int
    source_lang: str
    is_selected: bool

class RerankedResult(BaseModel):
    """
    Structured result returned by the reranker.
    """
    chunk_id: str
    text: str
    rerank_score: float
    original_rank: int
    final_rank: int
    source_lang: str
    is_selected: bool
