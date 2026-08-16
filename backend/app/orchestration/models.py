from pydantic import BaseModel
from typing import List, Optional

class VoiceRAGRequest(BaseModel):
    audio_data: bytes
    language_hint: Optional[str] = None
    top_k: int = 3
    # Use 'extractive' by default to meet the latency requirement.
    generation_mode: str = "extractive" 

class VoiceRAGResponse(BaseModel):
    success: bool
    transcript: str
    answer: str
    source_chunk_ids: List[str]
    language: Optional[str] = None
    stt_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    error: Optional[str] = None
