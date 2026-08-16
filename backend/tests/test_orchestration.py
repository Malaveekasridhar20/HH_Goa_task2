import pytest
from unittest.mock import MagicMock

from app.orchestration.models import VoiceRAGRequest
from app.orchestration.pipeline import VoiceRAGPipeline
from app.stt.models import SpeechToTextResponse
from app.retrieval.models import RetrievalResult
from app.generation.models import GenerationResponse

def test_pipeline_empty_audio():
    pipeline = VoiceRAGPipeline(stt_service=MagicMock(), en_retriever=MagicMock(), hi_retriever=MagicMock(), extractive_generator=MagicMock())
    req = VoiceRAGRequest(audio_data=b"")
    resp = pipeline.execute(req)
    
    assert not resp.success
    assert "Empty audio" in resp.error

def test_pipeline_stt_failure():
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = SpeechToTextResponse(
        transcript="", provider="test", latency=0.1, success=False, error="API down"
    )
    
    pipeline = VoiceRAGPipeline(stt_service=mock_stt, en_retriever=MagicMock(), hi_retriever=MagicMock(), extractive_generator=MagicMock())
    req = VoiceRAGRequest(audio_data=b"dummy")
    resp = pipeline.execute(req)
    
    assert not resp.success
    assert "STT failed" in resp.error

def test_pipeline_empty_transcript():
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = SpeechToTextResponse(
        transcript=" ", provider="test", latency=0.1, success=True
    )
    
    pipeline = VoiceRAGPipeline(stt_service=mock_stt, en_retriever=MagicMock(), hi_retriever=MagicMock(), extractive_generator=MagicMock())
    req = VoiceRAGRequest(audio_data=b"dummy")
    resp = pipeline.execute(req)
    
    assert not resp.success
    assert "couldn't hear" in resp.answer
    assert "Empty or extremely short" in resp.error

def test_pipeline_success_extractive():
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = SpeechToTextResponse(
        transcript="What color are apples?", detected_language="en", provider="test", latency=0.1, success=True
    )
    
    mock_retriever = MagicMock()
    mock_retriever.retrieve_vector.return_value = [
        RetrievalResult(chunk_id="c1", text="Apples are red.", score=0.9, rank=1, source_lang="en", is_selected=True)
    ]
    
    mock_gen = MagicMock()
    mock_gen.generate.return_value = GenerationResponse(
        answer="Apples are red.", source_chunk_ids=["c1"], model="extractive", generation_latency=0.05
    )
    
    pipeline = VoiceRAGPipeline(
        stt_service=mock_stt, 
        en_retriever=mock_retriever, 
        hi_retriever=MagicMock(), 
        extractive_generator=mock_gen
    )
    
    req = VoiceRAGRequest(audio_data=b"dummy", generation_mode="extractive")
    resp = pipeline.execute(req)
    
    assert resp.success
    assert resp.transcript == "What color are apples?"
    assert resp.answer == "Apples are red."
    assert resp.source_chunk_ids == ["c1"]
    assert resp.stt_latency_ms >= 0
    assert resp.retrieval_latency_ms >= 0
    assert resp.generation_latency_ms >= 0
    assert resp.total_latency_ms >= 0
