import pytest
from app.orchestration.pipeline import VoiceRAGPipeline
from app.stt.sarvam import SarvamSTTProvider
from app.retrieval.retriever import Retriever
import os

# Create mock retrievers that don't load FAISS to run tests quickly
class MockRetriever:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.embedding_service = None

def test_language_routing():
    # Provide mock retrievers
    pipeline = VoiceRAGPipeline(
        en_retriever=MockRetriever("data/indexes/english"),
        hi_retriever=MockRetriever("data/indexes/hindi"),
        ta_retriever=MockRetriever("data/indexes/tamil"),
        te_retriever=MockRetriever("data/indexes/telugu"),
        ml_retriever=MockRetriever("data/indexes/malayalam")
    )
    
    # Test internal routing mechanism based on logic in VoiceRAGPipeline.execute
    # Simulate a partial request
    # Instead of a full execute, we can just test if the pipeline has the retrievers
    assert pipeline.en_retriever.index_dir == "data/indexes/english"
    assert pipeline.ta_retriever.index_dir == "data/indexes/tamil"
    assert pipeline.te_retriever.index_dir == "data/indexes/telugu"
    assert pipeline.ml_retriever.index_dir == "data/indexes/malayalam"
    
def test_stt_language_mapping():
    provider = SarvamSTTProvider()
    
    # We test mapping
    assert provider._get_language_code("en") == "en-IN"
    assert provider._get_language_code("hi") == "hi-IN"
    assert provider._get_language_code("ta") == "ta-IN"
    assert provider._get_language_code("te") == "te-IN"
    assert provider._get_language_code("ml") == "ml-IN"
    # fallback
    assert provider._get_language_code("unknown") == "en-IN"
