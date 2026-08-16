import pytest
from app.generation.generator import AnswerGenerator, Provider
from app.retrieval.models import RetrievalResult
from app.generation.models import GenerationResponse
from app.generation.prompts import SYSTEM_PROMPT
import json

class MockProvider(Provider):
    def __init__(self, response_data):
        self.response_data = response_data
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.model_name = "mock-model"
        
    def generate(self, system_prompt: str, user_prompt: str):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return {
            "content": json.dumps(self.response_data),
            "latency": 0.1,
            "model": self.model_name
        }

def test_empty_context():
    generator = AnswerGenerator(provider=MockProvider({"answer": "Should not be called"}))
    response = generator.generate("What is X?", [])
    
    assert response.answer == "I don't have enough information in the retrieved context to answer that."
    assert response.source_chunk_ids == []
    assert response.generation_latency == 0.0

def test_context_formatting():
    mock_provider = MockProvider({
        "answer": "This is the answer.",
        "source_chunk_ids": ["c1", "c2"]
    })
    generator = AnswerGenerator(provider=mock_provider)
    generator.top_k = 2
    
    chunks = [
        RetrievalResult(chunk_id="c1", text="Text 1", document_id="d1", is_selected=True, score=0.9, rank=1, source_lang="en"),
        RetrievalResult(chunk_id="c2", text="Text 2", document_id="d2", is_selected=True, score=0.8, rank=2, source_lang="en"),
        RetrievalResult(chunk_id="c3", text="Text 3", document_id="d3", is_selected=True, score=0.7, rank=3, source_lang="en")
    ]
    
    response = generator.generate("What is X?", chunks)
    
    assert response.answer == "This is the answer."
    assert response.source_chunk_ids == ["c1", "c2"]
    assert response.model == "mock-model"
    
    # Check that it truncated to top_k=2
    assert "[Source ID: c1]" in mock_provider.last_user_prompt
    assert "[Source ID: c2]" in mock_provider.last_user_prompt
    assert "[Source ID: c3]" not in mock_provider.last_user_prompt
    
    assert "User Query: What is X?" in mock_provider.last_user_prompt
    assert mock_provider.last_system_prompt == SYSTEM_PROMPT

def test_json_parsing_failure():
    class BadMockProvider(Provider):
        def generate(self, system_prompt, user_prompt):
            return {"content": "This is not JSON", "latency": 0.1, "model": "bad-model"}
            
    generator = AnswerGenerator(provider=BadMockProvider())
    chunks = [RetrievalResult(chunk_id="c1", text="Text", document_id="d1", is_selected=True, score=0.9, rank=1, source_lang="en")]
    
    response = generator.generate("What?", chunks)
    assert response.answer == "I don't have enough information in the retrieved context to answer that."
    assert response.source_chunk_ids == []
