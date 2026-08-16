import pytest
from app.generation.extractive_generator import ExtractiveAnswerGenerator
from app.retrieval.models import RetrievalResult

def test_sentence_splitting():
    generator = ExtractiveAnswerGenerator()
    
    # English
    en_text = "Hello world. How are you? I am fine! Great."
    sents = generator._split_sentences(en_text)
    assert len(sents) == 4
    assert sents[0] == "Hello world."
    assert sents[1] == "How are you?"
    
    # Hindi
    hi_text = "नमस्ते दुनिया। आप कैसे हैं? मैं ठीक हूँ!"
    sents = generator._split_sentences(hi_text)
    assert len(sents) == 3
    assert sents[0] == "नमस्ते दुनिया।"
    assert sents[1] == "आप कैसे हैं?"

def test_extractive_generation_empty():
    generator = ExtractiveAnswerGenerator()
    resp = generator.generate("What is the capital of France?", [])
    assert "I don't have enough information" in resp.answer
    assert resp.source_chunk_ids == []

def test_extractive_generation_success():
    generator = ExtractiveAnswerGenerator()
    chunks = [
        RetrievalResult(chunk_id="chunk1", text="Paris is the capital of France. It is very beautiful.", score=0.9, rank=1, source_lang="en", is_selected=True),
        RetrievalResult(chunk_id="chunk2", text="London is the capital of England.", score=0.8, rank=2, source_lang="en", is_selected=True)
    ]
    resp = generator.generate("What is the capital of France?", chunks)
    assert resp.answer == "Paris is the capital of France."
    assert resp.source_chunk_ids == ["chunk1"]

def test_extractive_generation_safe_refusal():
    generator = ExtractiveAnswerGenerator()
    chunks = [
        RetrievalResult(chunk_id="chunk1", text="The sky is blue today. The weather is nice.", score=0.9, rank=1, source_lang="en", is_selected=True)
    ]
    resp = generator.generate("What is the current barometric pressure?", chunks)
    assert "I don't have enough information" in resp.answer
    assert resp.source_chunk_ids == []

def test_source_id_preservation():
    generator = ExtractiveAnswerGenerator()
    chunks = [
        RetrievalResult(chunk_id="abc-123", text="Apples are red. Bananas are yellow.", score=0.9, rank=1, source_lang="en", is_selected=True)
    ]
    resp = generator.generate("What color are apples? Are they red?", chunks)
    assert "Apples are red" in resp.answer
    assert resp.source_chunk_ids == ["abc-123"]
