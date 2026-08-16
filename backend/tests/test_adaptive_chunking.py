import pytest
from app.chunking.adaptive import AdaptiveChunker

def test_adaptive_router_short():
    chunker = AdaptiveChunker(short_threshold=50)
    text = "This is a short text."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 1
    assert chunks[0][3] == "whole_passage"

def test_adaptive_router_medium():
    chunker = AdaptiveChunker(short_threshold=10, semantic_target=20)
    text = "First sentence. Second sentence."
    chunks = chunker.chunk_text(text)
    
    # Should use semantic chunking
    assert chunks[0][3] == "semantic"

def test_adaptive_router_fallback():
    chunker = AdaptiveChunker(short_threshold=10, long_threshold=20, semantic_target=100)
    # A single sentence that exceeds long_threshold without any punctuation to break on
    text = "ThisIsAReallyLongStringWithNoPunctuationThatWillTriggerTheFallbackStrategy"
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 1
    assert chunks[0][3] == "sliding_window"
