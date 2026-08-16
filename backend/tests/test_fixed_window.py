import pytest
from app.chunking.fixed_window import FixedWindowChunker

def test_fixed_window():
    chunker = FixedWindowChunker(chunk_size=10, overlap=5)
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 5
    assert chunks[0][0] == "ABCDEFGHIJ"
    assert chunks[1][0] == "FGHIJKLMNO"
    assert chunks[2][0] == "KLMNOPQRST"
    assert chunks[3][0] == "PQRSTUVWXY"
    assert chunks[4][0] == "UVWXYZ" # The remainder

def test_fixed_window_small_text():
    chunker = FixedWindowChunker(chunk_size=100, overlap=20)
    text = "Short text."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0][0] == text

def test_fixed_window_empty():
    chunker = FixedWindowChunker()
    assert chunker.chunk_text("") == []
