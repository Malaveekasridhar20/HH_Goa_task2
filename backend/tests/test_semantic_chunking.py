import pytest
from app.chunking.semantic import SentenceAwareChunker

def test_split_into_sentences():
    chunker = SentenceAwareChunker()
    # English + Hindi
    text = "Hello world! How are you? नमस्ते दुनिया। मैं ठीक हूँ।"
    sentences = chunker.split_into_sentences(text)
    
    assert len(sentences) == 4
    assert sentences[0][0] == "Hello world!"
    assert sentences[1][0] == "How are you?"
    assert sentences[2][0] == "नमस्ते दुनिया।"
    assert sentences[3][0] == "मैं ठीक हूँ।"

def test_semantic_grouping():
    # Set target chunk size to 40 so "Short sentence one." (19) + " Short sentence two." (19) fits
    chunker = SentenceAwareChunker(target_chunk_size=40)
    text = "Short sentence one. Short sentence two. A slightly longer sentence that exceeds the target size on its own."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 2
    # The first two sentences group up
    assert chunks[0][0] == "Short sentence one. Short sentence two."
    # The third sentence exceeds target but stays whole
    assert chunks[1][0] == "A slightly longer sentence that exceeds the target size on its own."

def test_semantic_empty():
    chunker = SentenceAwareChunker()
    assert chunker.chunk_text("") == []
