import pytest
from app.chunking.adaptive import AdaptiveChunker

def test_text_preservation_semantic():
    chunker = AdaptiveChunker(short_threshold=50, long_threshold=200, semantic_target=100)
    text = "This is the first sentence. This is the second sentence. And here is the third one that might trigger a split."
    chunks = chunker.chunk_text(text)
    
    # Reconstruct
    reconstructed = ""
    for idx, (c_text, start, end, strat, reason) in enumerate(chunks):
        # We assume chunks are returned in order and contiguous for semantic chunking
        if idx > 0:
            # Add spaces if there were gaps between sentences that were stripped,
            # but start/end positions reflect original string.
            # To properly test, we should slice from original text using start/end
            pass
            
    # The true test of preservation is if the slices match the original text
    sliced_reconstruction = ""
    last_end = 0
    for c_text, start, end, strat, reason in chunks:
        # Check that the extracted text matches the slice
        assert text[start:end].strip() == c_text.strip()
        # Check that we didn't skip any non-whitespace characters between chunks
        gap = text[last_end:start].strip()
        assert gap == "", f"Found lost text in gap: '{gap}'"
        last_end = end
        
    # Check that we reached the end (excluding trailing whitespace)
    assert text[last_end:].strip() == ""

def test_text_preservation_sliding_window():
    chunker = AdaptiveChunker(short_threshold=10, long_threshold=50, semantic_target=20)
    # A single very long string with no punctuation
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    chunks = chunker.chunk_text(text)
    
    # For sliding window, they overlap.
    # We verify that union of all chunks covers the original text
    covered = [False] * len(text)
    for c_text, start, end, strat, reason in chunks:
        for i in range(start, end):
            covered[i] = True
            
    assert all(covered), "Not all characters were covered by overlapping chunks"
