from typing import List, Tuple
from app.chunking.fixed_window import FixedWindowChunker
from app.chunking.semantic import SentenceAwareChunker

class AdaptiveChunker:
    def __init__(self, short_threshold: int = 300, long_threshold: int = 1500, semantic_target: int = 800):
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        self.semantic_chunker = SentenceAwareChunker(target_chunk_size=semantic_target)
        self.fixed_chunker = FixedWindowChunker(chunk_size=long_threshold, overlap=long_threshold // 4)

    def chunk_text(self, text: str) -> List[Tuple[str, int, int, str, str]]:
        """
        Routes the text to the appropriate chunking strategy.
        Returns tuples: (chunk_text, start, end, strategy, strategy_reason)
        """
        text_length = len(text)
        
        # Strategy A: Whole Passage
        if text_length <= self.short_threshold:
            return [(text, 0, text_length, "whole_passage", "passage below short threshold")]
            
        # Strategy B & C routing
        # Try semantic first
        semantic_chunks = self.semantic_chunker.chunk_text(text)
        
        final_chunks = []
        for c_text, c_start, c_end in semantic_chunks:
            # Strategy D Fallback for super long sentences
            if len(c_text) > self.long_threshold:
                # If a single sentence or unbreakable block is too long, fallback to sliding window
                sub_chunks = self.fixed_chunker.chunk_text(c_text)
                for sub_text, sub_start, sub_end in sub_chunks:
                    # Adjust relative offsets to absolute offsets
                    abs_start = c_start + sub_start
                    abs_end = c_start + sub_end
                    final_chunks.append((
                        sub_text, abs_start, abs_end, 
                        "sliding_window", 
                        "fallback for sentence exceeding long threshold"
                    ))
            else:
                final_chunks.append((
                    c_text, c_start, c_end, 
                    "semantic", 
                    "medium passage with detectable sentence boundaries"
                ))
                
        return final_chunks
