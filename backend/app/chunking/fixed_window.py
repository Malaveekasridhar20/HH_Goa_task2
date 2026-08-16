from typing import List, Tuple

class FixedWindowChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits text using a sliding window.
        Returns a list of tuples: (chunk_text, start_position, end_position)
        """
        if not text:
            return []
            
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Extract chunk
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))
            
            if end == text_length:
                break
            
            # Move start pointer for the next chunk
            start += (self.chunk_size - self.overlap)
            
            # Avoid infinite loop if overlap >= chunk_size
            if self.chunk_size <= self.overlap:
                break
                
        return chunks
