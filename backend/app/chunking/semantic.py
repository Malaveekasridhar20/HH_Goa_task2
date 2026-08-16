import re
from typing import List, Tuple

class SentenceAwareChunker:
    def __init__(self, target_chunk_size: int = 800):
        self.target_chunk_size = target_chunk_size
        # Regex for sentence boundaries: English (.?!) and Indic danda (।)
        # We use a positive lookbehind and lookahead to keep the punctuation with the sentence
        self.sentence_pattern = re.compile(r'(?<=[.?!।])\s+(?=[A-Z\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F])')

    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits text into sentences based on punctuation, returning (sentence, start, end).
        """
        sentences = []
        start = 0
        
        # We'll iteratively find matches and slice
        for match in self.sentence_pattern.finditer(text):
            end = match.start()
            sentence_text = text[start:end].strip()
            if sentence_text:
                sentences.append((sentence_text, start, end))
            start = match.end()
            
        # Add the final sentence
        if start < len(text):
            sentence_text = text[start:].strip()
            if sentence_text:
                sentences.append((sentence_text, start, len(text)))
                
        return sentences

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Groups sentences to approach target_chunk_size without cutting them mid-sentence.
        Returns a list of tuples: (chunk_text, start_position, end_position)
        """
        if not text:
            return []
            
        sentences = self.split_into_sentences(text)
        if not sentences:
            return [(text, 0, len(text))]
            
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        current_start = -1
        
        for sent_text, sent_start, sent_end in sentences:
            sent_length = len(sent_text)
            
            # Initialize chunk start position
            if current_start == -1:
                current_start = sent_start
                
            # If a single sentence is larger than target, we still keep it together 
            # (fallback strategy in adaptive chunker handles splitting it later if needed)
            if current_length + sent_length > self.target_chunk_size and current_length > 0:
                # Flush current chunk
                current_end = current_chunk_sentences[-1][2]
                chunk_text = text[current_start:current_end].strip()
                if chunk_text:
                    chunks.append((chunk_text, current_start, current_end))
                
                # Start new chunk
                current_chunk_sentences = [(sent_text, sent_start, sent_end)]
                current_length = sent_length
                current_start = sent_start
            else:
                current_chunk_sentences.append((sent_text, sent_start, sent_end))
                current_length += sent_length
                
        # Flush remaining
        if current_chunk_sentences:
            current_end = current_chunk_sentences[-1][2]
            chunk_text = text[current_start:current_end].strip()
            if chunk_text:
                chunks.append((chunk_text, current_start, current_end))
                
        return chunks
