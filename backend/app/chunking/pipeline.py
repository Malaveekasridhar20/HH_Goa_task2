import hashlib
from typing import List, Dict, Any
from app.chunking.models import Passage, Chunk
from app.chunking.passage_extractor import PassageExtractor
from app.chunking.adaptive import AdaptiveChunker
from app.chunking.fixed_window import FixedWindowChunker

class ChunkingPipeline:
    def __init__(self, adaptive: bool = True):
        self.extractor = PassageExtractor()
        self.adaptive = adaptive
        self.adaptive_chunker = AdaptiveChunker()
        self.naive_chunker = FixedWindowChunker(chunk_size=800, overlap=200)

    def _generate_chunk_id(self, query_id: str, passage_index: int, chunk_index: int) -> str:
        """Generates a deterministic chunk ID."""
        raw_id = f"{query_id}_{passage_index}_{chunk_index}"
        return hashlib.md5(raw_id.encode('utf-8')).hexdigest()

    def process_record(self, record: Dict[str, Any], extract_english: bool = False) -> List[Chunk]:
        """
        Process a single dataset record and generate chunks.
        """
        chunks = []
        passages = self.extractor.extract_passages(record, extract_english=extract_english)
        chunks = []
        
        for p in passages:
            if self.adaptive:
                raw_chunks = self.adaptive_chunker.chunk_text(p.text)
            else:
                # Naive fixed window baseline for comparison
                naive_res = self.naive_chunker.chunk_text(p.text)
                raw_chunks = [(txt, st, en, "naive_fixed", "naive baseline") for txt, st, en in naive_res]
                
            for c_idx, (c_text, c_start, c_end, c_strat, c_reason) in enumerate(raw_chunks):
                chunk_id = self._generate_chunk_id(p.query_id, p.passage_index, c_idx)
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=c_text,
                    query_id=p.query_id,
                    passage_index=p.passage_index,
                    chunk_index=c_idx,
                    is_selected=p.is_selected,
                    source_lang=p.source_lang,
                    target_lang=p.target_lang,
                    query_type=p.query_type,
                    strategy=c_strat,
                    strategy_reason=c_reason,
                    start_position=c_start,
                    end_position=c_end
                ))
                
        return chunks
