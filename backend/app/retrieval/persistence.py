import json
import os
from typing import List, Dict
from app.retrieval.models import IndexedChunk

class MetadataStore:
    def __init__(self):
        # We store metadata in memory during inference
        # The index maps naturally to the faiss and bm25 indices
        self.chunks: List[IndexedChunk] = []

    def add_chunks(self, chunks: List[IndexedChunk]):
        self.chunks.extend(chunks)

    def get_chunk(self, index: int) -> IndexedChunk:
        return self.chunks[index]

    def save(self, file_path: str):
        """
        Saves the metadata to a JSONL file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + '\n')

    @classmethod
    def load(cls, file_path: str) -> 'MetadataStore':
        """
        Loads metadata from a JSONL file.
        """
        instance = cls()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                instance.chunks.append(IndexedChunk.model_validate_json(line))
        return instance
