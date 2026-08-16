import os
import pickle
from typing import List
from rank_bm25 import BM25Okapi
import re

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.doc_count = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text for BM25.
        Converts to lowercase and splits by non-alphanumeric characters,
        but crucially preserves Unicode alphanumeric characters for Hindi/Indic support.
        """
        text = text.lower()
        import string
        tokens = text.split()
        return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]

    def build_index(self, documents: List[str]):
        """
        Builds the BM25 index from a list of text documents.
        """
        tokenized_corpus = [self.tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_count = len(documents)

    def search(self, query: str, top_k: int = 10):
        """
        Searches the BM25 index.
        Returns (scores, indices).
        """
        if not self.bm25:
            raise ValueError("BM25 index not built or loaded.")
            
        tokenized_query = self.tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        # We need to handle cases where top_k > len(doc_scores)
        top_k = min(top_k, len(doc_scores))
        
        # Argsort in descending order
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        top_scores = [doc_scores[i] for i in top_indices]
        
        return top_scores, top_indices

    def save(self, file_path: str):
        """
        Saves the BM25 index to disk using pickle.
        """
        if not self.bm25:
            raise ValueError("Cannot save empty BM25 index.")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, file_path: str) -> 'BM25Index':
        """
        Loads a BM25 index from disk.
        """
        with open(file_path, 'rb') as f:
            bm25 = pickle.load(f)
            
        instance = cls()
        instance.bm25 = bm25
        instance.doc_count = bm25.corpus_size
        return instance
