import os
import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = None):
        """
        Initializes the embedding service.
        Defaults to intfloat/multilingual-e5-small unless EMBEDDING_MODEL is set.
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.is_e5 = "e5" in self.model_name.lower()

    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a batch of documents. Applies 'passage: ' prefix if E5 model.
        Returns normalized embeddings suitable for Inner Product search.
        """
        if self.is_e5:
            documents = [f"passage: {doc}" for doc in documents]
            
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query. Applies 'query: ' prefix if E5 model.
        Returns normalized embedding suitable for Inner Product search.
        """
        text = f"query: {query}" if self.is_e5 else query
        
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embedding, dtype=np.float32)
