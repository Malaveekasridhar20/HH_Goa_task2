import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, dimension: int):
        """
        Initializes an empty FAISS IndexFlatIP.
        We use Inner Product (IP) because embeddings are normalized (Cosine Similarity).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Adds normalized embeddings to the FAISS index.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}")
        
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """
        Searches the index for the nearest neighbors of the query_embedding.
        Returns (distances, indices).
        """
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

    def save(self, file_path: str):
        """
        Saves the FAISS index to disk.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        faiss.write_index(self.index, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'FaissIndex':
        """
        Loads a FAISS index from disk.
        """
        index = faiss.read_index(file_path)
        instance = cls(dimension=index.d)
        instance.index = index
        return instance
