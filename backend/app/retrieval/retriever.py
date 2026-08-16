import os
from typing import List
from app.retrieval.models import RetrievalResult
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.bm25_index import BM25Index
from app.retrieval.persistence import MetadataStore

class Retriever:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        
        # Load indexes and metadata
        self.metadata = MetadataStore.load(os.path.join(index_dir, "metadata.jsonl"))
        self.faiss_index = FaissIndex.load(os.path.join(index_dir, "faiss.index"))
        self.bm25_index = BM25Index.load(os.path.join(index_dir, "bm25.pkl"))
        
        # Embedding service
        self.embedding_service = EmbeddingService()

    def retrieve_vector(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieves top_k results using FAISS (Vector search).
        """
        query_embedding = self.embedding_service.encode_query(query)
        scores, indices = self.faiss_index.search(query_embedding, top_k=top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            if idx == -1: # FAISS returns -1 if not enough results
                continue
                
            chunk = self.metadata.get_chunk(idx)
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                rank=rank,
                source_lang=chunk.source_lang,
                is_selected=chunk.is_selected
            ))
            
        return results

    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieves top_k results using BM25 (Lexical search).
        """
        scores, indices = self.bm25_index.search(query, top_k=top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            chunk = self.metadata.get_chunk(idx)
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                rank=rank,
                source_lang=chunk.source_lang,
                is_selected=chunk.is_selected
            ))
            
        return results

    def retrieve_hybrid(self, query: str, top_k: int = 10, dense_weight: float = 0.7, bm25_weight: float = 0.3) -> List[RetrievalResult]:
        """
        Retrieves using both FAISS and BM25, normalizes scores (min-max), and fuses them.
        """
        # Fetch more candidates to ensure good fusion overlap
        fetch_k = max(20, top_k * 2)
        dense_results = self.retrieve_vector(query, top_k=fetch_k)
        bm25_results = self.retrieve_bm25(query, top_k=fetch_k)
        
        # Helper to normalize scores
        def normalize(results):
            if not results: return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            norm_map = {}
            for r in results:
                if max_s > min_s:
                    norm_s = (r.score - min_s) / (max_s - min_s)
                else:
                    norm_s = 1.0 if max_s > 0 else 0.0
                norm_map[r.chunk_id] = (r, norm_s)
            return norm_map

        dense_map = normalize(dense_results)
        bm25_map = normalize(bm25_results)
        
        all_ids = set(dense_map.keys()) | set(bm25_map.keys())
        
        fused = []
        for cid in all_ids:
            d_res, d_norm = dense_map.get(cid, (None, 0.0))
            b_res, b_norm = bm25_map.get(cid, (None, 0.0))
            
            final_score = (dense_weight * d_norm) + (bm25_weight * b_norm)
            
            # Take chunk info from whichever found it
            base_res = d_res if d_res else b_res
            
            fused.append(RetrievalResult(
                chunk_id=base_res.chunk_id,
                text=base_res.text,
                score=final_score,
                rank=0, # Will set after sort
                source_lang=base_res.source_lang,
                is_selected=base_res.is_selected
            ))
            
        # Sort by final fused score
        fused.sort(key=lambda x: x.score, reverse=True)
        
        # Re-assign ranks
        for i, res in enumerate(fused):
            res.rank = i + 1
            
        return fused[:top_k]

