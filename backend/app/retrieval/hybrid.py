import os
from typing import List, Dict, Optional
from app.retrieval.retriever import Retriever
from app.retrieval.models import HybridRetrievalResult

class HybridRetriever:
    def __init__(self, index_dir: str):
        self.retriever = Retriever(index_dir)
        # Default alpha
        self.alpha = float(os.getenv("HYBRID_ALPHA", "0.6"))

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Applies Min-Max normalization to a list of scores.
        Handles the edge case where max_score == min_score gracefully.
        """
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # If all scores are identical, normalized score is 1.0.
            return [1.0] * len(scores)
            
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def retrieve_hybrid(self, query: str, top_k: int = 10, candidate_k: int = 20, alpha: Optional[float] = None) -> List[HybridRetrievalResult]:
        """
        Performs hybrid retrieval using both FAISS and BM25.
        Deduplicates candidates and fuses scores.
        """
        if alpha is None:
            alpha = self.alpha
            
        # 1. Encode query once
        # Using the underlying embedding service from the Retriever
        query_embedding = self.retriever.embedding_service.encode_query(query)
        
        # 2. Retrieve top candidate_k from FAISS (using pre-encoded query directly)
        faiss_scores, faiss_indices = self.retriever.faiss_index.search(query_embedding, top_k=candidate_k)
        
        # 3. Retrieve top candidate_k from BM25 (using raw string)
        bm25_scores, bm25_indices = self.retriever.bm25_index.search(query, top_k=candidate_k)
        
        # We need to filter out -1 indices from FAISS (which means not enough results)
        valid_faiss = [(score, idx) for score, idx in zip(faiss_scores, faiss_indices) if idx != -1]
        if valid_faiss:
            faiss_scores, faiss_indices = zip(*valid_faiss)
        else:
            faiss_scores, faiss_indices = [], []
            
        # 4. Normalize scores independently
        faiss_norm_scores = self._normalize_scores(list(faiss_scores))
        bm25_norm_scores = self._normalize_scores(list(bm25_scores))
        
        # 5. Fuse candidates & 6. Deduplicate by chunk_id
        candidates = {} # chunk_id -> dict of properties
        
        # Add FAISS results
        for idx, raw_score, norm_score in zip(faiss_indices, faiss_scores, faiss_norm_scores):
            chunk = self.retriever.metadata.get_chunk(idx)
            candidates[chunk.chunk_id] = {
                "chunk": chunk,
                "vector_score": float(raw_score),
                "bm25_score": 0.0,
                "normalized_vector_score": float(norm_score),
                "normalized_bm25_score": 0.0
            }
            
        # Add/Merge BM25 results
        for idx, raw_score, norm_score in zip(bm25_indices, bm25_scores, bm25_norm_scores):
            chunk = self.retriever.metadata.get_chunk(idx)
            if chunk.chunk_id in candidates:
                candidates[chunk.chunk_id]["bm25_score"] = float(raw_score)
                candidates[chunk.chunk_id]["normalized_bm25_score"] = float(norm_score)
            else:
                candidates[chunk.chunk_id] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "bm25_score": float(raw_score),
                    "normalized_vector_score": 0.0,
                    "normalized_bm25_score": float(norm_score)
                }
                
        # 7. Rank by fused score
        results_list = []
        for chunk_id, data in candidates.items():
            fused_score = alpha * data["normalized_vector_score"] + (1.0 - alpha) * data["normalized_bm25_score"]
            data["hybrid_score"] = fused_score
            results_list.append(data)
            
        # Sort descending by hybrid score
        results_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # 8. Return top_k
        final_results = []
        for rank, data in enumerate(results_list[:top_k], 1):
            chunk = data["chunk"]
            final_results.append(HybridRetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=data["vector_score"],
                hybrid_score=data["hybrid_score"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                normalized_vector_score=data["normalized_vector_score"],
                normalized_bm25_score=data["normalized_bm25_score"],
                rank=rank,
                source_lang=chunk.source_lang,
                is_selected=chunk.is_selected
            ))
            
        return final_results
