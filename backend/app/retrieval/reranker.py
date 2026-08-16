import os
from typing import List
from sentence_transformers import CrossEncoder
from app.retrieval.models import RetrievalResult, RerankedResult

class Reranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        # Load the CrossEncoder model. sentence-transformers will automatically use CPU if GPU is not available.
        self.model = CrossEncoder(self.model_name, max_length=128)
        self.batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RerankedResult]:
        """
        Reranks a list of retrieval candidates using a Cross-Encoder.
        """
        if not candidates:
            return []

        # Construct pairs for the cross-encoder: (query, passage_text)
        pairs = [(query, candidate.text) for candidate in candidates]
        
        # Batch inference
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Associate scores with candidates
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            scored_candidates.append({
                "candidate": candidate,
                "score": float(scores[i])
            })
            
        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k
        reranked_results = []
        for new_rank, item in enumerate(scored_candidates[:top_k], 1):
            cand = item["candidate"]
            reranked_results.append(RerankedResult(
                chunk_id=cand.chunk_id,
                text=cand.text,
                rerank_score=item["score"],
                original_rank=cand.rank,
                final_rank=new_rank,
                source_lang=cand.source_lang,
                is_selected=cand.is_selected
            ))
            
        return reranked_results
