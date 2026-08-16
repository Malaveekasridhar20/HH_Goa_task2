import os
import json
import time
from collections import defaultdict
import numpy as np
from typing import List, Dict, Set

from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker

def calculate_mrr(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks) / len(ranks)

def calculate_recall(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    retrieved_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 0.0
    return len(retrieved_k.intersection(relevant_ids)) / len(relevant_ids)

def evaluate_retrieval():
    lang = os.getenv("INDEX_LANGUAGE", "hi").lower()
    index_dir = os.path.join(os.path.dirname(__file__), f"../data/indexes/{'english' if lang == 'en' else 'hindi'}")
    
    print(f"=== Evaluating Reranking for '{lang}' ===")
    
    print("Loading FAISS Retriever...")
    t0 = time.time()
    retriever = Retriever(index_dir)
    print(f"Index loaded in {time.time() - t0:.2f}s")
    
    print("Loading CrossEncoder Reranker...")
    t0 = time.time()
    reranker = Reranker()
    print(f"Reranker loaded in {time.time() - t0:.2f}s")
    print(f"Reranker Model: {reranker.model_name}")
    print(f"Reranker Batch Size: {reranker.batch_size}")
    
    print("Loading query texts from raw dataset...")
    queries: Dict[str, str] = {} 
    raw_data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            qid = str(row.get('query_id'))
            query_text = row.get('Eng_Query') if lang == 'en' else row.get('query')
            if query_text:
                queries[qid] = query_text
                
    relevant_mapping: Dict[str, Set[str]] = defaultdict(set)
    for chunk in retriever.metadata.chunks:
        qid = chunk.query_id
        if chunk.is_selected:
            relevant_mapping[qid].add(chunk.chunk_id)
            
    valid_qids = [qid for qid, rels in relevant_mapping.items() if len(rels) > 0 and qid in queries]
    print(f"Found {len(valid_qids)} valid queries with relevant chunks to evaluate.")
    
    use_cap = os.getenv("CAP_QUERY_LENGTH", "0") == "1"
    
    # Store metrics for FAISS Baseline
    faiss_recalls_1 = []
    faiss_recalls_5 = []
    faiss_recalls_10 = []
    faiss_mrrs = []
    
    # Store metrics for Reranker with different candidate_k
    candidate_k_values = [10, 20, 30]
    rerank_recalls_1 = {k: [] for k in candidate_k_values}
    rerank_recalls_5 = {k: [] for k in candidate_k_values}
    rerank_recalls_10 = {k: [] for k in candidate_k_values}
    rerank_mrrs = {k: [] for k in candidate_k_values}
    
    # Latencies
    lat_emb = []
    lat_faiss = []
    lat_rerank = {k: [] for k in candidate_k_values}
    lat_total = {k: [] for k in candidate_k_values}
    
    # Warmup
    warmup_q = queries[valid_qids[0]]
    if use_cap and len(warmup_q) > 500: warmup_q = warmup_q[:500]
    q_emb_warm = retriever.embedding_service.encode_query(warmup_q)
    cands_warm = retriever.retrieve_vector(warmup_q, top_k=30)
    reranker.rerank(warmup_q, cands_warm, top_k=10)
    
    for i, qid in enumerate(valid_qids):
        query_text = queries[qid]
        if use_cap and len(query_text) > 500:
            query_text = query_text[:500]
            
        rels = relevant_mapping[qid]
        
        # 1. Embedding
        t0 = time.time()
        q_emb = retriever.embedding_service.encode_query(query_text)
        dur_emb = time.time() - t0
        lat_emb.append(dur_emb)
        
        # 2. FAISS Retrieval (get 30)
        t0 = time.time()
        cands_all = retriever.retrieve_vector(query_text, top_k=30)
        dur_faiss = time.time() - t0
        lat_faiss.append(dur_faiss)
        
        # Evaluate Baseline (FAISS top 10)
        faiss_top10 = [r.chunk_id for r in cands_all[:10]]
        faiss_recalls_1.append(calculate_recall(faiss_top10, rels, 1))
        faiss_recalls_5.append(calculate_recall(faiss_top10, rels, 5))
        faiss_recalls_10.append(calculate_recall(faiss_top10, rels, 10))
        rr_faiss = 0.0
        for rank, cid in enumerate(faiss_top10, 1):
            if cid in rels:
                rr_faiss = 1.0 / rank
                break
        faiss_mrrs.append(rr_faiss)
        
        # Run reranker once for max candidate_k to save time
        max_cand_k = max(candidate_k_values)
        t0 = time.time()
        cands_for_reranking = cands_all[:max_cand_k]
        
        # We need raw scores to subset properly
        pairs = [(query_text, c.text) for c in cands_for_reranking]
        scores = reranker.model.predict(pairs, batch_size=reranker.batch_size)
        
        for cand_k in candidate_k_values:
            # Subset the raw scores for this cand_k
            subset_cands = cands_for_reranking[:cand_k]
            subset_scores = scores[:cand_k]
            
            # Sort subset
            scored_subset = [{"candidate": c, "score": float(s)} for c, s in zip(subset_cands, subset_scores)]
            scored_subset.sort(key=lambda x: x["score"], reverse=True)
            
            rerank_ids = [item["candidate"].chunk_id for item in scored_subset[:10]]
            
            # Estimate latency by scaling linearly (good enough for rough profiling)
            dur_rerank = (time.time() - t0) * (cand_k / max_cand_k)
            
            lat_rerank[cand_k].append(dur_rerank)
            lat_total[cand_k].append(dur_emb + dur_faiss + dur_rerank)
            
            rerank_recalls_1[cand_k].append(calculate_recall(rerank_ids, rels, 1))
            rerank_recalls_5[cand_k].append(calculate_recall(rerank_ids, rels, 5))
            rerank_recalls_10[cand_k].append(calculate_recall(rerank_ids, rels, 10))
            
            rr = 0.0
            for rank, cid in enumerate(rerank_ids, 1):
                if cid in rels:
                    rr = 1.0 / rank
                    break
            rerank_mrrs[cand_k].append(rr)
            
        import sys
        print(f"  Evaluated {i+1}/{len(valid_qids)} queries...\n", end='')
        sys.stdout.flush()
        
    print("\n")
    
    # Build report
    report = {
        "language": lang,
        "reranker_model": reranker.model_name,
        "queries_evaluated": len(valid_qids),
        "faiss_baseline": {
            "Recall@1": float(np.mean(faiss_recalls_1)),
            "Recall@5": float(np.mean(faiss_recalls_5)),
            "Recall@10": float(np.mean(faiss_recalls_10)),
            "MRR@10": float(np.mean(faiss_mrrs))
        },
        "candidate_k_sweep": {}
    }
    
    best_mrr = -1.0
    best_cand_k = 10
    
    for cand_k in candidate_k_values:
        mrr_val = float(np.mean(rerank_mrrs[cand_k]))
        report["candidate_k_sweep"][str(cand_k)] = {
            "Recall@1": float(np.mean(rerank_recalls_1[cand_k])),
            "Recall@5": float(np.mean(rerank_recalls_5[cand_k])),
            "Recall@10": float(np.mean(rerank_recalls_10[cand_k])),
            "MRR@10": mrr_val,
            "latency_seconds": {
                "reranker_only": {
                    "P50": float(np.percentile(lat_rerank[cand_k], 50)),
                    "P70": float(np.percentile(lat_rerank[cand_k], 70)),
                    "P100": float(np.max(lat_rerank[cand_k]))
                },
                "total_retrieval": {
                    "P50": float(np.percentile(lat_total[cand_k], 50)),
                    "P70": float(np.percentile(lat_total[cand_k], 70)),
                    "P100": float(np.max(lat_total[cand_k]))
                }
            }
        }
        if mrr_val > best_mrr:
            best_mrr = mrr_val
            best_cand_k = cand_k
            
    report["best_candidate_k"] = best_cand_k
    
    # Base latencies
    report["base_latency_seconds"] = {
        "embedding": {
            "P50": float(np.percentile(lat_emb, 50)),
            "P70": float(np.percentile(lat_emb, 70)),
            "P100": float(np.max(lat_emb))
        },
        "faiss_search": {
            "P50": float(np.percentile(lat_faiss, 50)),
            "P70": float(np.percentile(lat_faiss, 70)),
            "P100": float(np.max(lat_faiss))
        }
    }
    
    report_path = os.path.join(index_dir, "reranking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Saved to {report_path}")
    print(f"Best Candidate K: {best_cand_k} (MRR@10: {best_mrr:.4f})")
    
if __name__ == "__main__":
    evaluate_retrieval()
