import os
import json
import time
import numpy as np
from collections import defaultdict
from app.retrieval.retriever import Retriever
from app.generation.generator import AnswerGenerator
from dotenv import load_dotenv

import sys

load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))

def evaluate_generation(language: str):
    print(f"\n=== Evaluating Answer Generation for '{language}' ===")
    sys.stdout.flush()
    lang_dir = "english" if language == "en" else "hindi"
    index_dir = os.path.join(os.path.dirname(__file__), f"../data/indexes/{lang_dir}")
    retriever = Retriever(index_dir)
    generator = AnswerGenerator()
    
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    queries = {}
    
    print("Loading queries...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            q = row.get("query") if language == "hi" else row.get("Eng_Query")
            if q: queries[str(row.get('query_id'))] = q
            
    # Sample 20 queries for manual auditing
    qids = list(queries.keys())[:20]
    
    lat_retrieval = []
    lat_generation = []
    lat_total = []
    
    results = []
    
    for i, qid in enumerate(qids):
        query_text = queries[qid]
        
        t0 = time.time()
        cands = retriever.retrieve_vector(query_text, top_k=10)
        dur_ret = time.time() - t0
        
        t1 = time.time()
        response = generator.generate(query_text, cands, language=language)
        dur_gen = time.time() - t1
        
        dur_tot = dur_ret + dur_gen
        
        lat_retrieval.append(dur_ret)
        lat_generation.append(dur_gen)
        lat_total.append(dur_tot)
        
        results.append({
            "query_id": qid,
            "query": query_text,
            "answer": response.answer,
            "source_chunk_ids": response.source_chunk_ids,
            "model": response.model,
            "latencies": {
                "retrieval": dur_ret,
                "generation": dur_gen,
                "total": dur_tot
            }
        })
        
        print(f"  Evaluated {i+1}/20 queries...")
        sys.stdout.flush()
        
    print(f"\n--- Latency ({language}) ---")
    print(f"Retrieval P50: {np.percentile(lat_retrieval, 50):.3f}s")
    print(f"Retrieval P100: {np.max(lat_retrieval):.3f}s")
    print(f"Generation P50: {np.percentile(lat_generation, 50):.3f}s")
    print(f"Generation P100: {np.max(lat_generation):.3f}s")
    print(f"Total P50: {np.percentile(lat_total, 50):.3f}s")
    print(f"Total P100: {np.max(lat_total):.3f}s")
    
    out_path = os.path.join(index_dir, "generation_audit.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved audit log to {out_path}")

if __name__ == "__main__":
    evaluate_generation("en")
    # evaluate_generation("hi")
