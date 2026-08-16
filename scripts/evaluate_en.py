import time
import json
import numpy as np
from app.retrieval.retriever import Retriever
from app.generation.generator import AnswerGenerator
from dotenv import load_dotenv
import os
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))

def evaluate_generation(language: str):
    print(f"\n=== Evaluating Answer Generation for '{language}' ===")
    sys.stdout.flush()
    
    retriever = Retriever(index_dir="data/indexes")
    retriever.load_indexes()
    
    generator = AnswerGenerator()
    
    with open(f"data/test_{language}.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print("Loading queries...")
    sys.stdout.flush()
    
    audit_log = []
    
    latencies_retrieval = []
    latencies_generation = []
    latencies_total = []
    
    # Process exact 20 requests
    for i, q in enumerate(queries[:20]):
        query_id = q.get("query_id", str(i))
        query_text = q.get("query", "")
        
        t0 = time.time()
        results = retriever.search(query_text, language=language, k=5, alpha=1.0)
        t1 = time.time()
        
        generated = generator.generate_answer(query_text, results)
        t2 = time.time()
        
        retrieval_ms = t1 - t0
        generation_ms = t2 - t1
        total_ms = t2 - t0
        
        latencies_retrieval.append(retrieval_ms)
        latencies_generation.append(generation_ms)
        latencies_total.append(total_ms)
        
        audit_log.append({
            "query_id": query_id,
            "query": query_text,
            "answer": generated.answer,
            "source_chunk_ids": generated.source_chunk_ids,
            "model": generator.provider.model_name,
            "latencies": {
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": total_ms
            }
        })
        
        print(f"  Evaluated {i+1}/20 queries...")
        sys.stdout.flush()
        
    lat_r = np.array(latencies_retrieval)
    lat_g = np.array(latencies_generation)
    lat_t = np.array(latencies_total)
    
    print(f"\n--- Latency ({language}) ---")
    print(f"Retrieval P50: {np.percentile(lat_r, 50):.3f}s")
    print(f"Retrieval P100: {np.percentile(lat_r, 100):.3f}s")
    print(f"Generation P50: {np.percentile(lat_g, 50):.3f}s")
    print(f"Generation P100: {np.percentile(lat_g, 100):.3f}s")
    print(f"Total P50: {np.percentile(lat_t, 50):.3f}s")
    print(f"Total P100: {np.percentile(lat_t, 100):.3f}s")
    
    output_path = os.path.join("data", "indexes", "english" if language == "en" else "hindi", "generation_audit.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)
    print(f"Saved audit log to {output_path}")

if __name__ == "__main__":
    evaluate_generation("en")
