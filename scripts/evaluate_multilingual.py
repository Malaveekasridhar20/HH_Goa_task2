import os
import sys
import json
import time
import numpy as np

# Fix unicode printing in Windows console
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app.retrieval.retriever import Retriever

def calculate_mrr(ranks):
    if not ranks: return 0.0
    return np.mean([1.0 / r if r > 0 else 0.0 for r in ranks])

def evaluate_language(lang_name: str, lang_code: str):
    print(f"\n=== Evaluating {lang_name.upper()} Retrieval ===")
    
    # 1. Load ground truth
    val_file = f"hinval_500.jsonl" if lang_name == "hindi" else f"english_chunks.jsonl" # wait, we evaluate on validation files
    # Actually, the validation files are:
    # english: hinval_500.jsonl has Eng_Query and Eng_Answer and source_lang=="en". Actually no, let's just use the generated jsonl files for queries.
    
    if lang_name == "english":
        val_file = "../data/processed/hinval_500.jsonl" # the english queries are in Eng_Query
    elif lang_name == "hindi":
        val_file = "../data/processed/hinval_500.jsonl"
    else:
        val_file = f"../data/processed/{lang_name}_validation_500.jsonl"
        
    val_path = os.path.abspath(os.path.join(os.path.dirname(__file__), val_file))
    
    if not os.path.exists(val_path):
        print(f"Validation file {val_path} not found.")
        return None
        
    # 2. Load Retriever
    index_dir = f"data/indexes/{lang_name}"
    try:
        retriever = Retriever(index_dir=index_dir)
    except Exception as e:
        print(f"Failed to load retriever for {lang_name}: {e}")
        return None
        
    # 3. Read queries
    queries = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 100: # evaluate 100 queries for speed
                break
            try:
                row = json.loads(line)
                if lang_name == "english":
                    query = row.get("Eng_Query", row.get("query"))
                else:
                    query = row.get("query")
                    
                target_passages = row.get("passages", {}).get("Translated_passages", [])
                if lang_name == "english":
                    target_passages = row.get("passages", {}).get("English_passages", [])
                    
                is_selected = row.get("passages", {}).get("is_selected", [])
                
                # Find the golden passage that answers the query
                golden_passage = None
                for p, s in zip(target_passages, is_selected):
                    if s == 1:
                        golden_passage = p
                        break
                        
                if query and golden_passage:
                    queries.append({
                        "query": query,
                        "golden": golden_passage
                    })
            except:
                pass
                
    if not queries:
        print("No queries found.")
        return None
        
    print(f"Loaded {len(queries)} queries for evaluation.")
    
    # 4. Evaluate
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    
    latencies = []
    
    for q in queries:
        t0 = time.time()
        results = retriever.retrieve_vector(q["query"], top_k=10)
        latencies.append((time.time() - t0) * 1000)
        
        # Check if golden is in results text
        # Since chunks might be partial, we check if result text is in golden or golden in result text
        # or just exact chunk match if we had IDs. We just do string overlap.
        golden = q["golden"].strip()
        rank = 0
        for i, res in enumerate(results):
            # strict inclusion or significant overlap
            if res.text in golden or golden in res.text or len(set(res.text.split()) & set(golden.split())) > 5:
                rank = i + 1
                break
                
        ranks.append(rank)
        if rank == 1: hits_at_1 += 1
        if 1 <= rank <= 5: hits_at_5 += 1
        if 1 <= rank <= 10: hits_at_10 += 1
        
    stats = {
        "queries": len(queries),
        "recall_1": hits_at_1 / len(queries),
        "recall_5": hits_at_5 / len(queries),
        "recall_10": hits_at_10 / len(queries),
        "mrr_10": calculate_mrr(ranks),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p70_latency_ms": np.percentile(latencies, 70),
        "p100_latency_ms": np.percentile(latencies, 100)
    }
    
    print(f"Recall@1: {stats['recall_1']:.4f}")
    print(f"Recall@5: {stats['recall_5']:.4f}")
    print(f"Recall@10: {stats['recall_10']:.4f}")
    print(f"MRR@10: {stats['mrr_10']:.4f}")
    
    return stats

def main():
    langs = {
        "english": "en",
        "hindi": "hi",
        "tamil": "ta",
        "telugu": "te",
        "malayalam": "ml"
    }
    
    report = {}
    for name, code in langs.items():
        stats = evaluate_language(name, code)
        if stats:
            report[name] = stats
            
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, "multilingual_retrieval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
