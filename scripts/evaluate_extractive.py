import os
import sys
import time
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))
from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

def run_evaluation():
    en_retriever = Retriever(index_dir=os.path.join(os.path.dirname(__file__), "../data/indexes/english"))
    hi_retriever = Retriever(index_dir=os.path.join(os.path.dirname(__file__), "../data/indexes/hindi"))
    
    en_ext = ExtractiveAnswerGenerator(embedding_service=en_retriever.embedding_service)
    hi_ext = ExtractiveAnswerGenerator(embedding_service=hi_retriever.embedding_service)
    
    en_queries, hi_queries = [], []
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            en_queries.append(row.get("Eng_Query"))
            hi_queries.append(row.get("query"))
            
    en_queries = en_queries[:20]
    hi_queries = hi_queries[:20]
    
    def evaluate(generator, retriever, queries):
        gen_list = []
        tot_list = []
        grounded, safe_refusal, unsupported, incomplete = 0, 0, 0, 0
        
        for q in queries:
            t0 = time.time()
            chunks = retriever.retrieve_vector(q, top_k=3)
            
            t1 = time.time()
            resp = generator.generate(q, chunks)
            t2 = time.time()
            
            gen_dur = t2 - t1
            tot_dur = t2 - t0
            
            gen_list.append(gen_dur)
            tot_list.append(tot_dur)
            
            ans = resp.answer
            if "I don't have enough" in ans:
                safe_refusal += 1
            elif len(resp.source_chunk_ids) > 0:
                grounded += 1
            else:
                unsupported += 1
                    
        return {
            "Grounded": grounded,
            "Safe refusal": safe_refusal,
            "Unsupported": unsupported,
            "Incomplete": incomplete,
            "Gen P50": np.percentile(gen_list, 50),
            "Gen P70": np.percentile(gen_list, 70),
            "Gen P100": np.percentile(gen_list, 100),
            "Tot P50": np.percentile(tot_list, 50),
            "Tot P70": np.percentile(tot_list, 70),
            "Tot P100": np.percentile(tot_list, 100)
        }

    print("--- English Extractive ---", flush=True)
    res_en = evaluate(en_ext, en_retriever, en_queries)
    print(res_en, flush=True)
    
    print("--- Hindi Extractive ---", flush=True)
    res_hi = evaluate(hi_ext, hi_retriever, hi_queries)
    print(res_hi, flush=True)

if __name__ == "__main__":
    run_evaluation()
