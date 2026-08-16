import os
import sys
import time
import json
import requests
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))
from app.retrieval.retriever import Retriever
from app.generation.generator import AnswerGenerator

def evaluate_params(en_queries, retriever, api_key, base_url, top_k, max_tokens, json_mode):
    gen_list = []
    grounded_count = 0
    total = len(en_queries)
    model = os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant")
    
    for i, q in enumerate(en_queries):
        query_text = q["query"]
        
        results = retriever.retrieve_vector(query_text, top_k=top_k)
        
        context_text = "\n\n".join([r.text for r in results])
        sys_prompt = "You are a helpful assistant. Use ONLY the following context to answer the question.\n\nContext:\n" + context_text
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query_text}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            payload["messages"][0]["content"] += '\n\nYou MUST format your output as a JSON object with two keys: "answer" and "source_chunk_ids".'
            
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        req_start = time.time()
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=10)
        req_end = time.time()
        
        gen_list.append(req_end - req_start)
        
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            if json_mode:
                parsed = json.loads(content)
                ans = parsed.get("answer", "")
                if len(parsed.get("source_chunk_ids", [])) > 0:
                    grounded_count += 1
            else:
                ans = content
                if len(ans) > 10 and not ans.startswith("I don't have enough"):
                    grounded_count += 1
        except:
            pass
            
    print(f"TopK={top_k}, MaxT={max_tokens}, JSON={json_mode} | Gen P50: {np.percentile(gen_list, 50):.3f}s | Grounded: {grounded_count}/{total}")

def run_sweeps():
    retriever = Retriever(index_dir=os.path.join(os.path.dirname(__file__), "../data/indexes/english"))
    
    api_key = os.getenv("GENERATION_API_KEY")
    base_url = os.getenv("GENERATION_BASE_URL")
    
    en_queries = []
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            en_queries.append({"query": row.get("Eng_Query")})
    en_queries = en_queries[:10] # Subset for fast sweep
    
    print("\n--- TOP_K Sweep ---")
    for k in [1, 2, 3, 5]:
        evaluate_params(en_queries, retriever, api_key, base_url, top_k=k, max_tokens=128, json_mode=True)
        
    print("\n--- MAX_TOKENS Sweep ---")
    for t in [32, 64, 128, 256]:
        evaluate_params(en_queries, retriever, api_key, base_url, top_k=5, max_tokens=t, json_mode=True)
        
    print("\n--- JSON vs Plain Text ---")
    evaluate_params(en_queries, retriever, api_key, base_url, top_k=5, max_tokens=128, json_mode=True)
    evaluate_params(en_queries, retriever, api_key, base_url, top_k=5, max_tokens=128, json_mode=False)

if __name__ == "__main__":
    run_sweeps()
