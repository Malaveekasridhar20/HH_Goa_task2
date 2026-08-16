import os
import sys
import time
import json
import requests
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load .env first
load_dotenv(os.path.join(os.path.dirname(__file__), "../backend/.env"))

# Import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))
from app.retrieval.retriever import Retriever
from app.generation.generator import AnswerGenerator

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def count_tokens(text):
    return len(tokenizer.encode(text))

def run_tests():
    print("Loading Retriever...")
    retriever = Retriever(index_dir=os.path.join(os.path.dirname(__file__), "../data/indexes/english"))
    
    generator = AnswerGenerator()
    api_key = os.getenv("GENERATION_API_KEY")
    base_url = os.getenv("GENERATION_BASE_URL")
    model = os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant")
    
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    
    # --- 1. Token Profile & Latency Breakdown (English) ---
    print("\n--- English Token Profile & Latency Breakdown ---")
    en_queries = []
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            en_queries.append({"query": row.get("Eng_Query")})
    en_queries = en_queries[:20]
        
    en_in_tokens = []
    en_out_tokens = []
    
    ttfb_list = []
    ttft_list = []
    gen_list = []
    ctx_list = []
    parse_list = []
    total_list = []
    
    for i, q in enumerate(en_queries):
        query_text = q["query"]
        
        # Retrieval
        t0 = time.time()
        results = retriever.retrieve_vector(query_text, top_k=5)
        t1 = time.time()
        ctx_list.append(t1 - t0)
        
        # Build context
        context_text = "\n\n".join([r.text for r in results])
        sys_prompt = "You are a helpful assistant. Use ONLY the following context to answer the question.\n\nContext:\n" + context_text
        user_prompt = query_text
        
        total_in = count_tokens(sys_prompt + user_prompt)
        en_in_tokens.append(total_in)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": True,
            "response_format": {"type": "json_object"}
        }
        
        # Modify system prompt to ask for JSON to match production
        payload["messages"][0]["content"] += '\n\nYou MUST format your output as a JSON object with two keys: "answer" and "source_chunk_ids".'
        
        req_start = time.time()
        
        # Use stream=True to measure TTFB/TTFT
        try:
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, stream=True, timeout=10)
            
            ttfb = None
            ttft = None
            full_text = ""
            
            for line in resp.iter_lines():
                if line:
                    if ttfb is None:
                        ttfb = time.time() - req_start
                        ttft = ttfb # First token is approx TTFB for SSE
                    
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                        except:
                            pass
                            
            req_end = time.time()
            gen_time = req_end - req_start
            ttfb_list.append(ttfb if ttfb else gen_time)
            ttft_list.append(ttft if ttft else gen_time)
            gen_list.append(gen_time)
            
            en_out_tokens.append(count_tokens(full_text))
            
            # Parsing time
            parse_start = time.time()
            try:
                json.loads(full_text)
            except:
                pass
            parse_end = time.time()
            parse_list.append(parse_end - parse_start)
            
            total_list.append(parse_end - t0)
            
        except Exception as e:
            print(f"Error on query {i}: {e}")
            
    print(f"EN Input P50: {np.percentile(en_in_tokens, 50):.0f}, P100: {np.percentile(en_in_tokens, 100):.0f}")
    print(f"EN Output P50: {np.percentile(en_out_tokens, 50):.0f}, P100: {np.percentile(en_out_tokens, 100):.0f}")
    print(f"EN Context P50: {np.percentile(ctx_list, 50):.3f}s")
    print(f"EN TTFB P50: {np.percentile(ttfb_list, 50):.3f}s, P100: {np.percentile(ttfb_list, 100):.3f}s")
    print(f"EN Generation P50: {np.percentile(gen_list, 50):.3f}s, P100: {np.percentile(gen_list, 100):.3f}s")
    print(f"EN Parsing P50: {np.percentile(parse_list, 50):.4f}s")
    print(f"EN Total P50: {np.percentile(total_list, 50):.3f}s")

if __name__ == "__main__":
    run_tests()
