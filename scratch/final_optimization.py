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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def count_tokens(text):
    return len(tokenizer.encode(text))

def run_tests():
    api_key = os.getenv("GENERATION_API_KEY")
    base_url = os.getenv("GENERATION_BASE_URL")
    
    en_retriever = Retriever(index_dir=os.path.join(os.path.dirname(__file__), "../data/indexes/english"))
    hi_retriever = Retriever(index_dir=os.path.join(os.path.dirname(__file__), "../data/indexes/hindi"))
    
    en_queries, hi_queries = [], []
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            en_queries.append(row.get("Eng_Query"))
            hi_queries.append(row.get("query"))
            
    # Subset to 5 queries for speed and avoiding rate limits
    en_queries = en_queries[:5]
    hi_queries = hi_queries[:5]
    
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    
    # Pre-warm connection
    session.get(f"{base_url}/models", timeout=5)
    
    def evaluate(model, retriever, queries, top_k, max_tokens, json_mode, lang_name):
        ttfb_list, ttft_list, gen_list, total_list, ret_list = [], [], [], [], []
        in_tokens, out_tokens = [], []
        grounded, unsupported, failed = 0, 0, 0
        status_429 = 0
        
        region = "Unknown"
        http_setup_list = []
        
        for i, q in enumerate(queries):
            t0 = time.time()
            results = retriever.retrieve_vector(q, top_k=top_k)
            t1 = time.time()
            ret_dur = t1 - t0
            ret_list.append(ret_dur)
            
            context_text = "\n\n".join([f"[{r.chunk_id}]: {r.text}" for r in results])
            
            if json_mode:
                sys_prompt = "Answer purely using context. If unknown, say 'I don't have enough information'. JSON required: {'answer': '...', 'source_chunk_ids': ['id1']}\nContext:\n" + context_text
                payload_msg = [{"role": "user", "content": sys_prompt + "\nQ: " + q}]
                fmt = {"type": "json_object"}
            else:
                sys_prompt = "Answer using context only. If unknown say 'I don't have enough information'. Cite source_chunk_ids.\nContext:\n" + context_text
                payload_msg = [{"role": "user", "content": sys_prompt + "\nQ: " + q}]
                fmt = None
                
            in_t = count_tokens(payload_msg[0]["content"])
            in_tokens.append(in_t)
            
            payload = {
                "model": model,
                "messages": payload_msg,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True
            }
            if fmt: payload["response_format"] = fmt
                
            try:
                req_start = time.time()
                resp = session.post(f"{base_url}/chat/completions", json=payload, stream=True, timeout=10)
                
                http_setup_list.append(resp.elapsed.total_seconds())
                
                if resp.status_code == 429:
                    status_429 += 1
                    failed += 1
                    time.sleep(2)
                    continue
                elif resp.status_code != 200:
                    failed += 1
                    continue
                    
                ttfb, ttft = None, None
                full_text = ""
                
                if 'cf-ray' in resp.headers:
                    region = resp.headers['cf-ray'].split('-')[-1]
                if 'x-groq-region' in resp.headers:
                    region = resp.headers['x-groq-region']
                
                for line in resp.iter_lines():
                    if line:
                        if ttfb is None:
                            ttfb = time.time() - req_start
                            ttft = ttfb
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]": break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                full_text += delta
                            except: pass
                
                req_end = time.time()
                gen_dur = req_end - req_start
                tot_dur = req_end - t0
                
                out_tokens.append(count_tokens(full_text))
                ttfb_list.append(ttfb)
                ttft_list.append(ttft)
                gen_list.append(gen_dur)
                total_list.append(tot_dur)
                
                if json_mode:
                    try:
                        parsed = json.loads(full_text)
                        ans = parsed.get("answer", "")
                        if "I don't have enough" in ans: unsupported += 1
                        elif parsed.get("source_chunk_ids"): grounded += 1
                    except: pass
                else:
                    if "I don't have enough" in full_text: unsupported += 1
                    elif "source_chunk_ids" in full_text: grounded += 1
                        
            except Exception as e:
                failed += 1
            
            time.sleep(1) # avoid 429
            
        print(f"\n[{model}] | {lang_name} | TopK={top_k} | MaxT={max_tokens} | JSON={json_mode}", flush=True)
        if len(total_list) > 0:
            print(f"Total P50: {np.percentile(total_list, 50):.3f}s, P70: {np.percentile(total_list, 70):.3f}s, P100: {np.percentile(total_list, 100):.3f}s", flush=True)
            print(f"Gen P50: {np.percentile(gen_list, 50):.3f}s, Ret P50: {np.percentile(ret_list, 50):.3f}s", flush=True)
            print(f"TTFB P50: {np.percentile(ttfb_list, 50):.3f}s, TTFT P50: {np.percentile(ttft_list, 50):.3f}s", flush=True)
            print(f"HTTP Setup P50: {np.percentile(http_setup_list, 50):.3f}s", flush=True)
            print(f"Grounded: {grounded}, Unsupported: {unsupported}, Failed: {failed}, 429: {status_429}", flush=True)
            print(f"InTokens P50: {np.percentile(in_tokens, 50):.0f}, OutTokens P50: {np.percentile(out_tokens, 50):.0f}", flush=True)
            print(f"Region: {region}", flush=True)
        else:
            print("All failed.", flush=True)

    print("\n=== Baseline ===", flush=True)
    evaluate("llama-3.1-8b-instant", en_retriever, en_queries, 5, 256, True, "EN")
    evaluate("llama-3.1-8b-instant", hi_retriever, hi_queries, 5, 256, True, "HI")
    
    print("\n=== Model Alternatives ===", flush=True)
    evaluate("openai/gpt-oss-20b", en_retriever, en_queries, 5, 256, True, "EN")
    evaluate("openai/gpt-oss-20b", hi_retriever, hi_queries, 5, 256, True, "HI")
    
    print("\n=== TopK Sweep ===", flush=True)
    evaluate("llama-3.1-8b-instant", en_retriever, en_queries, 1, 128, True, "EN")
    evaluate("llama-3.1-8b-instant", en_retriever, en_queries, 2, 128, True, "EN")
    evaluate("llama-3.1-8b-instant", en_retriever, en_queries, 3, 128, True, "EN")
    
    print("\n=== Plain Text Fallback ===", flush=True)
    evaluate("llama-3.1-8b-instant", en_retriever, en_queries, 2, 128, False, "EN")

if __name__ == "__main__":
    run_tests()
