import json
import os
import sys
import time
import requests
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def count_tokens(text):
    return len(tokenizer.encode(text))

def analyze_lengths(language):
    with open(f"data/test_{language}.json", "r", encoding="utf-8") as f:
        queries = json.load(f)[:20]
        
    audit_file = f"data/indexes/english/generation_audit.json" if language == "en" else f"data/indexes/hindi/generation_audit.json"
    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)
        
    input_tokens = []
    output_tokens = []
    
    for i, q in enumerate(queries):
        query_text = q["query"]
        ans = audit[i].get("answer", "")
        # rough context estimation
        ctx = "This is a placeholder context. " * 50
        system_prompt = "You are a helpful AI."
        total_in = count_tokens(system_prompt + query_text + ctx)
        out_t = count_tokens(ans)
        
        input_tokens.append(total_in)
        output_tokens.append(out_t)
        
    print(f"--- {language} ---")
    print(f"Input P50: {np.percentile(input_tokens, 50)}")
    print(f"Input P100: {np.percentile(input_tokens, 100)}")
    print(f"Output P50: {np.percentile(output_tokens, 50)}")
    print(f"Output P100: {np.percentile(output_tokens, 100)}")

if __name__ == "__main__":
    analyze_lengths("en")
    analyze_lengths("hi")
