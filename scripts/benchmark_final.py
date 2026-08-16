import os
import sys
import time
import json
import asyncio
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app.main import startup_event, pipeline
from app.orchestration.models import VoiceRAGRequest
from app.stt.models import SpeechToTextResponse

# Use deterministic benchmark queries
QUERIES = {
    "en": "what is a corporation?",
    "hi": "एक निगम क्या है?",
    "ta": "கழுகு எவ்வளவு வேகமாக பயணிக்கிறது?",
    "te": "కార్పొరేషన్ అంటే ఏమిటి?",
    "ml": "ഒരു കോർപ്പറേഷൻ എന്നാൽ എന്ത്?"
}

class MockSTT:
    def __init__(self, transcript, lang):
        self.t = transcript
        self.l = lang
    def transcribe(self, req):
        return SpeechToTextResponse(
            transcript=self.t, detected_language=self.l, success=True, latency=0.01, provider="mock"
        )

def pct(arr, p):
    return round(float(np.percentile(arr, p)), 2) if arr else 0.0

async def run_benchmark():
    print("==================================================")
    print("FINAL LATENCY BENCHMARK")
    print("==================================================")
    
    # 1. Simulate server startup
    print("Simulating server startup (loading precomputed embeddings)...")
    await startup_event()
    
    results = {}
    
    for lang, query_text in QUERIES.items():
        print(f"\n==================================================")
        print(f"LANGUAGE: {lang.upper()}")
        print(f"==================================================")
        
        pipeline.stt_service = MockSTT(query_text, lang)
        req = VoiceRAGRequest(audio_data=b"dummy")
        
        # 2. Cold-start Request
        print("Executing 1 Cold Request...")
        cold_res = pipeline.execute(req)
        cold_metrics = {
            'total_rag': round(cold_res.total_rag_latency_ms, 2),
            'embedding': round(cold_res.embedding_latency_ms, 2),
            'faiss': round(cold_res.faiss_latency_ms, 2),
            'bm25': round(cold_res.bm25_latency_ms, 2),
            'fusion': round(cold_res.fusion_latency_ms, 2),
            'generation': round(cold_res.generation_latency_ms, 2),
            'grounding': round(cold_res.grounding_latency_ms, 2),
            'guardrails': round(cold_res.guardrails_latency_ms, 2)
        }
        print(f"Cold-start RAG: {cold_metrics['total_rag']} ms")
        print("Cold-start Breakdown:", cold_metrics)
        
        # 3. Warm-ups
        print("Executing 5 Warm-up Requests...")
        for _ in range(5):
            pipeline.execute(req)
            
        # 4. Measured Runs
        print("Executing 30 Measured Requests...")
        metrics = {
            'total_rag': [], 'embedding': [], 'faiss': [], 'bm25': [],
            'fusion': [], 'generation': [], 'grounding': [], 'guardrails': []
        }
        for _ in range(30):
            res = pipeline.execute(req)
            metrics['total_rag'].append(res.total_rag_latency_ms)
            metrics['embedding'].append(res.embedding_latency_ms)
            metrics['faiss'].append(res.faiss_latency_ms)
            metrics['bm25'].append(res.bm25_latency_ms)
            metrics['fusion'].append(res.fusion_latency_ms)
            metrics['generation'].append(res.generation_latency_ms)
            metrics['grounding'].append(res.grounding_latency_ms)
            metrics['guardrails'].append(res.guardrails_latency_ms)
            
        print("\n--- WARM CACHE METRICS (30 RUNS) ---")
        for k, arr in metrics.items():
            print(f"{k.ljust(15)} P50: {pct(arr, 50):>6.2f} ms | P70: {pct(arr, 70):>6.2f} ms | P100: {pct(arr, 100):>6.2f} ms")
            
        results[lang] = {
            "query": query_text,
            "cold": cold_metrics,
            "warm": {
                k: {"p50": pct(arr, 50), "p70": pct(arr, 70), "p100": pct(arr, 100)}
                for k, arr in metrics.items()
            }
        }
        
    # Write JSON
    out_dir = os.path.join(os.path.dirname(__file__), "../data/processed")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "final_latency_benchmark.json")
    
    final_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "description": "Precomputed sentence embedding optimization baseline",
        "environment": "Windows / CUDA",
        "queries_per_language": 30,
        "languages": results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)
        
    # Write Markdown
    md_path = os.path.join(out_dir, "final_latency_benchmark.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Final Latency Benchmark Report\n\n")
        f.write(f"**Timestamp:** {final_data['timestamp']}\n")
        f.write(f"**Description:** {final_data['description']}\n\n")
        
        for lang, d in results.items():
            f.write(f"## {lang.upper()}\n")
            f.write(f"- **Cold RAG:** {d['cold']['total_rag']} ms\n")
            f.write(f"- **Warm P100 RAG:** {d['warm']['total_rag']['p100']} ms\n\n")
            f.write("| Stage | Cold (ms) | Warm P50 (ms) | Warm P100 (ms) |\n")
            f.write("|---|---|---|---|\n")
            for k in d['cold'].keys():
                f.write(f"| {k} | {d['cold'][k]} | {d['warm'][k]['p50']} | {d['warm'][k]['p100']} |\n")
            f.write("\n")
            
    print(f"\nSaved reports to {json_path} and {md_path}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
