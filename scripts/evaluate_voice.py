import os
import sys
import json
import time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../backend/.env'))

from app.orchestration.pipeline import VoiceRAGPipeline
from app.orchestration.models import VoiceRAGRequest

def main():
    manifest_path = 'data/human_audio/manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Filter to only Tamil, Telugu, Malayalam
    manifest = [m for m in manifest if m['language'] in ['tamil', 'telugu', 'malayalam']]

    print("Initializing Voice RAG Pipeline...")
    pipeline = VoiceRAGPipeline()

    results = {
        'tamil': {'stt_latencies': [], 'rag_latencies': [], 'total_latencies': [], 'success': 0, 'fail': 0, 'grounded': 0, 'refusal': 0, 'unsupported': 0, 'incomplete': 0},
        'telugu': {'stt_latencies': [], 'rag_latencies': [], 'total_latencies': [], 'success': 0, 'fail': 0, 'grounded': 0, 'refusal': 0, 'unsupported': 0, 'incomplete': 0},
        'malayalam': {'stt_latencies': [], 'rag_latencies': [], 'total_latencies': [], 'success': 0, 'fail': 0, 'grounded': 0, 'refusal': 0, 'unsupported': 0, 'incomplete': 0}
    }

    lang_map = {'tamil': 'ta-IN', 'telugu': 'te-IN', 'malayalam': 'ml-IN'}

    for entry in manifest:
        lang = entry['language']
        filename = entry['filename']
        filepath = os.path.join('data/human_audio', filename)
        orig_transcript = entry['original transcript']
        
        print(f"\n--- Testing {filename} ({lang}) ---")
        
        with open(filepath, 'rb') as af:
            audio_data = af.read()
            
        req = VoiceRAGRequest(
            audio_data=audio_data,
            language_hint=lang_map[lang],
            top_k=5,
            generation_mode="extractive"
        )
        
        resp = pipeline.execute(req)
        
        print(f"Original Transcript: {orig_transcript}")
        print(f"Sarvam STT: {resp.transcript}")
        print(f"Answer: {resp.answer}")
        
        if resp.success:
            results[lang]['success'] += 1
            # Simple heuristic for metrics
            # A real check would use Guardrails.is_grounded, but VoiceRAGResponse might not have that flag
            if "I don't have enough information" in resp.answer or "I couldn't hear" in resp.answer:
                results[lang]['refusal'] += 1
                print("Status: Safe Refusal")
            else:
                results[lang]['grounded'] += 1
                print("Status: Grounded Answer")
        else:
            results[lang]['fail'] += 1
            print(f"Status: Failed ({resp.error})")
            
        if resp.stt_latency_ms is not None:
            results[lang]['stt_latencies'].append(resp.stt_latency_ms / 1000.0)
        
        rag_latency = 0
        if resp.retrieval_latency_ms: rag_latency += resp.retrieval_latency_ms
        if resp.generation_latency_ms: rag_latency += resp.generation_latency_ms
        if rag_latency > 0:
            results[lang]['rag_latencies'].append(rag_latency / 1000.0)
            
        if resp.total_latency_ms is not None:
            results[lang]['total_latencies'].append(resp.total_latency_ms / 1000.0)

    # Output metrics
    for lang, metrics in results.items():
        if len(metrics['stt_latencies']) == 0:
            continue
            
        def p_val(arr, p):
            return np.percentile(arr, p) if arr else 0.0
            
        print(f"\n{lang.upper()} METRICS:")
        print(f"Samples: {len(metrics['stt_latencies'])}")
        print(f"Successful: {metrics['success']}")
        print(f"Failed: {metrics['fail']}")
        print(f"Grounded: {metrics['grounded']}")
        print(f"Safe refusal: {metrics['refusal']}")
        print(f"Unsupported: {metrics['unsupported']}")
        
        print(f"STT P50/P70/P100: {p_val(metrics['stt_latencies'], 50):.3f}s / {p_val(metrics['stt_latencies'], 70):.3f}s / {p_val(metrics['stt_latencies'], 100):.3f}s")
        print(f"RAG P50/P70/P100: {p_val(metrics['rag_latencies'], 50):.3f}s / {p_val(metrics['rag_latencies'], 70):.3f}s / {p_val(metrics['rag_latencies'], 100):.3f}s")
        print(f"Total P50/P70/P100: {p_val(metrics['total_latencies'], 50):.3f}s / {p_val(metrics['total_latencies'], 70):.3f}s / {p_val(metrics['total_latencies'], 100):.3f}s")

if __name__ == "__main__":
    main()
