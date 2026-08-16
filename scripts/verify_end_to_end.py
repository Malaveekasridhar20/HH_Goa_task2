import os
import requests
import time

base_url = "http://127.0.0.1:8000"

def test_query(file_path, lang):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            res = requests.post(f"{base_url}/api/voice/query", files={"audio": f}, data={"language": lang})
        return res.json()
    except Exception as e:
        print(f"Error testing {file_path}: {e}")
        return None

en_files = [f"data/human_audio/en_{i}.wav" for i in range(5)]
hi_files = [f"data/human_audio/hi_{i}.wav" for i in range(5)]

print("--- ENGLISH QUERIES ---")
for f in en_files:
    resp = test_query(f, "en")
    if resp:
        print(f"File: {f}")
        print(f"Transcript: {resp.get('transcript')}")
        print(f"Answer: {resp.get('answer')}")
        print(f"Grounded: {'PASS' if resp.get('source_chunk_ids') else 'FAIL'}")
        print(f"STT latency: {resp.get('stt_latency_ms', 0):.2f}ms")
        print(f"Retrieval latency: {resp.get('retrieval_latency_ms', 0):.2f}ms")
        print(f"Generation latency: {resp.get('generation_latency_ms', 0):.2f}ms")
        print(f"Total latency: {resp.get('total_latency_ms', 0):.2f}ms")
        print("-" * 30)

print("--- HINDI QUERIES ---")
for f in hi_files:
    resp = test_query(f, "hi")
    if resp:
        print(f"File: {f}")
        try:
            # Safely print transcript and answer to avoid UnicodeEncodeError in Windows terminal
            print(f"Transcript: {resp.get('transcript').encode('unicode_escape').decode('utf-8') if resp.get('transcript') else ''}")
            print(f"Answer: {resp.get('answer').encode('unicode_escape').decode('utf-8') if resp.get('answer') else ''}")
        except:
            print("Transcript: [Hindi text]")
            print("Answer: [Hindi text]")
        print(f"Grounded: {'PASS' if resp.get('source_chunk_ids') else 'FAIL'}")
        print(f"STT latency: {resp.get('stt_latency_ms', 0):.2f}ms")
        print(f"Retrieval latency: {resp.get('retrieval_latency_ms', 0):.2f}ms")
        print(f"Generation latency: {resp.get('generation_latency_ms', 0):.2f}ms")
        print(f"Total latency: {resp.get('total_latency_ms', 0):.2f}ms")
        print("-" * 30)
