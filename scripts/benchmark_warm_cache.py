"""
Warm Cache Benchmark
====================
Stage A: Run 30 queries per language to populate cache (not measured).
Stage B: Run SAME 30 queries again on the warm cache (measured).

One persistent generator instance per language throughout.
Timer boundaries identical to the original cold benchmark.
"""
import os, sys, time, json
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../backend/.env'))

from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

N = 30

def load_queries(path, field, n=N):
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            q = rec.get(field, '') or rec.get('query', '')
            if q and len(q.strip()) > 5:
                queries.append(q.strip())
            if len(queries) >= n:
                break
    return queries

configs = [
    ('English',   'data/indexes/english',   'data/processed/hinval_500.jsonl',              'Eng_Query'),
    ('Hindi',     'data/indexes/hindi',     'data/processed/hinval_500.jsonl',              'query'),
    ('Tamil',     'data/indexes/tamil',     'data/processed/tamil_validation_500.jsonl',    'query'),
    ('Telugu',    'data/indexes/telugu',    'data/processed/telugu_validation_500.jsonl',   'query'),
    ('Malayalam', 'data/indexes/malayalam', 'data/processed/malayalam_validation_500.jsonl','query'),
]

# Cold P50/P70/P100 from previous benchmark for comparison
cold_results = {
    'English':   {'P50': 154.4, 'P70': 172.9, 'P100': 366.8},
    'Hindi':     {'P50': 175.8, 'P70': 190.4, 'P100': 814.3},
    'Tamil':     {'P50': 228.7, 'P70': 282.8, 'P100': 1742.9},
    'Telugu':    {'P50': 210.1, 'P70': 250.4, 'P100': 577.7},
    'Malayalam': {'P50': 210.2, 'P70': 240.3, 'P100': 1012.7},
}

warm_results = {}

for lang, idx, path, field in configs:
    queries = load_queries(path, field, N)
    print(f"\n{'='*60}")
    print(f"LANGUAGE: {lang}  (n={len(queries)} queries)")
    print(f"{'='*60}")

    # One persistent retriever and generator — not recreated between stages
    retriever = Retriever(index_dir=idx)
    generator = ExtractiveAnswerGenerator(embedding_service=retriever.embedding_service)

    # ── STAGE A: Warm-up (populate cache, NOT measured) ──
    print("Stage A: Warming cache...")
    stage_a_outputs = []
    for q in queries:
        chunks_faiss = retriever.retrieve_vector(q, top_k=5)
        chunks_bm25  = retriever.retrieve_bm25(q, top_k=5)
        seen = set(); chunks = []
        for c in chunks_faiss + chunks_bm25:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id); chunks.append(c)
        chunks = chunks[:5]
        resp = generator.generate(q, chunks)
        stage_a_outputs.append({
            'answer': resp.answer,
            'source_chunk_ids': resp.source_chunk_ids,
            'is_refusal': "I don't have enough information" in resp.answer,
        })

    cs_after_warmup = generator.cache_stats()
    print(f"Cache after warm-up: hits={cs_after_warmup['hits']} misses={cs_after_warmup['misses']}")

    # ── STAGE B: Measurement on warm cache ──
    print("Stage B: Measuring warm cache performance...")

    # Reset hit/miss counters so Stage B stats are isolated
    generator._cache.hits = 0
    generator._cache.misses = 0

    t_ret_list, t_gen_list, t_total_list = [], [], []
    quality = {'grounded': 0, 'safe_refusal': 0, 'unsupported': 0, 'incomplete': 0}
    stage_b_outputs = []

    for q in queries:
        # ── RAG TIMER START ──
        t0 = time.perf_counter()

        # Retrieval
        t_r0 = time.perf_counter()
        chunks_faiss = retriever.retrieve_vector(q, top_k=5)
        chunks_bm25  = retriever.retrieve_bm25(q, top_k=5)
        seen = set(); chunks = []
        for c in chunks_faiss + chunks_bm25:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id); chunks.append(c)
        chunks = chunks[:5]
        t_r1 = time.perf_counter()
        t_ret_list.append((t_r1 - t_r0) * 1000)

        # Generation (warm cache)
        t_g0 = time.perf_counter()
        resp = generator.generate(q, chunks)
        t_g1 = time.perf_counter()
        t_gen_list.append((t_g1 - t_g0) * 1000)

        # ── RAG TIMER STOP ──
        t_total_list.append((t_g1 - t0) * 1000)

        stage_b_outputs.append({
            'answer': resp.answer,
            'source_chunk_ids': resp.source_chunk_ids,
            'is_refusal': "I don't have enough information" in resp.answer,
        })

        if "I don't have enough information" in resp.answer:
            quality['safe_refusal'] += 1
        elif resp.source_chunk_ids:
            quality['grounded'] += 1
        else:
            quality['unsupported'] += 1

    cs_stage_b = generator.cache_stats()

    def pct(arr, p): return round(np.percentile(arr, p), 1)

    warm = {
        'retrieval': {'P50': pct(t_ret_list,50), 'P70': pct(t_ret_list,70), 'P100': pct(t_ret_list,100)},
        'generation': {'P50': pct(t_gen_list,50), 'P70': pct(t_gen_list,70), 'P100': pct(t_gen_list,100)},
        'total': {'P50': pct(t_total_list,50), 'P70': pct(t_total_list,70), 'P100': pct(t_total_list,100)},
        'quality': quality,
        'cache_stage_b': cs_stage_b,
    }
    warm_results[lang] = warm

    # ── Output equivalence check ──
    answer_ok = source_ok = refusal_ok = True
    for i, (a, b) in enumerate(zip(stage_a_outputs, stage_b_outputs)):
        if a['answer'] != b['answer']:
            print(f"  FAIL answer[{i}]: StageA={a['answer'][:60]} StageB={b['answer'][:60]}")
            answer_ok = False
        if a['source_chunk_ids'] != b['source_chunk_ids']:
            print(f"  FAIL source_ids[{i}]")
            source_ok = False
        if a['is_refusal'] != b['is_refusal']:
            print(f"  FAIL refusal[{i}]")
            refusal_ok = False

    cold = cold_results[lang]
    print(f"\nOUTPUT EQUIVALENCE:")
    print(f"  Answer text:     {'PASS' if answer_ok else 'FAIL'}")
    print(f"  Source IDs:      {'PASS' if source_ok else 'FAIL'}")
    print(f"  Refusal flag:    {'PASS' if refusal_ok else 'FAIL'}")
    print(f"\nLATENCY:")
    print(f"  Retrieval   P50={warm['retrieval']['P50']}ms  P70={warm['retrieval']['P70']}ms  P100={warm['retrieval']['P100']}ms")
    print(f"  Generation  P50={warm['generation']['P50']}ms  P70={warm['generation']['P70']}ms  P100={warm['generation']['P100']}ms")
    print(f"  TOTAL COLD  P50={cold['P50']}ms  P70={cold['P70']}ms  P100={cold['P100']}ms")
    print(f"  TOTAL WARM  P50={warm['total']['P50']}ms  P70={warm['total']['P70']}ms  P100={warm['total']['P100']}ms")
    p50_ok = warm['total']['P50'] < 200
    p70_ok = warm['total']['P70'] < 200
    p100_ok = warm['total']['P100'] < 200
    print(f"  <200ms      P50={'PASS' if p50_ok else 'FAIL'}  P70={'PASS' if p70_ok else 'FAIL'}  P100={'PASS' if p100_ok else 'FAIL'}")
    print(f"  Quality     Grounded={quality['grounded']} SafeRefusal={quality['safe_refusal']} Unsupported={quality['unsupported']} Incomplete={quality['incomplete']}")
    print(f"  Cache StageB hits={cs_stage_b['hits']} misses={cs_stage_b['misses']} hit_rate={cs_stage_b['hit_rate']}")

# Save
with open('data/processed/rag_latency_warm_cache.json', 'w', encoding='utf-8') as f:
    json.dump(warm_results, f, indent=2, ensure_ascii=False)
print("\nResults saved to data/processed/rag_latency_warm_cache.json")
