"""
Post-optimization regression check + benchmark.
Verifies output equivalence vs baseline, then runs 30-query benchmark
with cache statistics.
"""
import os, sys, json, re, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../backend/.env'))

from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

N_REG = 10   # regression check
N_BENCH = 30 # benchmark

def load_queries(path, field, n):
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
    ('english',   'data/indexes/english',   'data/processed/hinval_500.jsonl',              'Eng_Query'),
    ('hindi',     'data/indexes/hindi',     'data/processed/hinval_500.jsonl',              'query'),
    ('tamil',     'data/indexes/tamil',     'data/processed/tamil_validation_500.jsonl',    'query'),
    ('telugu',    'data/indexes/telugu',    'data/processed/telugu_validation_500.jsonl',   'query'),
    ('malayalam', 'data/indexes/malayalam', 'data/processed/malayalam_validation_500.jsonl','query'),
]

# ── Load baseline ──
with open('data/processed/baseline_outputs.json', 'r', encoding='utf-8') as f:
    baseline = json.load(f)

print("=" * 60)
print("REGRESSION CHECK (n=10 per language)")
print("=" * 60)

regression_ok = True
for lang, idx, path, field in configs:
    queries = load_queries(path, field, N_REG)
    retriever = Retriever(index_dir=idx)
    generator = ExtractiveAnswerGenerator(embedding_service=retriever.embedding_service)
    baseline_entries = baseline[lang]

    answer_match = score_match = refusal_match = source_match = True
    for i, q in enumerate(queries):
        chunks = retriever.retrieve_vector(q, top_k=5)
        resp = generator.generate(q, chunks)
        b = baseline_entries[i]

        # Answer text
        if resp.answer != b['answer']:
            print(f"  FAIL answer [{lang}][{i}]: got={resp.answer[:60]} expected={b['answer'][:60]}")
            answer_match = False
        # Source IDs
        if resp.source_chunk_ids != b['source_chunk_ids']:
            print(f"  FAIL source_ids [{lang}][{i}]")
            source_match = False
        # Refusal flag
        is_refusal = "I don't have enough information" in resp.answer
        if is_refusal != b['is_refusal']:
            print(f"  FAIL refusal [{lang}][{i}]")
            refusal_match = False

    stats = generator.cache_stats()
    print(f"{lang}: answer={'PASS' if answer_match else 'FAIL'}  "
          f"source={'PASS' if source_match else 'FAIL'}  "
          f"refusal={'PASS' if refusal_match else 'FAIL'}  "
          f"cache_hits={stats['hits']} misses={stats['misses']}")
    if not (answer_match and source_match and refusal_match):
        regression_ok = False

if not regression_ok:
    print("\nREGRESSION DETECTED — stopping.")
    sys.exit(1)

print("\nAll regression checks PASS. Proceeding to benchmark.")

# ── Full 30-query benchmark ──
print("\n" + "=" * 60)
print("AFTER-OPTIMIZATION BENCHMARK (n=30 per language)")
print("=" * 60)

bench_results = {}
total_cache_hits = 0
total_cache_misses = 0

for lang, idx, path, field in configs:
    queries = load_queries(path, field, N_BENCH)
    retriever = Retriever(index_dir=idx)
    generator = ExtractiveAnswerGenerator(embedding_service=retriever.embedding_service)

    t_ret_list, t_gen_list, t_total_list = [], [], []
    quality = {'grounded': 0, 'safe_refusal': 0, 'unsupported': 0, 'incomplete': 0}

    for q in queries:
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

        # Generation (with cache)
        t_g0 = time.perf_counter()
        resp = generator.generate(q, chunks)
        t_g1 = time.perf_counter()
        t_gen_list.append((t_g1 - t_g0) * 1000)

        t_total_list.append((t_g1 - t0) * 1000)

        if "I don't have enough information" in resp.answer:
            quality['safe_refusal'] += 1
        elif resp.source_chunk_ids:
            quality['grounded'] += 1
        else:
            quality['unsupported'] += 1

    cs = generator.cache_stats()
    total_cache_hits += cs['hits']
    total_cache_misses += cs['misses']

    def pct(arr, p): return round(np.percentile(arr, p), 1)

    bench_results[lang] = {
        'retrieval': {'P50': pct(t_ret_list,50), 'P70': pct(t_ret_list,70), 'P100': pct(t_ret_list,100)},
        'generation': {'P50': pct(t_gen_list,50), 'P70': pct(t_gen_list,70), 'P100': pct(t_gen_list,100)},
        'total': {'P50': pct(t_total_list,50), 'P70': pct(t_total_list,70), 'P100': pct(t_total_list,100)},
        'quality': quality,
        'cache': cs,
    }

    r = bench_results[lang]
    print(f"\n{lang.upper()} (n={len(queries)}):")
    print(f"  Retrieval  P50={r['retrieval']['P50']}ms  P70={r['retrieval']['P70']}ms  P100={r['retrieval']['P100']}ms")
    print(f"  Generation P50={r['generation']['P50']}ms  P70={r['generation']['P70']}ms  P100={r['generation']['P100']}ms")
    print(f"  TOTAL RAG  P50={r['total']['P50']}ms  P70={r['total']['P70']}ms  P100={r['total']['P100']}ms")
    p50_ok = r['total']['P50'] < 200
    p70_ok = r['total']['P70'] < 200
    p100_ok = r['total']['P100'] < 200
    print(f"  <200ms     P50={'PASS' if p50_ok else 'FAIL'}  P70={'PASS' if p70_ok else 'FAIL'}  P100={'PASS' if p100_ok else 'FAIL'}")
    q = r['quality']
    print(f"  Quality    Grounded={q['grounded']} SafeRefusal={q['safe_refusal']} Unsupported={q['unsupported']} Incomplete={q['incomplete']}")
    print(f"  Cache      hits={cs['hits']} misses={cs['misses']} hit_rate={cs['hit_rate']}")

# Save results
out = {'benchmark': bench_results, 'total_hits': total_cache_hits, 'total_misses': total_cache_misses}
with open('data/processed/rag_latency_benchmark_cached.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

total = total_cache_hits + total_cache_misses
rate = round(total_cache_hits / total, 4) if total else 0
print(f"\nOVERALL CACHE: hits={total_cache_hits} misses={total_cache_misses} rate={rate}")
print("Results saved to data/processed/rag_latency_benchmark_cached.json")
