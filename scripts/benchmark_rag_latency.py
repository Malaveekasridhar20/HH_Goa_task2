"""
RAG Latency Benchmark
=====================
Measures ONLY the transcript-to-answer pipeline (Task RAG Latency).
STT latency is NOT included — timer starts after transcript is available.

Components timed:
  1. Input validation / guardrail (empty check)
  2. Query embedding
  3. FAISS retrieval
  4. BM25 retrieval (via hybrid)
  5. Fusion / ranking
  6. Extractive answer generation (embedding + similarity scoring)
  7. Grounding threshold check (safe refusal logic)
  8. Response serialization
"""

import os
import sys
import time
import json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../backend/.env'))

from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

N_QUERIES = 30  # per language — enough for P50/P70/P100

def load_queries(path, query_field, n=N_QUERIES):
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get(query_field, '') or rec.get('query', '')
            if q and len(q.strip()) > 5:
                queries.append(q.strip())
            if len(queries) >= n:
                break
    return queries

def benchmark_language(lang_name, index_dir, queries):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {lang_name} ({len(queries)} queries)")
    print(f"Index: {index_dir}")
    print(f"{'='*50}")

    retriever = Retriever(index_dir=index_dir)
    generator = ExtractiveAnswerGenerator(embedding_service=retriever.embedding_service)

    # Per-component timings
    t_validation = []
    t_retrieval = []
    t_generation = []
    t_guardrail = []
    t_serialization = []
    t_total = []

    quality = {'grounded': 0, 'safe_refusal': 0, 'unsupported': 0, 'incomplete': 0}

    for q in queries:
        # ── RAG TIMER STARTS HERE ──
        # (Transcript is already available; STT is NOT included)
        t0 = time.perf_counter()

        # 1. Input validation / guardrail (empty transcript check)
        t_val_0 = time.perf_counter()
        if not q or len(q.strip()) < 2:
            t_val_1 = time.perf_counter()
            t_validation.append(t_val_1 - t_val_0)
            quality['incomplete'] += 1
            t_total.append(t_val_1 - t0)
            continue
        t_val_1 = time.perf_counter()
        t_validation.append(t_val_1 - t_val_0)

        # 2. Retrieval: FAISS + BM25 + fusion (timing all together as pipeline.py does)
        t_ret_0 = time.perf_counter()
        # FAISS vector retrieval (includes embedding)
        chunks_faiss = retriever.retrieve_vector(q, top_k=5)
        # BM25 lexical retrieval
        chunks_bm25 = retriever.retrieve_bm25(q, top_k=5)
        # Simple fusion: deduplicate, prefer FAISS
        seen = set()
        chunks = []
        for c in chunks_faiss + chunks_bm25:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                chunks.append(c)
        chunks = chunks[:5]
        t_ret_1 = time.perf_counter()
        t_retrieval.append(t_ret_1 - t_ret_0)

        # 3. Extractive generation (embedding + similarity scoring)
        t_gen_0 = time.perf_counter()
        resp = generator.generate(q, chunks)
        t_gen_1 = time.perf_counter()
        t_generation.append(t_gen_1 - t_gen_0)

        # 4. Grounding threshold check (already done inside generator, measure overhead)
        t_guard_0 = time.perf_counter()
        refusal = "I don't have enough information" in resp.answer
        t_guard_1 = time.perf_counter()
        t_guardrail.append(t_guard_1 - t_guard_0)

        # 5. Serialization (dict construction)
        t_ser_0 = time.perf_counter()
        result = {
            'answer': resp.answer,
            'source_chunk_ids': resp.source_chunk_ids,
            'model': resp.model,
        }
        t_ser_1 = time.perf_counter()
        t_serialization.append(t_ser_1 - t_ser_0)

        # ── RAG TIMER STOPS HERE ──
        t_end = time.perf_counter()
        t_total.append(t_end - t0)

        # Quality classification
        if refusal:
            quality['safe_refusal'] += 1
        elif resp.source_chunk_ids:
            quality['grounded'] += 1
        else:
            quality['unsupported'] += 1

    def pct(arr, p):
        return np.percentile(arr, p) * 1000 if arr else 0.0  # ms

    n = len(t_total)
    return {
        'language': lang_name,
        'n_queries': n,
        'retrieval_ms': {
            'P50': pct(t_retrieval, 50),
            'P70': pct(t_retrieval, 70),
            'P100': pct(t_retrieval, 100),
        },
        'generation_ms': {
            'P50': pct(t_generation, 50),
            'P70': pct(t_generation, 70),
            'P100': pct(t_generation, 100),
        },
        'guardrail_ms': {
            'P50': pct(t_guardrail, 50),
            'P70': pct(t_guardrail, 70),
            'P100': pct(t_guardrail, 100),
        },
        'serialization_ms': {
            'P50': pct(t_serialization, 50),
            'P70': pct(t_serialization, 70),
            'P100': pct(t_serialization, 100),
        },
        'total_rag_ms': {
            'P50': pct(t_total, 50),
            'P70': pct(t_total, 70),
            'P100': pct(t_total, 100),
        },
        'quality': quality,
    }

def main():
    # ── Data sources ──
    configs = [
        {
            'lang': 'English',
            'index_dir': 'data/indexes/english',
            'data_path': 'data/processed/english_chunks.jsonl',
            'query_field': 'text',  # English chunks have text, we extract queries from hybrid eval
            'use_hindi_eng': False,
            'use_english_hinval': True,  # use Eng_Query from hinval
        },
        {
            'lang': 'Hindi',
            'index_dir': 'data/indexes/hindi',
            'data_path': 'data/processed/hinval_500.jsonl',
            'query_field': 'query',
        },
        {
            'lang': 'Tamil',
            'index_dir': 'data/indexes/tamil',
            'data_path': 'data/processed/tamil_validation_500.jsonl',
            'query_field': 'query',
        },
        {
            'lang': 'Telugu',
            'index_dir': 'data/indexes/telugu',
            'data_path': 'data/processed/telugu_validation_500.jsonl',
            'query_field': 'query',
        },
        {
            'lang': 'Malayalam',
            'index_dir': 'data/indexes/malayalam',
            'data_path': 'data/processed/malayalam_validation_500.jsonl',
            'query_field': 'query',
        },
    ]

    all_results = []

    for cfg in configs:
        lang = cfg['lang']
        # Special English handling: use Eng_Query from hinval
        if cfg.get('use_english_hinval'):
            queries = load_queries('data/processed/hinval_500.jsonl', 'Eng_Query', N_QUERIES)
        else:
            queries = load_queries(cfg['data_path'], cfg['query_field'], N_QUERIES)

        if not queries:
            print(f"WARNING: No queries loaded for {lang}")
            continue

        result = benchmark_language(lang, cfg['index_dir'], queries)
        all_results.append(result)

        # Print per-language summary
        print(f"\n{lang} RESULTS (n={result['n_queries']}):")
        for comp in ['retrieval_ms', 'generation_ms', 'guardrail_ms', 'serialization_ms', 'total_rag_ms']:
            d = result[comp]
            print(f"  {comp}: P50={d['P50']:.1f}ms  P70={d['P70']:.1f}ms  P100={d['P100']:.1f}ms")
        q = result['quality']
        print(f"  Quality: Grounded={q['grounded']}  SafeRefusal={q['safe_refusal']}  Unsupported={q['unsupported']}  Incomplete={q['incomplete']}")
        print(f"  <200ms: {'PASS' if result['total_rag_ms']['P100'] < 200 else 'FAIL'}")

    # Save results
    out_path = 'data/processed/rag_latency_benchmark.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
