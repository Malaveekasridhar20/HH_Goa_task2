"""
Profiling script for ExtractiveAnswerGenerator bottleneck.
Read-only investigation. No files modified.
"""
import os, sys, time, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../backend/.env'))

from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator
import re

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

def split_sentences(text):
    sentences = re.split(r'([.?!।]+)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        sent = sentences[i].strip() + sentences[i+1].strip()
        if sent: result.append(sent)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result

def profile_language(lang_name, index_dir, queries):
    retriever = Retriever(index_dir=index_dir)
    generator = ExtractiveAnswerGenerator(embedding_service=retriever.embedding_service)

    stats = {
        'n_chunks': [],
        'n_sentences': [],
        't_retrieval': [],
        't_encode_query': [],
        't_encode_docs': [],
        't_cosine': [],
        't_selection': [],
        't_total_gen': [],
    }

    for q in queries:
        # --- Retrieval ---
        t_ret0 = time.perf_counter()
        chunks = retriever.retrieve_vector(q, top_k=5)
        t_ret1 = time.perf_counter()
        stats['t_retrieval'].append((t_ret1 - t_ret0) * 1000)
        stats['n_chunks'].append(len(chunks))

        # --- Candidate sentences ---
        candidate_sentences = []
        for chunk in chunks:
            for s in split_sentences(chunk.text):
                if len(s.strip()) > 5:
                    candidate_sentences.append(s.strip())
        stats['n_sentences'].append(len(candidate_sentences))

        if not candidate_sentences:
            continue

        # --- Encode query ---
        t_qe0 = time.perf_counter()
        query_emb = retriever.embedding_service.encode_query(q)
        t_qe1 = time.perf_counter()
        stats['t_encode_query'].append((t_qe1 - t_qe0) * 1000)

        # --- Encode all candidate sentences ---
        t_de0 = time.perf_counter()
        sent_embs = retriever.embedding_service.encode_documents(candidate_sentences)
        t_de1 = time.perf_counter()
        stats['t_encode_docs'].append((t_de1 - t_de0) * 1000)

        # --- Cosine similarity ---
        t_cos0 = time.perf_counter()
        scores = np.dot(sent_embs, query_emb)
        t_cos1 = time.perf_counter()
        stats['t_cosine'].append((t_cos1 - t_cos0) * 1000)

        # --- Selection ---
        t_sel0 = time.perf_counter()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        t_sel1 = time.perf_counter()
        stats['t_selection'].append((t_sel1 - t_sel0) * 1000)

        # --- Total generation (what generate() actually does) ---
        t_gen0 = time.perf_counter()
        _ = generator.generate(q, chunks)
        t_gen1 = time.perf_counter()
        stats['t_total_gen'].append((t_gen1 - t_gen0) * 1000)

    def pct(arr, p): return round(np.percentile(arr, p), 2) if arr else 0.0

    print(f"\n{'='*55}")
    print(f"LANGUAGE: {lang_name}  (n={len(queries)})")
    print(f"{'='*55}")
    print(f"Chunks retrieved:     avg={np.mean(stats['n_chunks']):.1f}  min={min(stats['n_chunks'])}  max={max(stats['n_chunks'])}")
    print(f"Candidate sentences:  avg={np.mean(stats['n_sentences']):.1f}  min={min(stats['n_sentences'])}  max={max(stats['n_sentences'])}")
    print(f"")
    for label, key in [
        ("Query embedding (ms)", "t_encode_query"),
        ("Doc encoding (ms)  ", "t_encode_docs"),
        ("Cosine similarity  ", "t_cosine"),
        ("Best selection     ", "t_selection"),
        ("TOTAL generate()   ", "t_total_gen"),
    ]:
        arr = stats[key]
        if arr:
            print(f"  {label}:  P50={pct(arr,50)}  P70={pct(arr,70)}  P100={pct(arr,100)}")

    # Correlation: sentence count vs generation time
    if stats['n_sentences'] and stats['t_total_gen']:
        corr = np.corrcoef(stats['n_sentences'][:len(stats['t_total_gen'])],
                           stats['t_total_gen'])[0,1]
        print(f"  Correlation(n_sentences, gen_time): {corr:.3f}")

    return stats

configs = [
    ('English',   'data/indexes/english',   'data/processed/hinval_500.jsonl',              'Eng_Query'),
    ('Hindi',     'data/indexes/hindi',     'data/processed/hinval_500.jsonl',              'query'),
    ('Tamil',     'data/indexes/tamil',     'data/processed/tamil_validation_500.jsonl',    'query'),
    ('Telugu',    'data/indexes/telugu',    'data/processed/telugu_validation_500.jsonl',   'query'),
    ('Malayalam', 'data/indexes/malayalam', 'data/processed/malayalam_validation_500.jsonl','query'),
]

for lang, idx, path, field in configs:
    queries = load_queries(path, field, N)
    profile_language(lang, idx, queries)
