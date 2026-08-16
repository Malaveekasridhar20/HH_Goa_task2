"""
Capture baseline outputs BEFORE optimization for regression comparison.
Saves answers, scores, source_ids, refusal flags for 10 queries per language.
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../backend/.env'))

from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

N = 10

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
    ('english',   'data/indexes/english',   'data/processed/hinval_500.jsonl',              'Eng_Query'),
    ('hindi',     'data/indexes/hindi',     'data/processed/hinval_500.jsonl',              'query'),
    ('tamil',     'data/indexes/tamil',     'data/processed/tamil_validation_500.jsonl',    'query'),
    ('telugu',    'data/indexes/telugu',    'data/processed/telugu_validation_500.jsonl',   'query'),
    ('malayalam', 'data/indexes/malayalam', 'data/processed/malayalam_validation_500.jsonl','query'),
]

baseline = {}
for lang, idx, path, field in configs:
    queries = load_queries(path, field, N)
    retriever = Retriever(index_dir=idx)
    generator = ExtractiveAnswerGenerator(embedding_service=retriever.embedding_service)
    results = []
    for q in queries:
        chunks = retriever.retrieve_vector(q, top_k=5)
        resp = generator.generate(q, chunks)
        # Also capture the raw best score by inspecting generate internals
        import numpy as np
        candidate_sentences = []
        for chunk in chunks:
            sentences = re.split(r'([.?!।]+)', chunk.text)
            result_sents = []
            for i in range(0, len(sentences)-1, 2):
                sent = sentences[i].strip() + sentences[i+1].strip()
                if sent: result_sents.append(sent)
            if len(sentences) % 2 == 1 and sentences[-1].strip():
                result_sents.append(sentences[-1].strip())
            candidate_sentences.extend([s for s in result_sents if len(s.strip()) > 5])
        
        if candidate_sentences:
            qemb = retriever.embedding_service.encode_query(q)
            sembs = retriever.embedding_service.encode_documents(candidate_sentences)
            scores = np.dot(sembs, qemb)
            best_score = float(np.max(scores))
        else:
            best_score = 0.0

        results.append({
            'query': q,
            'answer': resp.answer,
            'source_chunk_ids': resp.source_chunk_ids,
            'is_refusal': "I don't have enough information" in resp.answer,
            'best_score': round(best_score, 8),
        })
    baseline[lang] = results
    print(f'{lang}: {N} queries captured')

out_path = 'data/processed/baseline_outputs.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)
print(f'Baseline saved to {out_path}')
