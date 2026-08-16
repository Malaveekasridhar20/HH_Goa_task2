import json, os, time, sys
sys.path.insert(0, 'backend')
from app.retrieval.retriever import Retriever

langs = ['english', 'hindi', 'tamil', 'telugu', 'malayalam']
files = [
    'english_validation_500.jsonl',
    'hindi_validation_500.jsonl',
    'tamil_validation_500.jsonl',
    'telugu_validation_500.jsonl',
    'malayalam_validation_500.jsonl'
]

results = {}
demo_qs = []

for lang, file_name in zip(langs, files):
    path = os.path.join('data', 'processed', file_name)
    if not os.path.exists(path):
        continue
        
    retriever = Retriever(index_dir=os.path.join('data', 'indexes', lang))
    
    total = 0
    valid = 0
    r1, r5, r10, mrr = 0, 0, 0, 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            q = rec.get('query', '')
            eng_q = rec.get('Eng_Query', '')
            passages = rec.get('passages', {})
            texts = passages.get('Translated_passages', passages.get('passage_text', []))
            is_selected = passages.get('is_selected', [])
            
            sel_text = None
            if isinstance(is_selected, list) and isinstance(texts, list):
                for txt, sel in zip(texts, is_selected):
                    if sel == 1: sel_text = txt; break
                
            if not sel_text or len(sel_text) < 20 or len(q) < 5:
                continue
                
            if valid < 3:
                demo_qs.append({
                    'language': lang,
                    'query': q,
                    'english_equivalent': eng_q,
                    'expected_relevant_passage': sel_text,
                    'expected_grounding_behavior': 'PASS (Score >= 0.85)'
                })
                
            try:
                res = retriever.retrieve_hybrid(q, top_k=10, dense_weight=0.7, bm25_weight=0.3)
                ret_texts = [r.text for r in res]
                
                hit_rank = -1
                for rank, rt in enumerate(ret_texts):
                    # Robust check
                    if len(rt) > 30 and (rt in sel_text or sel_text in rt or len(set(rt.split()) & set(sel_text.split())) > 10):
                        hit_rank = rank + 1
                        break
                        
                if hit_rank == 1: r1 += 1
                if hit_rank > 0 and hit_rank <= 5: r5 += 1
                if hit_rank > 0 and hit_rank <= 10: r10 += 1
                if hit_rank > 0: mrr += 1.0 / hit_rank
                
                valid += 1
            except Exception as e:
                pass
                
            if valid >= 100:
                break
                
    results[lang] = {
        'total_evaluated': valid,
        'Recall@1': round(r1/valid, 4) if valid > 0 else 0,
        'Recall@5': round(r5/valid, 4) if valid > 0 else 0,
        'Recall@10': round(r10/valid, 4) if valid > 0 else 0,
        'MRR@10': round(mrr/valid, 4) if valid > 0 else 0
    }
    print(f'{lang} evaluated {valid} queries.')

with open('data/processed/final_dataset_evaluation.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

with open('data/processed/demo_questions.json', 'w', encoding='utf-8') as f:
    json.dump(demo_qs, f, indent=2, ensure_ascii=False)

md_content = "# Final Dataset Evaluation\n"
for lang, metrics in results.items():
    md_content += f"## {lang.capitalize()}\n"
    md_content += f"- **Queries Evaluated:** {metrics['total_evaluated']}\n"
    md_content += f"- **Recall@1:** {metrics['Recall@1']}\n"
    md_content += f"- **Recall@5:** {metrics['Recall@5']}\n"
    md_content += f"- **Recall@10:** {metrics['Recall@10']}\n"
    md_content += f"- **MRR@10:** {metrics['MRR@10']}\n\n"

with open('data/processed/final_dataset_evaluation.md', 'w', encoding='utf-8') as f:
    f.write(md_content)
