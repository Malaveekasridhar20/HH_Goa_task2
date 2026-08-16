import os
import sys
import json
import time
from typing import List, Dict

# Fix unicode printing in Windows console
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app.config import settings
from app.chunking.pipeline import ChunkingPipeline

def fetch_dataset_rows(file_path: str, max_records: int) -> List[Dict]:
    """Reads from local JSONL extracted from official dataset files."""
    rows = []
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return rows
        
    print(f"Reading up to {max_records} rows from direct access file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_records:
                break
            try:
                row = json.loads(line)
                rows.append(row)
            except Exception as e:
                print(f"Error parsing row {idx}: {e}")
                
    return rows

def generate_statistics(chunks, time_taken: float, record_count: int, passage_count: int) -> dict:
    stats = {
        "records_processed": record_count,
        "passages_extracted": passage_count,
        "chunks_generated": len(chunks),
        "average_chunks_per_passage": len(chunks) / max(1, passage_count),
        "processing_duration_seconds": time_taken,
        "strategies": {
            "whole_passage": 0,
            "semantic": 0,
            "sliding_window": 0,
            "naive_fixed": 0
        }
    }
    
    for c in chunks:
        stats["strategies"][c.strategy] = stats["strategies"].get(c.strategy, 0) + 1
        
    return stats

def process_language(lang_name: str, file_path: str, max_records: int):
    print(f"\n=== Building Chunks for {lang_name} ===")
    
    records = fetch_dataset_rows(file_path, max_records)
    print(f"Actually fetched records: {len(records)}")
    if not records:
        print("Failed to fetch records. Skipping.")
        return

    pipeline = ChunkingPipeline(adaptive=True)
    extract_english = False
    
    start_time = time.time()
    adaptive_passages = []
    adaptive_chunks = []
    
    for row in records:
        try:
            extracted = pipeline.extractor.extract_passages(row, extract_english=extract_english)
            adaptive_passages.extend(extracted)
            c = pipeline.process_record(row, extract_english=extract_english)
            adaptive_chunks.extend(c)
        except Exception as e:
            print(f"Malformed record error: {e}")
            
    time_taken = time.time() - start_time
    
    stats = generate_statistics(adaptive_chunks, time_taken, len(records), len(adaptive_passages))
    
    out_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    os.makedirs(out_dir, exist_ok=True)
    
    chunks_path = os.path.join(out_dir, f'chunks_{lang_name}.jsonl')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for c in adaptive_chunks:
            f.write(c.model_dump_json() + '\n')
            
    report_path = os.path.join(out_dir, f'chunking_report_{lang_name}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    print(f"Passages extracted: {len(adaptive_passages)}")
    print(f"Chunks generated: {len(adaptive_chunks)}")
    print(f"Saved to {chunks_path}")

def main():
    max_records = 500
    languages = {
        "tamil": "../data/processed/tamil_validation_500.jsonl",
        "telugu": "../data/processed/telugu_validation_500.jsonl",
        "malayalam": "../data/processed/malayalam_validation_500.jsonl"
    }
    
    for lang, path in languages.items():
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
        process_language(lang, full_path, max_records)

if __name__ == "__main__":
    main()
