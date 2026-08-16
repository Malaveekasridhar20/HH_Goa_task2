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
import requests

def fetch_dataset_rows(dataset: str, config: str, split: str, max_records: int) -> List[Dict]:
    """Reads from local JSONL extracted from official dataset files."""
    rows = []
    
    file_path = os.path.join(os.path.dirname(__file__), '../data/processed/hinval_500.jsonl')
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
        "average_processing_time_per_record": time_taken / max(1, record_count),
        "average_processing_time_per_passage": time_taken / max(1, passage_count),
        "strategies": {
            "whole_passage": 0,
            "semantic": 0,
            "sliding_window": 0,
            "naive_fixed": 0
        },
        "selection": {
            "selected_chunks": 0,
            "non_selected_chunks": 0
        },
        "lengths": {
            "min": float('inf'),
            "max": 0,
            "avg": 0,
            "total": 0
        }
    }
    
    for c in chunks:
        stats["strategies"][c.strategy] = stats["strategies"].get(c.strategy, 0) + 1
        if c.is_selected:
            stats["selection"]["selected_chunks"] += 1
        else:
            stats["selection"]["non_selected_chunks"] += 1
            
        l = len(c.text)
        if l < stats["lengths"]["min"]:
            stats["lengths"]["min"] = l
        if l > stats["lengths"]["max"]:
            stats["lengths"]["max"] = l
        stats["lengths"]["total"] += l
        
    if len(chunks) > 0:
        stats["lengths"]["avg"] = stats["lengths"]["total"] / len(chunks)
    else:
        stats["lengths"]["min"] = 0
        
    return stats

def main():
    DATASET_NAME = settings.dataset_name
    DATASET_CONFIG = os.getenv("DATASET_CONFIG", "default")
    DATASET_SPLIT = os.getenv("DATASET_SPLIT", "validation")  # Validation is smaller and more reliable
    MAX_RECORDS = int(os.getenv("MAX_RECORDS", "500"))
    
    print("=== Phase 2: Building Chunks ===")
    print(f"Dataset config: {DATASET_CONFIG}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Target Records: {MAX_RECORDS}")
    
    records = fetch_dataset_rows(DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, MAX_RECORDS)
    print(f"Actually fetched records: {len(records)}")
    if not records:
        print("Failed to fetch records. Exiting.")
        return

    # Adaptive Pipeline
    pipeline = ChunkingPipeline(adaptive=True)
    naive_pipeline = ChunkingPipeline(adaptive=False)
    
    # Process Adaptive
    start_time = time.time()
    extract_english = os.getenv("EXTRACT_ENGLISH", "0") == "1"
    
    # 2. Extract Passages
    adaptive_passages = []
    malformed_records = 0
    adaptive_chunks = []
    
    for row in records:
        try:
            extracted = pipeline.extractor.extract_passages(row, extract_english=extract_english)
            adaptive_passages.extend(extracted)
            c = pipeline.process_record(row, extract_english=extract_english)
            adaptive_chunks.extend(c)
        except Exception as e:
            print(f"Malformed record error: {e}")
            malformed_records += 1
            
    adaptive_time = time.time() - start_time
    time_taken = adaptive_time
    
    # Process Naive Baseline
    start_time_n = time.time()
    naive_chunks = []
    naive_passage_count = 0
    for r in records:
        passages = naive_pipeline.extractor.extract_passages(r, extract_english=extract_english)
        naive_passage_count += len(passages)
        c = naive_pipeline.process_record(r, extract_english=extract_english)
        naive_chunks.extend(c)
    naive_time = time.time() - start_time_n
    
    print("\n--- Generating Reports ---")
    naive_stats = generate_statistics(naive_chunks, naive_time, len(records), naive_passage_count)
    
    # Comparison Report
    comparison = {
        "dataset_config": DATASET_CONFIG,
        "records_processed": len(records),
        "naive_baseline": naive_stats,
        "malfored_records": malformed_records
    }
    
    # 4. Save results
    prefix = "english_" if extract_english else ""
    out_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    os.makedirs(out_dir, exist_ok=True)
    
    chunks_path = os.path.join(out_dir, f'{prefix}chunks.jsonl')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for c in adaptive_chunks:
            f.write(c.model_dump_json() + '\n')
            
    # Save Report
    report_path = os.path.join(out_dir, f'{prefix}chunking_report.json')
    stats = generate_statistics(adaptive_chunks, time_taken, len(records), len(adaptive_passages))
    
    # Save full report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "records_processed": len(records),
            "passages_extracted": len(adaptive_passages),
            "selected_passages": len([p for p in adaptive_passages if p.is_selected]),
            "non_selected_passages": len([p for p in adaptive_passages if not p.is_selected]),
            "chunking_configuration": {
                "type": "adaptive",
                "short_threshold": pipeline.adaptive_chunker.short_threshold,
                "long_threshold": pipeline.adaptive_chunker.long_threshold,
                "semantic_target": pipeline.adaptive_chunker.semantic_chunker.target_chunk_size
            },
            "statistics": stats
        }, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(out_dir, f'{prefix}chunking_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
        
    print(f"Passages extracted: {len(adaptive_passages)}")
    print(f"Chunks generated: {len(adaptive_chunks)}")
    print(f"Files saved to data/processed/ (Prefix: '{prefix}')")
    
    # Synthetic Stress Test
    print("\n=== Synthetic Stress Test ===")
    
    synthetic_passages = [
        "Short passage.",
        "This is a medium passage. It has multiple sentences. It should use semantic chunking. We need it to be long enough to exceed the short threshold, so I am adding more text. " * 3,
        "This is a long passage. " * 50,
        "ThisIsAnExtremelyLongPassageWithNoPunctuation" * 50
    ]
    
    stress_pipeline = ChunkingPipeline(adaptive=True)
    # Manually configure thresholds for deterministic testing
    stress_pipeline.adaptive_chunker.short_threshold = 50
    stress_pipeline.adaptive_chunker.semantic_chunker.target_chunk_size = 100
    stress_pipeline.adaptive_chunker.long_threshold = 200
    stress_pipeline.adaptive_chunker.fixed_chunker.chunk_size = 200
    stress_pipeline.adaptive_chunker.fixed_chunker.overlap = 50
    
    coverage = set()
    for idx, p in enumerate(synthetic_passages):
        chunks = stress_pipeline.adaptive_chunker.chunk_text(p)
        strat = chunks[0][3] if chunks else "none"
        coverage.add(strat)
        
        name = ["Short", "Medium", "Long", "Very long"][idx]
        print(f"{name} passage (len={len(p)}): routed to '{strat}', chunks={len(chunks)}")
        
    if "whole_passage" in coverage and "semantic" in coverage and "sliding_window" in coverage:
        print("Strategy coverage: PASS")
    else:
        print("Strategy coverage: FAIL")

if __name__ == "__main__":
    main()
