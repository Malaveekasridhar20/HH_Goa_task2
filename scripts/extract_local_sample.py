import pyarrow.parquet as pq
import json
import os

def extract_sample():
    print("Reading local hinval.parquet...")
    table = pq.read_table('hinval.parquet')
    
    # We only need 500 rows for the Phase 2 benchmark
    num_records = min(500, table.num_rows)
    print(f"Extracting {num_records} records...")
    
    records = table.slice(0, num_records).to_pylist()
    
    out_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'hinval_500.jsonl')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Successfully saved {num_records} records to {out_file}")

if __name__ == "__main__":
    extract_sample()
