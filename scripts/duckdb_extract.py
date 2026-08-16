import duckdb
import os
import json

def extract_duckdb(lang_code, lang_name, num_records=500):
    print(f"Extracting {num_records} from {lang_name} using DuckDB...")
    if lang_code == "ta":
        parquet_name = "tamval.parquet"
    elif lang_code == "te":
        parquet_name = "telval.parquet"
    elif lang_code == "ml":
        parquet_name = "malval.parquet"
    else:
        raise ValueError("Unknown lang")
        
    url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/{parquet_name}"
    
    # DuckDB can run httpfs
    query = f"SELECT * FROM read_parquet('{url}') LIMIT {num_records}"
    
    conn = duckdb.connect()
    # Install and load httpfs if not loaded automatically
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    
    df = conn.execute(query).df()
    
    out_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{lang_name}_validation_500.jsonl")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Handle numpy arrays if any
            for k, v in row_dict.items():
                if hasattr(v, "tolist"):
                    row_dict[k] = v.tolist()
            f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(df)} records to {out_file}")

if __name__ == "__main__":
    extract_duckdb("te", "telugu", 500)
    extract_duckdb("ml", "malayalam", 500)
    # Tamil is already extracted using robust downloader + pandas
