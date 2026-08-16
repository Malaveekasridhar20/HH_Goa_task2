import pandas as pd
import json
import os
import numpy as np

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    return obj

def extract_from_local_parquet(lang_name, parquet_file, num_records=500):
    print(f"Reading local parquet for {lang_name} from {parquet_file}...")
    
    # Read parquet into pandas
    df = pd.read_parquet(parquet_file)
    
    print(f"Loaded {len(df)} total validation records for {lang_name}.")
    
    # Select first `num_records`
    df_sample = df.head(num_records)
    
    # Save as JSONL
    out_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{lang_name}_validation_500.jsonl")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for _, row in df_sample.iterrows():
            row_dict = convert_to_json_serializable(row.to_dict())
            f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
            
    print(f"Saved {num_records} records to {out_file}.")
    return out_file

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
    extract_from_local_parquet("tamil", os.path.join(base_dir, "tamval.parquet"), 500)
    extract_from_local_parquet("telugu", os.path.join(base_dir, "telval.parquet"), 500)
    extract_from_local_parquet("malayalam", os.path.join(base_dir, "malval.parquet"), 500)
