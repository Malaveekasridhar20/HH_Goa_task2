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

def download_and_extract(lang_code, lang_name, num_records=500):
    print(f"Downloading validation parquet for {lang_name} ({lang_code})...")
    if lang_code == "ta":
        parquet_name = "tamval.parquet"
    elif lang_code == "te":
        parquet_name = "telval.parquet"
    elif lang_code == "ml":
        parquet_name = "malval.parquet"
    else:
        raise ValueError(f"Unknown language {lang_code}")
        
    url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/{parquet_name}?download=true"
    
    # Read parquet into pandas directly
    df = pd.read_parquet(url)
    
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
    download_and_extract("ta", "tamil", 500)
    download_and_extract("te", "telugu", 500)
    download_and_extract("ml", "malayalam", 500)
