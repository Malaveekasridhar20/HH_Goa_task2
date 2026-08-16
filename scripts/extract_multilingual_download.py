import os
import requests
import pandas as pd
import json

def download_file(url, local_filename):
    print(f"Downloading {url} to {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def main():
    target_langs = ["ta", "te", "ml"]
    max_records = 500
    counts = {lang: 0 for lang in target_langs}
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    
    parquet_path = os.path.join(out_dir, "0000.parquet")
    
    url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
    
    if not os.path.exists(parquet_path):
        download_file(url, parquet_path)
    else:
        print("Parquet file already exists.")
        
    print("Reading parquet file...")
    df = pd.read_parquet(parquet_path)
    print(f"Total rows in parquet: {len(df)}")
    
    file_handles = {
        "ta": open(os.path.join(out_dir, "tamil_validation_500.jsonl"), "w", encoding="utf-8"),
        "te": open(os.path.join(out_dir, "telugu_validation_500.jsonl"), "w", encoding="utf-8"),
        "ml": open(os.path.join(out_dir, "malayalam_validation_500.jsonl"), "w", encoding="utf-8"),
    }
    
    try:
        for _, row in df.iterrows():
            lang = row.get("target_lang")
            if lang in target_langs and counts[lang] < max_records:
                row_dict = row.to_dict()
                for k, v in row_dict.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            if hasattr(v2, 'tolist'):
                                v[k2] = v2.tolist()
                    if hasattr(v, 'tolist'):
                        row_dict[k] = v.tolist()
                        
                file_handles[lang].write(json.dumps(row_dict, ensure_ascii=False) + "\n")
                counts[lang] += 1
                
                if all(c >= max_records for c in counts.values()):
                    break
        print(f"Extraction complete! Counts: {counts}")
    finally:
        for f in file_handles.values():
            f.close()

if __name__ == "__main__":
    main()
