import os
import urllib.request
import urllib.error
import sys

def report_hook(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        if percent % 10 == 0:
            print(f"\rDownloading... {percent}% ({count * block_size / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB)", end="")
            sys.stdout.flush()

def main():
    url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "0000.parquet")
    
    # If the file exists but is incomplete, remove it
    if os.path.exists(parquet_path) and os.path.getsize(parquet_path) < 400 * 1024 * 1024:
        print(f"Removing incomplete file {parquet_path}")
        os.remove(parquet_path)
        
    if not os.path.exists(parquet_path):
        print(f"Downloading {url} to {parquet_path}...")
        try:
            urllib.request.urlretrieve(url, parquet_path, reporthook=report_hook)
            print("\nDownload finished.")
        except Exception as e:
            print(f"\nError downloading: {e}")
            return
    else:
        print("Parquet file already exists and looks complete.")
        
    import pandas as pd
    import json
    
    print("Loading parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.")
    
    target_langs = ["ta", "te", "ml"]
    max_records = 500
    counts = {lang: 0 for lang in target_langs}
    
    file_handles = {
        "ta": open(os.path.join(out_dir, "tamil_validation_500.jsonl"), "w", encoding="utf-8"),
        "te": open(os.path.join(out_dir, "telugu_validation_500.jsonl"), "w", encoding="utf-8"),
        "ml": open(os.path.join(out_dir, "malayalam_validation_500.jsonl"), "w", encoding="utf-8"),
    }
    
    print("Extracting target languages...")
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
                    print(f"Extraction complete! Counts: {counts}")
                    break
    except Exception as e:
        print(f"Extraction error: {e}")
    finally:
        for f in file_handles.values():
            f.close()

if __name__ == "__main__":
    main()
