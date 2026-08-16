import pandas as pd
import json
import os
import io
import urllib.request

def main():
    target_langs = ["ta", "te", "ml"]
    max_records = 500
    counts = {lang: 0 for lang in target_langs}
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    
    file_handles = {
        "ta": open(os.path.join(out_dir, "tamil_validation_500.jsonl"), "w", encoding="utf-8"),
        "te": open(os.path.join(out_dir, "telugu_validation_500.jsonl"), "w", encoding="utf-8"),
        "ml": open(os.path.join(out_dir, "malayalam_validation_500.jsonl"), "w", encoding="utf-8"),
    }
    
    # We will just fetch validation parquet 0000 to 0002.
    urls = [
        "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/refs%2Fconvert%2Fparquet/default/validation/0001.parquet",
        "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/refs%2Fconvert%2Fparquet/default/validation/0002.parquet"
    ]
    
    try:
        for url in urls:
            if all(c >= max_records for c in counts.values()):
                break
                
            print(f"Reading parquet from {url}...")
            # We can use pandas direct http read
            df = pd.read_parquet(url)
            
            for _, row in df.iterrows():
                lang = row.get("target_lang")
                if lang in target_langs and counts[lang] < max_records:
                    # Convert row to dict
                    row_dict = row.to_dict()
                    # Convert numpy arrays to lists if any
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
        print(f"Error: {e}")
    finally:
        for f in file_handles.values():
            f.close()
            
if __name__ == "__main__":
    main()
