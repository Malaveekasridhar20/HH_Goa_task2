import pyarrow.parquet as pq
import fsspec
import json
import os
import numpy as np
import pandas as pd

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    return obj

def extract_lazy(lang_code, lang_name, num_records=500):
    print(f"Lazy reading parquet for {lang_name} ({lang_code})...")
    
    if lang_code == "ta":
        parquet_name = "tamval.parquet"
    elif lang_code == "te":
        parquet_name = "telval.parquet"
    elif lang_code == "ml":
        parquet_name = "malval.parquet"
    else:
        raise ValueError(f"Unknown language {lang_code}")
        
    url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/{parquet_name}?download=true"
    
    # Use fsspec to open the HTTP file dynamically
    fs = fsspec.filesystem('http')
    
    with fs.open(url, 'rb') as f:
        # Read parquet metadata using pyarrow
        pf = pq.ParquetFile(f)
        
        print(f"File has {pf.num_row_groups} row groups and {pf.metadata.num_rows} total rows.")
        
        # Read row groups until we have `num_records`
        df_list = []
        rows_read = 0
        
        for i in range(pf.num_row_groups):
            print(f"Reading row group {i}...")
            # Read just this row group
            rg = pf.read_row_group(i).to_pandas()
            df_list.append(rg)
            rows_read += len(rg)
            if rows_read >= num_records:
                break
                
        df = pd.concat(df_list, ignore_index=True)
        df_sample = df.head(num_records)
        
        out_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{lang_name}_validation_{num_records}.jsonl")
        
        with open(out_file, 'w', encoding='utf-8') as out_f:
            for _, row in df_sample.iterrows():
                row_dict = convert_to_json_serializable(row.to_dict())
                out_f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
                
        print(f"Saved {num_records} records to {out_file}.")
        return out_file

if __name__ == "__main__":
    extract_lazy("ta", "tamil", 500)
    extract_lazy("te", "telugu", 500)
    extract_lazy("ml", "malayalam", 500)
