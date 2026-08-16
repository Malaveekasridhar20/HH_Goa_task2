import os
import json
from datasets import load_dataset
import numpy as np

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    return obj

def extract_with_datasets(lang_code, lang_name, parquet_name, num_records=500):
    print(f"Extracting {lang_name} using HF datasets streaming...")
    try:
        # Load the specific language validation parquet file
        ds = load_dataset("ai4bharat/MSMARCO-XI", data_files={"validation": f"validation/{parquet_name}"}, split="validation", streaming=True)
        
        records = []
        for i, row in enumerate(ds):
            records.append(row)
            if i + 1 >= num_records:
                break
                
        out_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{lang_name}_validation_{num_records}.jsonl")
        
        with open(out_file, 'w', encoding='utf-8') as f:
            for row in records:
                row_dict = convert_to_json_serializable(row)
                f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
                
        print(f"Successfully saved {len(records)} records for {lang_name}")
        return True
    except Exception as e:
        print(f"Error extracting {lang_name}: {e}")
        return False

if __name__ == "__main__":
    extract_with_datasets("te", "telugu", "telval.parquet", 500)
    extract_with_datasets("ml", "malayalam", "malval.parquet", 500)
