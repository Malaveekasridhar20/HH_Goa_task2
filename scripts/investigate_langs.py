import traceback
from datasets import load_dataset

dataset_name = "ai4bharat/MSMARCO-XI"

try:
    print(f"Loading {dataset_name} in streaming mode to find unique languages...")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    unique_langs = set()
    count = 0
    # Just sample the first 5000 records
    for i, row in enumerate(ds):
        lang = row.get("target_lang")
        if lang:
            unique_langs.add(lang)
        count += 1
        if count >= 10000:
            break
            
    print(f"Unique languages in first {count} rows: {unique_langs}")
    
except Exception as e:
    print("Error:")
    traceback.print_exc()
