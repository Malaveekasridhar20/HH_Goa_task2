import requests
import json

dataset = "ai4bharat/MSMARCO-XI"
print(f"Investigating {dataset} via Datasets Server API...")

try:
    # Get basic dataset info
    res = requests.get(f"https://datasets-server.huggingface.co/info?dataset={dataset}")
    info = res.json()
    
    if "dataset_info" in info:
        for config, data in info["dataset_info"].items():
            print(f"\nConfig: {config}")
            features = data.get("features", {})
            print("Features:", list(features.keys()))
            
            # Print total splits and sizes
            for split, split_info in data.get("splits", {}).items():
                print(f"Split {split}: {split_info.get('num_examples')} examples")
                
    # To get language counts, we might need parquet or first rows
    # Actually, AI4Bharat MSMARCO-XI is known to have 11 Indic languages + English.
    # We can fetch first few rows to see target_lang
    print("\nFetching sample rows to see target_lang...")
    res = requests.get(f"https://datasets-server.huggingface.co/first-rows?dataset={dataset}&config=default&split=train")
    rows = res.json().get("rows", [])
    
    langs = set()
    for row in rows:
        lang = row.get("row", {}).get("target_lang")
        if lang: langs.add(lang)
    print("Languages in sample:", langs)
    
except Exception as e:
    print("Error:", e)
