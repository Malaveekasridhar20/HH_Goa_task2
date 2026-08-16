import sys
import os
import json
import requests

# Fix unicode printing in Windows console
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from app.config import settings  # type: ignore

def main():
    print(f"Inspecting dataset: {settings.dataset_name} via Hugging Face Datasets Server API")
    dataset = settings.dataset_name
    
    report = {
        "dataset_name": dataset,
        "splits": [],
        "records_per_split": {},
        "columns": {},
        "examples": {},
        "approximate_text_lengths": {},
        "missing_null_values": {},
        "metadata_fields": {}
    }

    try:
        # Get info to find configs and splits
        info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset}"
        res = requests.get(info_url, timeout=10)
        res.raise_for_status()
        info_data = res.json()
        
        # We will just inspect the 'hi' (Hindi) configuration or default if available
        # ai4bharat/MSMARCO-XI has language codes as configs: asm, ben, guj, hin, etc.
        configs = info_data.get("dataset_info", {})
        config_to_use = "hin" if "hin" in configs else list(configs.keys())[0] if configs else "default"
        print(f"Using configuration: {config_to_use}")
        
        if config_to_use in configs:
            splits_info = configs[config_to_use].get("splits", {})
            for split_name, split_obj in splits_info.items():
                report["splits"].append(split_name)
                report["records_per_split"][split_name] = split_obj.get("num_examples", 0)
        else:
            report["splits"] = ["train"]
            report["records_per_split"] = {"train": "Unknown"}

        for split in report["splits"]:
            print(f"\n--- Split: {split} ({config_to_use}) ---")
            print(f"Records: {report['records_per_split'].get(split)}")
            
            # Get first rows
            rows_url = f"https://datasets-server.huggingface.co/first-rows?dataset={dataset}&config={config_to_use}&split={split}"
            rows_res = requests.get(rows_url, timeout=10)
            if rows_res.status_code == 200:
                rows_data = rows_res.json()
                features = rows_data.get("features", [])
                report["columns"][split] = [f["name"] for f in features]
                print(f"Columns: {report['columns'][split]}")
                
                rows = rows_data.get("rows", [])
                examples = [r["row"] for r in rows[:5]]
                
                if examples:
                    report["examples"][split] = examples[:2]
                    print("First 2 examples:")
                    for ex in examples[:2]:
                        print(ex)
            else:
                print(f"Could not fetch rows for {split}: {rows_res.status_code}")
                
    except Exception as e:
        print(f"API request failed: {e}")
            
    os.makedirs(os.path.join(os.path.dirname(__file__), '../data/processed'), exist_ok=True)
    report_path = os.path.join(os.path.dirname(__file__), '../data/processed/dataset_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
