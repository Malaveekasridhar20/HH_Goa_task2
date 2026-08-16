import os
import json
import traceback
from datasets import load_dataset

def main():
    dataset_name = "ai4bharat/MSMARCO-XI"
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
    
    try:
        ds = load_dataset(dataset_name, split="validation", streaming=True)
        print("Extracting records from validation stream...")
        for i, row in enumerate(ds):
            lang = row.get("target_lang")
            if lang in target_langs and counts[lang] < max_records:
                file_handles[lang].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[lang] += 1
            if i % 10000 == 0:
                print(f"Scanned {i} rows. Found: {counts}")
            if all(c >= max_records for c in counts.values()):
                break
        print(f"Extraction complete! Final counts: {counts}")
    except Exception as e:
        print("Error during extraction:", e)
    finally:
        for f in file_handles.values():
            f.close()

if __name__ == "__main__":
    main()
