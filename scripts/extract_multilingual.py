import os
import json
import traceback
from datasets import load_dataset

def main():
    dataset_name = "ai4bharat/MSMARCO-XI"
    target_langs = ["ta", "te", "ml"]
    max_records = 500
    
    # Track counts
    counts = {lang: 0 for lang in target_langs}
    
    # Setup output files
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    
    file_handles = {
        "ta": open(os.path.join(out_dir, "tamil_validation_500.jsonl"), "w", encoding="utf-8"),
        "te": open(os.path.join(out_dir, "telugu_validation_500.jsonl"), "w", encoding="utf-8"),
        "ml": open(os.path.join(out_dir, "malayalam_validation_500.jsonl"), "w", encoding="utf-8"),
    }
    
    try:
        print(f"Loading {dataset_name} validation split in streaming mode...")
        # Use validation split if available, otherwise train
        try:
            ds = load_dataset(dataset_name, split="validation", streaming=True)
        except:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            
        print("Extracting records...")
        
        for row in ds:
            lang = row.get("target_lang")
            if lang in target_langs and counts[lang] < max_records:
                # Write to respective file
                file_handles[lang].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[lang] += 1
                
                # Check if all done
                if all(c >= max_records for c in counts.values()):
                    break
                    
        print(f"Extraction complete! Counts: {counts}")
        
    except Exception as e:
        print("Error during extraction:")
        traceback.print_exc()
    finally:
        for f in file_handles.values():
            f.close()

if __name__ == "__main__":
    main()
