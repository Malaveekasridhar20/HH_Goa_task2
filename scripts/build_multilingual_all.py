import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from scripts.build_multilingual_chunks import process_language as build_chunks
from scripts.build_multilingual_indexes import process_language as build_indexes

def main():
    languages = {
        "tamil": "../data/processed/tamil_validation_500.jsonl",
        "telugu": "../data/processed/telugu_validation_500.jsonl",
        "malayalam": "../data/processed/malayalam_validation_500.jsonl"
    }
    
    for lang, path in languages.items():
        print(f"\n--- Processing {lang.upper()} ---")
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
        
        # 1. Build chunks
        print(f"Building chunks for {lang}...")
        build_chunks(lang, full_path, 20)
        
        # 2. Build indexes
        print(f"Building indexes for {lang}...")
        build_indexes(lang)
        
if __name__ == "__main__":
    main()
