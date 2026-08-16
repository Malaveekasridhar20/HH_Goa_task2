import os
import json
import time
from deep_translator import GoogleTranslator

def translate_row(row, translator):
    new_row = json.loads(json.dumps(row)) # deep copy
    
    # Translate query
    if "Eng_Query" in new_row:
        try:
            new_row["query"] = translator.translate(new_row["Eng_Query"])
        except:
            pass
            
    # Translate passages
    if "passages" in new_row and "English_passages" in new_row["passages"]:
        translated_passages = []
        for idx, p in enumerate(new_row["passages"]["English_passages"]):
            if new_row["passages"]["is_selected"][idx] == 1:
                try:
                    p_text = p[:2000]
                    t = translator.translate(p_text)
                    translated_passages.append(t)
                except Exception as e:
                    translated_passages.append(p)
            else:
                translated_passages.append("dummy passage")
        new_row["passages"]["Translated_passages"] = translated_passages
        
    return new_row

def main():
    langs = {
        "tamil": "ta",
        "telugu": "te",
        "malayalam": "ml"
    }
    
    in_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/hinval_500.jsonl'))
    out_dir = os.path.dirname(in_file)
    
    rows = []
    with open(in_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 20:
                break
            rows.append(json.loads(line))
            
    print(f"Translating {len(rows)} records...")
    
    for name, code in langs.items():
        print(f"Translating to {name} ({code})...")
        translator = GoogleTranslator(source='auto', target=code)
        
        out_path = os.path.join(out_dir, f"{name}_validation_500.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, row in enumerate(rows):
                translated = translate_row(row, translator)
                translated["target_lang"] = code
                f.write(json.dumps(translated, ensure_ascii=False) + "\n")
                if i % 5 == 0:
                    print(f"  {i}/{len(rows)}")
                    
        print(f"Finished {name}")

if __name__ == "__main__":
    main()
