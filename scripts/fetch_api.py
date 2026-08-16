import urllib.request
import urllib.parse
import json
import os

def fetch_lang_data(lang, num_records=500):
    url = f"https://datasets-server.huggingface.co/filter?dataset=ai4bharat/MSMARCO-XI&config=default&split=validation&where=target_lang='{lang}'&length={num_records}"
    print(f"Fetching {lang} from {url}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(out_dir, exist_ok=True)
    
    file_name = f"{'tamil' if lang=='ta' else 'telugu' if lang=='te' else 'malayalam'}_validation_500.jsonl"
    out_path = os.path.join(out_dir, file_name)
    
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for row in data.get('rows', []):
            if count >= num_records: break
            row_data = row.get('row', {})
            f.write(json.dumps(row_data, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"Saved {count} records to {out_path}")

def main():
    try:
        fetch_lang_data('ta', 500)
        fetch_lang_data('te', 500)
        fetch_lang_data('ml', 500)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
