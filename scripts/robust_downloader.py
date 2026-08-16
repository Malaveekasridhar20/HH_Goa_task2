import os
import requests
import time

def download_robust(url, dest_path):
    print(f"Downloading {url} to {dest_path}")
    headers = {}
    if os.path.exists(dest_path):
        downloaded = os.path.getsize(dest_path)
        headers['Range'] = f'bytes={downloaded}-'
        print(f"Resuming from {downloaded} bytes")
    else:
        downloaded = 0

    mode = 'ab' if downloaded > 0 else 'wb'
    
    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code == 416: # Range not satisfiable (already fully downloaded)
            print("Already fully downloaded.")
            return True
        r.raise_for_status()
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download complete.")
    return True

def download_with_retries(url, dest_path, retries=10):
    for attempt in range(retries):
        try:
            download_robust(url, dest_path)
            return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    download_with_retries('https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/tamval.parquet?download=true', 'data/raw/tamval.parquet')
    download_with_retries('https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/telval.parquet?download=true', 'data/raw/telval.parquet')
    download_with_retries('https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/malval.parquet?download=true', 'data/raw/malval.parquet')
