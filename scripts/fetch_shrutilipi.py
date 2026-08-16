import os
import sys
import urllib.request
import pyarrow as pa
import pyarrow.ipc as ipc
import soundfile as sf
import io
import json

sys.stdout.reconfigure(encoding='utf-8')

urls = {
    'tamil': 'https://huggingface.co/datasets/skesiraju/Shrutilipi/resolve/main/tamil/data-00000-of-00235.arrow?download=true',
    'telugu': 'https://huggingface.co/datasets/skesiraju/Shrutilipi/resolve/main/telugu/data-00000-of-00198.arrow?download=true',
    'malayalam': 'https://huggingface.co/datasets/skesiraju/Shrutilipi/resolve/main/malayalam/data-00000-of-00212.arrow?download=true'
}

prefixes = {
    'tamil': 'ta',
    'telugu': 'te',
    'malayalam': 'ml'
}

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/human_audio"))
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    os.makedirs(raw_dir, exist_ok=True)
    
    manifest = []
    
    for lang, url in urls.items():
        print(f"\nProcessing {lang}...")
        arrow_path = os.path.join(raw_dir, f"{lang}.arrow")
        
        if not os.path.exists(arrow_path):
            print(f"Downloading {lang}.arrow (approx 500MB)...")
            urllib.request.urlretrieve(url, arrow_path)
            print("Download complete.")
        else:
            print(f"{arrow_path} already exists. Using it.")
            
        print("Reading Arrow file...")
        with pa.OSFile(arrow_path, 'r') as f:
            reader = ipc.RecordBatchStreamReader(f)
            table = reader.read_all()
            
        print(f"Loaded {table.num_rows} rows.")
        
        count = 0
        prefix = prefixes[lang]
        
        for i in range(table.num_rows):
            if count >= 5:
                break
                
            row = table.slice(i, 1).to_pylist()[0]
            
            transcript = row.get('transcript') or row.get('sentence') or row.get('text')
            audio_col = row.get('audio')
            
            if not transcript or not audio_col:
                continue
                
            # PyArrow gives audio as a dict containing 'bytes' or a struct
            if isinstance(audio_col, dict) and 'bytes' in audio_col:
                audio_bytes = audio_col['bytes']
            elif hasattr(audio_col, 'get') and audio_col.get('bytes'):
                audio_bytes = audio_col.get('bytes')
            else:
                try:
                    # In some datasets, audio is [{'bytes': b'...', 'path': ...}] or just raw bytes
                    audio_bytes = audio_col['bytes']
                except:
                    continue
                    
            if not audio_bytes:
                continue
                
            filename = f"{prefix}_{count}.wav"
            out_path = os.path.join(out_dir, filename)
            
            # The bytes in Shrutilipi are usually encoded WAV/MP3/FLAC
            # Let's write the bytes to a temp file, then read with soundfile to verify/resave
            temp_path = out_path + ".temp"
            with open(temp_path, 'wb') as tf:
                tf.write(audio_bytes)
                
            try:
                data, sr = sf.read(temp_path)
                sf.write(out_path, data, sr)
                duration = len(data) / sr
                
                manifest.append({
                    "filename": filename,
                    "language": lang,
                    "original transcript": transcript,
                    "utterance_id": row.get('id', f"{prefix}_{count}"),
                    "score": row.get('score'),
                    "duration": duration,
                    "original sample rate": sr,
                    "final sample rate": sr
                })
                count += 1
                print(f"Saved {filename} - {duration:.2f}s")
            except Exception as e:
                print(f"Error processing audio {i}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
