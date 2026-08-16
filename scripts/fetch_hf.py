import os
import sys
import json
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.ipc as ipc
import soundfile as sf
import io

sys.stdout.reconfigure(encoding='utf-8')

targets = {
    'tamil': 'tamil/data-00000-of-00235.arrow',
    'telugu': 'telugu/data-00000-of-00198.arrow',
    'malayalam': 'malayalam/data-00000-of-00212.arrow'
}

prefixes = {
    'tamil': 'ta',
    'telugu': 'te',
    'malayalam': 'ml'
}

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/human_audio"))
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        manifest = []
        
    for lang, filename in targets.items():
        print(f"\nProcessing {lang}...")
        
        # Check if already 5 files exist
        prefix = prefixes[lang]
        existing = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith(".wav")]
        if len(existing) >= 5:
            print(f"Already have {len(existing)} files for {lang}. Skipping.")
            continue
            
        print(f"Downloading {filename}...")
        try:
            arrow_path = hf_hub_download(repo_id='skesiraju/Shrutilipi', filename=filename, repo_type='dataset')
        except Exception as e:
            print(f"Failed to download {lang}: {e}")
            continue
            
        print("Reading Arrow file...")
        with pa.OSFile(arrow_path, 'r') as f:
            reader = ipc.RecordBatchStreamReader(f)
            table = reader.read_all()
            
        print(f"Loaded {table.num_rows} rows.")
        
        count = len(existing)
        
        for i in range(table.num_rows):
            if count >= 5:
                break
                
            row = table.slice(i, 1).to_pylist()[0]
            
            transcript = row.get('transcript') or row.get('sentence') or row.get('text')
            audio_col = row.get('audio')
            
            if not transcript or not audio_col:
                continue
                
            if isinstance(audio_col, dict) and 'bytes' in audio_col:
                audio_bytes = audio_col['bytes']
            elif hasattr(audio_col, 'get') and audio_col.get('bytes'):
                audio_bytes = audio_col.get('bytes')
            else:
                continue
                
            if not audio_bytes:
                continue
                
            out_filename = f"{prefix}_{count}.wav"
            out_path = os.path.join(out_dir, out_filename)
            temp_path = out_path + ".temp"
            
            with open(temp_path, 'wb') as tf:
                tf.write(audio_bytes)
                
            try:
                data, sr = sf.read(temp_path)
                sf.write(out_path, data, sr)
                duration = len(data) / sr
                
                manifest.append({
                    "filename": out_filename,
                    "language": lang,
                    "original transcript": transcript,
                    "utterance_id": row.get('utterance_id') or row.get('id') or f"{prefix}_{count}",
                    "score": row.get('score'),
                    "duration": duration,
                    "original sample rate": sr,
                    "final sample rate": sr
                })
                count += 1
                print(f"Saved {out_filename} - {duration:.2f}s")
            except Exception as e:
                print(f"Error processing audio {i}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
