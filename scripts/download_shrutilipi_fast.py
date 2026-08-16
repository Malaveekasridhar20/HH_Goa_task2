import os
import sys
import json
from datasets import load_dataset, Audio
import shutil

sys.stdout.reconfigure(encoding='utf-8')

def download_audio_samples():
    print("Loading specific arrow files directly...")
    
    # We download 1 arrow file per language to avoid downloading 100s of GBs
    data_files = {
        'tamil': 'tamil/data-00000-of-00235.arrow',
        'telugu': 'telugu/data-00000-of-00198.arrow',
        'malayalam': 'malayalam/data-00000-of-00212.arrow'
    }
    
    try:
        ds = load_dataset('skesiraju/Shrutilipi', data_files=data_files)
        # Avoid trying to decode using torchaudio
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/human_audio"))
    os.makedirs(out_dir, exist_ok=True)
    
    target_langs = {
        'tamil': {'count': 0, 'limit': 5, 'prefix': 'ta'},
        'telugu': {'count': 0, 'limit': 5, 'prefix': 'te'},
        'malayalam': {'count': 0, 'limit': 5, 'prefix': 'ml'}
    }
    
    manifest = []
    
    print("Iterating over loaded splits...")
    for split_name, split_ds in ds.items():
        lang_key = split_name
        if lang_key not in target_langs:
            continue
            
        print(f"Processing split {lang_key} with {len(split_ds)} rows")
        
        for idx in range(min(50, len(split_ds))):
            sample = split_ds[idx]
            
            if target_langs[lang_key]['count'] >= target_langs[lang_key]['limit']:
                break
                
            audio_data = sample.get('audio')
            if not audio_data:
                continue
                
            text = sample.get('transcript') or sample.get('sentence') or sample.get('text')
            if not text:
                continue
            
            prefix = target_langs[lang_key]['prefix']
            count = target_langs[lang_key]['count']
            filename = f"{prefix}_{count}.wav"
            out_path = os.path.join(out_dir, filename)
            
            # Save audio bytes
            if 'bytes' in audio_data and audio_data['bytes']:
                with open(out_path, 'wb') as af:
                    af.write(audio_data['bytes'])
            elif 'path' in audio_data and audio_data['path']:
                shutil.copy2(audio_data['path'], out_path)
            else:
                continue
                
            manifest.append({
                "filename": filename,
                "language": lang_key,
                "original transcript": text,
                "utterance_id": sample.get('id') or sample.get('path') or f"{prefix}_{count}",
                "score": sample.get('score'),
                "duration": 0.0,
                "original sample rate": 16000,
                "final sample rate": 16000
            })
            
            target_langs[lang_key]['count'] += 1
            print(f"Saved {filename} ({lang_key})")
            
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print("Done! Extracted samples:")
    for k, v in target_langs.items():
        print(f"  {k}: {v['count']}/{v['limit']}")

if __name__ == "__main__":
    download_audio_samples()
