import os
import sys
import json
import soundfile as sf
from datasets import load_dataset, Audio

sys.stdout.reconfigure(encoding='utf-8')

def download_audio_samples():
    print("Loading dataset in streaming mode...")
    
    try:
        ds = load_dataset('skesiraju/Shrutilipi', streaming=True, split='train')
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
    
    print("Iterating over dataset...")
    for idx, sample in enumerate(ds):
        lang = sample.get('language') or sample.get('locale') or sample.get('lang')
        if not lang:
            pass
            
        if isinstance(lang, str):
            lang = lang.lower()
            
        if lang in ['ta', 'tamil']: lang_key = 'tamil'
        elif lang in ['te', 'telugu']: lang_key = 'telugu'
        elif lang in ['ml', 'malayalam']: lang_key = 'malayalam'
        else:
            lang_key = None
            
        if not lang_key:
            if idx == 0:
                print("First sample keys:", list(sample.keys()))
                if 'language' in sample: print("Language field:", sample['language'])
            continue
            
        if target_langs[lang_key]['count'] >= target_langs[lang_key]['limit']:
            if all(v['count'] >= v['limit'] for v in target_langs.values()):
                break
            continue
            
        audio_data = sample.get('audio')
        if not audio_data:
            continue
            
        text = sample.get('transcript') or sample.get('sentence') or sample.get('text')
        
        prefix = target_langs[lang_key]['prefix']
        count = target_langs[lang_key]['count']
        filename = f"{prefix}_{count}.wav"
        out_path = os.path.join(out_dir, filename)
        
        # In decode=False, audio_data is a dict with 'bytes' or 'path'
        if 'bytes' in audio_data and audio_data['bytes']:
            with open(out_path, 'wb') as af:
                af.write(audio_data['bytes'])
        elif 'path' in audio_data and audio_data['path']:
            import shutil
            shutil.copy2(audio_data['path'], out_path)
        else:
            print("No audio bytes/path found.")
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
