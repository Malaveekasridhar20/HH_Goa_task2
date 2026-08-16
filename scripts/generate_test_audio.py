import os
import json
import pyttsx3
import wave
import struct

def generate():
    # Make sure we have a directory for them
    out_dir = os.path.join(os.path.dirname(__file__), "../data/test_audio")
    os.makedirs(out_dir, exist_ok=True)
    
    en_queries, hi_queries = [], []
    data_path = os.path.join(os.path.dirname(__file__), "../data/processed/hinval_500.jsonl")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            en_queries.append(row.get("Eng_Query"))
            hi_queries.append(row.get("query"))
            
    en_queries = en_queries[:5]
    hi_queries = hi_queries[:5]
    
    engine = pyttsx3.init()
    
    # We will generate them and then re-write them as proper 16000Hz PCM
    # pyttsx3 might output 22050 or 44100 depending on the system voice.
    # To fix this, we'll just save them, then use a quick pure Python wave converter
    # if necessary, or just rely on the API to resample if it accepts it.
    # We will declare sample_rate in the API as the actual file's sample rate.
    
    for i, q in enumerate(en_queries):
        path = os.path.join(out_dir, f"en_{i}.wav")
        engine.save_to_file(q, path)
        engine.runAndWait()
        print(f"Generated {path}")
        
        # Write metadata file alongside so eval script knows the text
        with open(path + ".txt", "w", encoding="utf-8") as tf:
            tf.write(q)

    for i, q in enumerate(hi_queries):
        path = os.path.join(out_dir, f"hi_{i}.wav")
        engine.save_to_file(q, path)
        engine.runAndWait()
        print(f"Generated {path}")
        
        with open(path + ".txt", "w", encoding="utf-8") as tf:
            tf.write(q)

if __name__ == "__main__":
    generate()
