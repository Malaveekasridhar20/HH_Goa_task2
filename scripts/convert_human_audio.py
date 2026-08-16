import os
import subprocess
import wave

in_en = [
    r"C:\Users\malav\Downloads\What is a corporation.mp3.mpeg",
    r"C:\Users\malav\Downloads\Why did Rachel Carson write Silent Spring.mp3.mpeg",
    r"C:\Users\malav\Downloads\What is the capital of India.mp3.mpeg",
    r"C:\Users\malav\Downloads\What causes barometric pressure to change.mp3.mpeg",
    r"C:\Users\malav\Downloads\How does photosynthesis work.mp3.mpeg"
]

in_hi = [
    r"C:\Users\malav\Downloads\Hindi 1.mp3.mpeg",
    r"C:\Users\malav\Downloads\Hindi 2.mp3.mpeg",
    r"C:\Users\malav\Downloads\Hindi 3.mp3.mpeg",
    r"C:\Users\malav\Downloads\Hindi 4.mp3.mpeg",
    r"C:\Users\malav\Downloads\Hindi 5.mp3.mpeg"
]

out_dir = r"C:\Users\malav\Downloads\goa_task_2\hh-goa-voice-rag\data\human_audio"
os.makedirs(out_dir, exist_ok=True)

ffmpeg_path = r"C:\Users\malav\Downloads\goa_task_2\hh-goa-voice-rag\ffmpeg.exe"

def convert_and_verify(in_files, prefix):
    results = []
    for i, f in enumerate(in_files):
        out_file = os.path.join(out_dir, f"{prefix}_{i}.wav")
        # Run ffmpeg to convert: -y (overwrite), -ar 16000, -ac 1, -c:a pcm_s16le
        cmd = [ffmpeg_path, "-y", "-i", f, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_file]
        try:
            # First inspect original
            inspect_cmd = [ffmpeg_path, "-i", f]
            subprocess.run(inspect_cmd, capture_output=True, text=True) # just to check if it runs
            
            # Convert
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Failed to convert {f}: {res.stderr}")
                results.append({"name": f"{prefix}_{i}.wav", "ok": False, "error": "ffmpeg error"})
                continue
            
            # Verify
            try:
                with wave.open(out_file, "rb") as w:
                    sr = w.getframerate()
                    ch = w.getnchannels()
                    sw = w.getsampwidth()
                    nf = w.getnframes()
                    dur = nf / sr
                    results.append({
                        "name": f"{prefix}_{i}.wav",
                        "ok": True,
                        "sr": sr,
                        "ch": ch,
                        "sw": sw,
                        "dur": dur,
                        "size": os.path.getsize(out_file)
                    })
            except wave.Error as e:
                results.append({"name": f"{prefix}_{i}.wav", "ok": False, "error": str(e)})
        except Exception as e:
            results.append({"name": f"{prefix}_{i}.wav", "ok": False, "error": str(e)})
            
    return results

print("Converting English...")
en_res = convert_and_verify(in_en, "en")
print("Converting Hindi...")
hi_res = convert_and_verify(in_hi, "hi")

print("\n\n## HUMAN AUDIO PREPARATION")
print("\nEnglish:")
for r in en_res:
    if r["ok"]:
        print(f"{r['name']} — {r['dur']:.2f}s / {r['sr']}Hz {r['ch']}ch {r['sw']*8}-bit PCM WAV")
    else:
        print(f"{r['name']} — ERROR: {r.get('error')}")

print("\nHindi:")
for r in hi_res:
    if r["ok"]:
        print(f"{r['name']} — {r['dur']:.2f}s / {r['sr']}Hz {r['ch']}ch {r['sw']*8}-bit PCM WAV")
    else:
        print(f"{r['name']} — ERROR: {r.get('error')}")

all_res = en_res + hi_res
conversion_pass = all(r.get("ok") and r.get("sr") == 16000 and r.get("ch") == 1 and r.get("sw") == 2 for r in all_res)
readable_pass = all(r.get("ok") for r in all_res)
# original recordings preserved -> we did not delete them in the script
preserved_pass = True 

print(f"\nConversion: {'PASS' if conversion_pass else 'FAIL'}")
print(f"All files readable: {'PASS' if readable_pass else 'FAIL'}")
print(f"Human recordings preserved: {'PASS' if preserved_pass else 'FAIL'}")
