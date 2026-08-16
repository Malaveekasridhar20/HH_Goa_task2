import os
import shutil
import pyttsx3

out_dir = os.path.join(os.path.dirname(__file__), "../data/test_audio")

# Copy en_0.wav to en_2, en_3, en_4
for i in range(2, 5):
    shutil.copy(os.path.join(out_dir, "en_0.wav"), os.path.join(out_dir, f"en_{i}.wav"))
    shutil.copy(os.path.join(out_dir, "en_0.wav.txt"), os.path.join(out_dir, f"en_{i}.wav.txt"))

# Generate one Hindi wav
engine = pyttsx3.init()
hi_text = "कॉर्पोरेशन क्या है?"
path = os.path.join(out_dir, "hi_0.wav")
engine.save_to_file(hi_text, path)
engine.runAndWait()
with open(path + ".txt", "w", encoding="utf-8") as f:
    f.write(hi_text)

# Copy hi_0.wav to hi_1..4
for i in range(1, 5):
    shutil.copy(path, os.path.join(out_dir, f"hi_{i}.wav"))
    shutil.copy(path + ".txt", os.path.join(out_dir, f"hi_{i}.wav.txt"))

print("Done generating 5 EN and 5 HI samples")
