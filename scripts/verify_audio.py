import os
import json
import soundfile as sf

out_dir = 'data/human_audio'
manifest_path = os.path.join(out_dir, 'manifest.json')
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

print(f'Manifest has {len(manifest)} entries.')

for entry in manifest:
    path = os.path.join(out_dir, entry['filename'])
    assert os.path.exists(path), f'Missing {path}'
    size = os.path.getsize(path)
    assert size > 0, f'Zero size {path}'
    info = sf.info(path)
    print(f"{entry['filename']}: {size} bytes, {info.duration:.2f}s, {info.samplerate}Hz, {info.channels}ch, text={bool(entry.get('original transcript'))}")
