import sys, os, re, glob
sys.stdout.reconfigure(encoding='utf-8')

# requirements.txt locations
for p in ['requirements.txt', 'backend/requirements.txt']:
    if os.path.exists(p):
        lines = open(p).readlines()
        key_deps = ['fastapi', 'uvicorn', 'faiss', 'sentence', 'rank_bm25', 'numpy', 'pydantic', 'dotenv']
        print(f'{p} ({len(lines)} lines):')
        for dep in key_deps:
            found = any(dep.lower() in l.lower() for l in lines)
            print(f'  {dep}: {"YES" if found else "MISSING"}')
        print()

# sarvam retry
sarvam_txt = open('backend/app/stt/sarvam.py', encoding='utf-8').read()
print('=== sarvam.py retry ===')
for l in sarvam_txt.splitlines():
    if any(x in l.lower() for x in ['retry', 'attempt', 'except', 'raise', 'backoff']):
        print(' ', l.strip())

# frontend check
print()
print('=== frontend/index.html ===')
html = open('frontend/index.html', encoding='utf-8').read()
print('Has microphone API:', 'getUserMedia' in html or 'MediaRecorder' in html)
print('Has language select:', 'select' in html.lower() and ('tamil' in html.lower() or 'hindi' in html.lower()))
print('Has transcript display:', 'transcript' in html.lower())
print('Has answer display:', 'answer' in html.lower())
print('Has latency display:', 'latency' in html.lower() or 'ms' in html.lower())
print('Has error display:', 'error' in html.lower())
print('API key in HTML:', 'SARVAM_API_KEY' in html)
scripts_external = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html)
print('External scripts:', scripts_external)
inline_scripts = re.findall(r'<script[^>]*>(.{1,50})', html, re.DOTALL)
print('Has inline JS:', len(inline_scripts) > 0)

# .env.example
print()
print('=== .env.example ===')
for p in ['.env.example', 'backend/.env.example']:
    print(f'{p}: {os.path.exists(p)}')
