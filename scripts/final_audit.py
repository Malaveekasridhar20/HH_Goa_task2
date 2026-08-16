import os, json, re, glob, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. .gitignore
with open('.gitignore', 'r') as f:
    gi = f.read()
print('=== .gitignore ===')
print('.env ignored:', '.env' in gi)

# 2. Git-tracked .env
result = subprocess.run(['git', 'ls-files', 'backend/.env', '.env'], capture_output=True, text=True)
print('Git-tracked .env:', repr(result.stdout.strip()) or 'none (GOOD)')

# 3. Secret scan in source + frontend
patterns = [r'["\'][a-zA-Z0-9_\-]{32,}["\']']
skip_paths = ['__pycache__', '.pyc', 'venv', 'node_modules']
found = []
search_files = (
    glob.glob('backend/**/*.py', recursive=True) +
    glob.glob('frontend/**/*.js', recursive=True) +
    glob.glob('frontend/**/*.html', recursive=True) +
    glob.glob('*.md')
)
for fpath in search_files:
    if any(s in fpath for s in skip_paths):
        continue
    try:
        txt = open(fpath, encoding='utf-8', errors='ignore').read()
        for pat in patterns:
            for m in re.finditer(pat, txt):
                val = m.group(0).strip('"\'')
                # Skip obvious non-secrets
                if any(x in val for x in ['multilingual', 'mmarco', 'intfloat', 'cross-encoder',
                                           'mMiniLM', 'saaras', 'passthrough', 'application',
                                           'deployment', 'extraction', 'retrieving', 'generated',
                                           'description', 'localhost', 'operations']):
                    continue
                line = txt[max(0,m.start()-50):m.end()+20]
                if 'os.getenv' in line or 'os.environ' in line or '#' in line[:m.start()-txt.rfind('\n',0,m.start())]:
                    continue
                found.append((fpath, val[:60]))
    except Exception as e:
        pass
print('\n=== Secret scan ===')
if found:
    for f,v in found[:10]:
        print(f'  FOUND in {f}: {v}')
else:
    print('  No hardcoded secrets found')

# 4. requirements.txt
print('\n=== requirements.txt ===')
exists = os.path.exists('requirements.txt')
print('Exists:', exists)
if exists:
    lines = open('requirements.txt').readlines()
    key_deps = ['fastapi', 'uvicorn', 'faiss', 'sentence-transformers', 'rank_bm25', 'numpy', 'pydantic', 'python-dotenv']
    for dep in key_deps:
        found_dep = any(dep.lower() in l.lower() for l in lines)
        print(f'  {dep}: {"YES" if found_dep else "MISSING"}')

# 5. Index files present
print('\n=== Index files ===')
for lang in ['english', 'hindi', 'tamil', 'telugu', 'malayalam']:
    idir = f'data/indexes/{lang}'
    has_faiss = os.path.exists(f'{idir}/faiss.index')
    has_bm25 = os.path.exists(f'{idir}/bm25.pkl')
    has_meta = os.path.exists(f'{idir}/metadata.jsonl')
    print(f'  {lang}: faiss={has_faiss} bm25={has_bm25} meta={has_meta}')

# 6. Frontend files
print('\n=== Frontend ===')
frontend_dir = 'frontend'
has_index = os.path.exists(f'{frontend_dir}/index.html')
has_js = len(glob.glob(f'{frontend_dir}/**/*.js', recursive=True)) > 0
print(f'  index.html: {has_index}')
print(f'  JS files: {has_js}')

# 7. Chunking strategies
print('\n=== Chunking files ===')
for f in ['backend/app/chunking/adaptive.py', 'backend/app/chunking/semantic.py',
          'backend/app/chunking/fixed_window.py']:
    print(f'  {f}: {os.path.exists(f)}')

# 8. STT
print('\n=== STT ===')
stt_file = 'backend/app/stt/sarvam.py'
print(f'  sarvam.py: {os.path.exists(stt_file)}')
if os.path.exists(stt_file):
    txt = open(stt_file, encoding='utf-8').read()
    print(f'  Uses os.getenv for API key: {"os.getenv" in txt}')
    print(f'  Has retry logic: {"retry" in txt.lower() or "Retry" in txt}')

# 9. Pipeline/orchestration
print('\n=== Orchestration ===')
pipe_file = 'backend/app/orchestration/pipeline.py'
print(f'  pipeline.py: {os.path.exists(pipe_file)}')
if os.path.exists(pipe_file):
    txt = open(pipe_file, encoding='utf-8').read()
    print(f'  Pydantic models: {"VoiceRAGRequest" in txt}')
    print(f'  Error recovery: {"try" in txt and "except" in txt}')
    print(f'  Language routing: {"language_hint" in txt or "lang" in txt.lower()}')

# 10. Guardrails
print('\n=== Guardrails ===')
gen_file = 'backend/app/generation/extractive_generator.py'
if os.path.exists(gen_file):
    txt = open(gen_file, encoding='utf-8').read()
    print(f'  Relevance threshold: {"relevance_threshold" in txt}')
    refusal_present = "I don't have enough information" in txt
    print(f'  Safe refusal text: {refusal_present}')

# 11. .env example
print('\n=== .env status ===')
print(f'  backend/.env exists: {os.path.exists("backend/.env")}')
print(f'  .env.example exists: {os.path.exists("backend/.env.example") or os.path.exists(".env.example")}')
print(f'  README documents env vars: {os.path.exists("README.md")}')
