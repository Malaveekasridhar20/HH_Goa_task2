import os, re

# Check .gitignore
with open('.gitignore', 'r') as f:
    gitignore = f.read()
if '.env' in gitignore:
    print('.env in .gitignore: PASS')
else:
    print('.env in .gitignore: FAIL')

# Check API keys are not hardcoded
hardcoded = []
for root, dirs, files in os.walk('.'):
    for skip in ['venv', '.git', 'node_modules', '__pycache__', 'data']:
        if skip in dirs:
            dirs.remove(skip)
    for fn in files:
        if not fn.endswith(('.py', '.js', '.ts')):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Look for suspicious patterns
            if re.search(r'SARVAM_API_KEY\s*=\s*["\'][A-Za-z0-9_-]{15,}["\']', content):
                hardcoded.append(path)
            if re.search(r'["\']sk-[A-Za-z0-9]{20,}["\']', content):
                hardcoded.append(path)
        except:
            pass

if hardcoded:
    for h in hardcoded:
        print(f'Hardcoded secret found: {h}')
else:
    print('No hardcoded API keys found: PASS')

# Check .env exists but is not tracked by git
import subprocess
result = subprocess.run(['git', 'ls-files', 'backend/.env'], capture_output=True, text=True)
if result.stdout.strip():
    print('.env is tracked by git: FAIL (security risk)')
else:
    print('.env not tracked by git: PASS')

print('backend/.env exists locally:', os.path.exists('backend/.env'))
