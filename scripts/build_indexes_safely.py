import subprocess
import sys

def main():
    languages = ["tamil", "telugu", "malayalam"]
    for lang in languages:
        print(f"Building indexes for {lang} in a separate process...")
        cmd = [sys.executable, "-c", f"from scripts.build_multilingual_indexes import process_language; process_language('{lang}')"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors for {lang}:", result.stderr)
        if result.returncode != 0:
            print(f"Failed to build indexes for {lang}.")
            sys.exit(1)

if __name__ == "__main__":
    main()
