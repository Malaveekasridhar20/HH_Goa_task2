import os
import subprocess
import glob

def run_tests():
    test_dir = os.path.join(os.path.dirname(__file__), "../backend/tests")
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    
    for test_file in test_files:
        print(f"Running {os.path.basename(test_file)}...")
        result = subprocess.run(
            ["pytest", test_file, "--tb=short", "-q"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"FAILED: {os.path.basename(test_file)}")
            print(result.stderr)
            failed += 1
        else:
            passed += 1

    print("================================")
    print(f"Test Files Summary: {passed} passed, {failed} failed")

if __name__ == "__main__":
    run_tests()
