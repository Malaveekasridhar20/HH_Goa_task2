import requests
import os

base_url = "http://127.0.0.1:8000"

def test_endpoint(file_path, lang):
    print(f"\n--- Testing {file_path} ({lang}) ---")
    if not os.path.exists(file_path):
        print("File does not exist!")
        return
        
    with open(file_path, "rb") as f:
        res = requests.post(f"{base_url}/api/voice/query", files={"audio": f}, data={"language": lang})
    
    print(f"Status: {res.status_code}")
    try:
        data = res.json()
        print(f"Success: {data.get('success')}")
        print(f"Transcript: {data.get('transcript')}")
        print(f"Answer: {data.get('answer')}")
        print(f"Sources: {data.get('source_chunk_ids')}")
        print(f"Error: {data.get('error')}")
    except Exception as e:
        print("Failed to parse JSON:", res.text)

print("Waiting for server to start...")
try:
    requests.get(f"{base_url}/health")
    print("Server is up!")
except Exception:
    print("Server not reachable yet.")
    exit(1)

test_endpoint("data/human_audio/en_0.wav", "en")
test_endpoint("data/human_audio/hi_0.wav", "hi")

# Create empty file
empty_file = "empty.wav"
with open(empty_file, "wb") as f:
    pass

test_endpoint(empty_file, "en")

print("\nAll tests completed.")
