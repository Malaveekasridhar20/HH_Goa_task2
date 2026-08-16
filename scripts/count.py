import json

def analyze(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    generated = 0
    grounded = 0
    unsupported = 0
    unclear = 0
    
    for item in data:
        ans = item.get("answer", "")
        chunks = item.get("source_chunk_ids", [])
        
        if not ans.startswith("I don't have enough information"):
            generated += 1
            if len(chunks) > 0:
                grounded += 1
            else:
                unclear += 1
        else:
            unsupported += 1
            
    print(f"File: {file_path}")
    print(f"Total: {total}")
    print(f"Generated: {generated}")
    print(f"Grounded: {grounded}")
    print(f"Unsupported: {unsupported}")
    print(f"Unclear: {unclear}")
    print("-" * 20)

analyze("data/indexes/english/generation_audit.json")
analyze("data/indexes/hindi/generation_audit.json")
