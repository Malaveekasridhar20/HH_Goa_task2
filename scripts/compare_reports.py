import json
import os

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
    
    with open(os.path.join(data_dir, 'chunking_report.json'), 'r', encoding='utf-8') as f:
        hindi = json.load(f)
        
    with open(os.path.join(data_dir, 'english_chunking_report.json'), 'r', encoding='utf-8') as f:
        english = json.load(f)
        
    report = {
        "comparison": [
            {
                "Language": "Hindi",
                "Records": hindi["records_processed"],
                "Passages": hindi["passages_extracted"],
                "Chunks": hindi["statistics"]["chunks_generated"],
                "Avg Chunk Length": hindi["statistics"]["lengths"]["avg"],
                "Whole": hindi["statistics"]["strategies"]["whole_passage"],
                "Semantic": hindi["statistics"]["strategies"]["semantic"],
                "Sliding": hindi["statistics"]["strategies"]["sliding_window"]
            },
            {
                "Language": "English",
                "Records": english["records_processed"],
                "Passages": english["passages_extracted"],
                "Chunks": english["statistics"]["chunks_generated"],
                "Avg Chunk Length": english["statistics"]["lengths"]["avg"],
                "Whole": english["statistics"]["strategies"]["whole_passage"],
                "Semantic": english["statistics"]["strategies"]["semantic"],
                "Sliding": english["statistics"]["strategies"]["sliding_window"]
            }
        ]
    }
    
    out_path = os.path.join(data_dir, 'multilingual_chunking_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print(f"Generated {out_path}")

if __name__ == "__main__":
    main()
