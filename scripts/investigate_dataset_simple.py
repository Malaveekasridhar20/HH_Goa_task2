import traceback
from datasets import get_dataset_config_names, load_dataset_builder, load_dataset

dataset_name = "ai4bharat/MSMARCO-XI"

try:
    print(f"Investigating {dataset_name}...")
    configs = get_dataset_config_names(dataset_name)
    print(f"Configurations: {configs}")
    
    for conf in configs:
        print(f"\n--- Config: {conf} ---")
        builder = load_dataset_builder(dataset_name, conf)
        info = builder.info
        print(f"Features: {info.features}")
        
except Exception as e:
    print("Error:")
    traceback.print_exc()
