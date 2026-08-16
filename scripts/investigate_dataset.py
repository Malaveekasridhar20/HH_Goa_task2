from datasets import get_dataset_config_names, load_dataset_builder, load_dataset
import sys

dataset_name = "ai4bharat/MSMARCO-XI"

try:
    configs = get_dataset_config_names(dataset_name)
    print(f"Configurations: {configs}")
except Exception as e:
    print(f"Error getting configs: {e}")
    sys.exit(1)

for conf in configs:
    builder = load_dataset_builder(dataset_name, conf)
    print(f"\nConfig: {conf}")
    
    # Try to get info without downloading if possible
    info = builder.info
    print(f"Description: {info.description}")
    print(f"Features: {info.features}")
    
    # Let's see if we can get a sample
    # The dataset might be small enough to load a split, or we stream it
    try:
        ds = load_dataset(dataset_name, conf, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Sample data: {sample}")
    except Exception as e:
        print(f"Error streaming {conf}: {e}")
