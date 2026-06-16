import json
import os
from datasets import load_dataset

def main():
    print("Loading deepset/prompt-injections dataset from Hugging Face...")
    ds = load_dataset("deepset/prompt-injections")
    os.makedirs("data/deepset_injections", exist_ok=True)
    
    # Save train split
    train_path = "data/deepset_injections/train.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(list(ds["train"]), f, indent=2)
        
    # Save test split
    test_path = "data/deepset_injections/test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(list(ds["test"]), f, indent=2)
        
    print(f"Downloaded and saved {len(ds['train'])} train, {len(ds['test'])} test samples.")

if __name__ == "__main__":
    main()
