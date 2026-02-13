from datasets import load_dataset
import sys
import os

with open("dataset_info.txt", "w") as f:
    try:
        f.write("Attempting to load dataset...\n")
        ds = load_dataset("saakshigupta/deepfake-detection-dataset-v3", streaming=True)
        f.write("Dataset loaded successfully\n")
        f.write(str(ds) + "\n\n")
        
        if 'train' in ds:
            f.write("First example in train split:\n")
            example = next(iter(ds['train']))
            for k, v in example.items():
                f.write(f"Key: {k}, Type: {type(v)}\n")
                if hasattr(v, 'size'):
                    f.write(f"Size: {v.size}\n")
                if isinstance(v, (str, int, float)):
                    f.write(f"Value: {v}\n")
        else:
            f.write("'train' split not found.\n")
            
    except Exception as e:
        f.write(f"Error loading dataset: {e}\n")
