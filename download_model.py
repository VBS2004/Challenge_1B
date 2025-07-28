#!/usr/bin/env python3
"""
Script to download the sentence transformer model for offline use.
Run this script before building the Docker container.
"""

import os
from sentence_transformers import SentenceTransformer

def download_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path = "./models/all-MiniLM-L6-v2"
    
    print(f"Downloading model: {model_name}")
    print(f"Saving to: {local_model_path}")
    
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    # Download and save model
    model = SentenceTransformer(model_name)
    model.save(local_model_path)
    
    print("✅ Model downloaded successfully!")
    print(f"Model files saved to: {local_model_path}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(local_model_path):
        for file in files:
            print(f"  {os.path.join(root, file)}")

if __name__ == "__main__":
    download_model()