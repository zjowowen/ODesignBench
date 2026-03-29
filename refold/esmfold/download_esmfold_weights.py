#!/usr/bin/env python3
"""
Download script for ESMFold v1 weights from HuggingFace.
Run this on a machine with network access to HuggingFace.

Usage:
    python download_esmfold_weights.py [--output_dir PATH]

Output directory defaults to ./weights in the current directory.
"""

import argparse
import os
from huggingface_hub import snapshot_download

def download_esmfold(output_dir: str = "./weights"):
    """Download ESMFold v1 model weights from HuggingFace."""
    
    print(f"Downloading ESMFold v1 model to: {os.path.abspath(output_dir)}")
    print("This may take a while as the model is several GB in size...")
    
    try:
        local_dir = snapshot_download(
            repo_id="facebook/esmfold_v1",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"\nDownload complete! Model saved to: {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                rel_path = os.path.relpath(path, local_dir)
                print(f"  {rel_path} ({size_str})")
                
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ESMFold v1 weights from HuggingFace")
    parser.add_argument("--output_dir", "-o", default="./weights", help="Output directory for weights")
    args = parser.parse_args()
    
    download_esmfold(args.output_dir)
