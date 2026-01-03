#!/usr/bin/env python3
"""
Model Download Script for AccessVision
Downloads required YOLO model files
"""

import os
import urllib.request
import sys

def download_file(url, filename):
    """Download file with progress"""
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            sys.stdout.write(f"\rDownloading {filename}: {percent}%")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, progress)
        print(f"\n✓ {filename} downloaded")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        return False

def main():
    print("AccessVision Model Downloader")
    print("=" * 30)
    
    # Create models directory
    if not os.path.exists("models"):
        os.makedirs("models")
    
    files = [
        {
            "url": "https://pjreddie.com/media/files/yolov3.weights",
            "path": "models/yolov3.weights"
        },
        {
            "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "path": "models/yolov3.cfg"
        }
    ]
    
    for file_info in files:
        if os.path.exists(file_info["path"]):
            print(f"✓ {file_info['path']} already exists")
            continue
        
        success = download_file(file_info["url"], file_info["path"])
        if not success:
            return False
    
    print("\n✓ All model files ready!")
    print("You can now run: python access_vision.py")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("Press Enter to exit...")
        sys.exit(1)