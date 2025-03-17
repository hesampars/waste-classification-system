#!/usr/bin/env python3
"""
Script to download and prepare waste classification datasets.
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
import shutil
import subprocess
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download waste classification datasets")
    parser.add_argument("--output-dir", type=str, default=config.DATA_DIR,
                        help="Directory to save datasets")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["trashnet", "taco", "waste-pictures", "mju-waste", "open-images"],
                        help="Datasets to download")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip datasets that already exist")
    return parser.parse_args()

def download_file(url, output_path, chunk_size=8192):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the file
        chunk_size: Download chunk size
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def extract_archive(archive_path, output_dir):
    """
    Extract an archive file.
    
    Args:
        archive_path: Path to the archive file
        output_dir: Directory to extract to
    """
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")

def download_trashnet(output_dir):
    """
    Download and prepare TrashNet dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    dataset_dir = os.path.join(output_dir, "trashnet")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(dataset_dir, "dataset-resized")) or os.path.exists(os.path.join(dataset_dir, "data")):
        print("TrashNet dataset already exists")
        return dataset_dir
    
    # Check if manually downloaded zip exists
    manual_zip = os.path.join(output_dir, "trashnet-master.zip")
    if os.path.exists(manual_zip):
        print(f"Using manually downloaded TrashNet zip: {manual_zip}")
        extract_archive(manual_zip, dataset_dir)
        return dataset_dir
    
    # Download from GitHub
    url = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
    zip_path = os.path.join(dataset_dir, "trashnet-master.zip")
    
    print(f"Downloading TrashNet dataset from: {url}")
    download_file(url, zip_path)
    
    # Extract archive
    print("Extracting TrashNet dataset...")
    extract_archive(zip_path, dataset_dir)
    
    print("TrashNet dataset downloaded successfully")
    return dataset_dir

def download_taco(output_dir):
    """
    Download and prepare TACO dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    dataset_dir = os.path.join(output_dir, "taco")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(dataset_dir, "data")) or os.path.exists(os.path.join(dataset_dir, "annotations.json")):
        print("TACO dataset already exists")
        return dataset_dir
    
    # Check if manually downloaded zip exists
    manual_zip = os.path.join(output_dir, "TACO-master.zip")
    if os.path.exists(manual_zip):
        print(f"Using manually downloaded TACO zip: {manual_zip}")
        extract_archive(manual_zip, dataset_dir)
        return dataset_dir
    
    # Download from GitHub
    url = "https://github.com/pedropro/TACO/archive/refs/heads/master.zip"
    zip_path = os.path.join(dataset_dir, "TACO-master.zip")
    
    print(f"Downloading TACO dataset from: {url}")
    download_file(url, zip_path)
    
    # Extract archive
    print("Extracting TACO dataset...")
    extract_archive(zip_path, dataset_dir)
    
    print("TACO dataset downloaded successfully")
    return dataset_dir

def download_waste_pictures(output_dir):
    """
    Download and prepare Waste-Pictures dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    dataset_dir = os.path.join(output_dir, "waste-pictures")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(dataset_dir, "plastic")) or os.path.exists(os.path.join(dataset_dir, "paper")):
        print("Waste-Pictures dataset already exists")
        return dataset_dir
    
    # Check if manually downloaded zip exists
    manual_zip = os.path.join(output_dir, "waste-pictures.zip")
    if os.path.exists(manual_zip):
        print(f"Using manually downloaded Waste-Pictures zip: {manual_zip}")
        extract_archive(manual_zip, dataset_dir)
        return dataset_dir
    
    # Download from Kaggle
    print("Waste-Pictures dataset needs to be downloaded manually from Kaggle")
    print("Please download from: https://www.kaggle.com/datasets/wangziang/waste-pictures")
    print(f"Save the zip file to: {os.path.join(output_dir, 'waste-pictures.zip')}")
    print("Then run this script again")
    
    return dataset_dir

def download_mju_waste(output_dir):
    """
    Download and prepare MJU-Waste dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    dataset_dir = os.path.join(output_dir, "mju-waste")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(dataset_dir, "battery")) or os.path.exists(os.path.join(dataset_dir, "biological")):
        print("MJU-Waste dataset already exists")
        return dataset_dir
    
    # Check if manually downloaded zip exists
    manual_zip = os.path.join(output_dir, "MJU-Waste.zip")
    if os.path.exists(manual_zip):
        print(f"Using manually downloaded MJU-Waste zip: {manual_zip}")
        extract_archive(manual_zip, dataset_dir)
        return dataset_dir
    
    # Download from GitHub
    url = "https://github.com/realwecan/mju-waste/archive/refs/heads/master.zip"
    zip_path = os.path.join(dataset_dir, "mju-waste-master.zip")
    
    print(f"Downloading MJU-Waste dataset from: {url}")
    download_file(url, zip_path)
    
    # Extract archive
    print("Extracting MJU-Waste dataset...")
    extract_archive(zip_path, dataset_dir)
    
    print("MJU-Waste dataset downloaded successfully")
    return dataset_dir

def download_open_images(output_dir):
    """
    Download and prepare waste-related images from Google Open Images.
    
    Args:
        output_dir: Directory to save the dataset
    """
    dataset_dir = os.path.join(output_dir, "open-images")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if OIDv4 toolkit is installed
    oidv4_dir = os.path.join(dataset_dir, "OIDv4_ToolKit")
    if not os.path.exists(oidv4_dir):
        print("Downloading OIDv4 toolkit...")
        subprocess.check_call(["git", "clone", "https://github.com/EscVM/OIDv4_ToolKit.git", oidv4_dir])
        
        # Install requirements
        requirements_path = os.path.join(oidv4_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print("Installing OIDv4 toolkit requirements...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    
    # Define waste-related classes to download
    waste_classes = [
        "Bottle", "Tin can", "Plastic bag", "Cardboard", "Paper",
        "Glass", "Mobile phone", "Computer", "Food", "Clothing"
    ]
    
    # Check if dataset already exists
    download_dir = os.path.join(oidv4_dir, "OID", "Dataset")
    if os.path.exists(download_dir):
        class_dirs = []
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(download_dir, split)
            if os.path.exists(split_dir):
                class_dirs.extend([os.path.join(split_dir, c) for c in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, c))])
        
        if class_dirs:
            print(f"Open Images dataset already exists with {len(class_dirs)} class directories")
            return dataset_dir
    
    # Download each class
    for class_name in waste_classes:
        print(f"Downloading Open Images for class: {class_name}")
        
        # Use OIDv4 toolkit to download images
        subprocess.check_call([
            "python", os.path.join(oidv4_dir, "main.py"), "downloader",
            "--classes", class_name,
            "--type", "train",
            "--limit", "500",
            "--yes",
            "--dataset", dataset_dir
        ])
    
    print("Open Images dataset downloaded successfully")
    return dataset_dir

def main():
    """Main function to download all datasets."""
    args = parse_args()
    
    # Download selected datasets
    dataset_paths = {}
    
    if "trashnet" in args.datasets:
        dataset_paths["trashnet"] = download_trashnet(args.output_dir)
    
    if "taco" in args.datasets:
        dataset_paths["taco"] = download_taco(args.output_dir)
    
    if "waste-pictures" in args.datasets:
        dataset_paths["waste-pictures"] = download_waste_pictures(args.output_dir)
    
    if "mju-waste" in args.datasets:
        dataset_paths["mju-waste"] = download_mju_waste(args.output_dir)
    
    if "open-images" in args.datasets:
        dataset_paths["open-images"] = download_open_images(args.output_dir)
    
    # Save dataset paths
    paths_file = os.path.join(args.output_dir, "dataset_paths.json")
    with open(paths_file, 'w') as f:
        json.dump(dataset_paths, f, indent=2)
    
    print(f"Dataset paths saved to: {paths_file}")
    print("All datasets downloaded successfully")

if __name__ == "__main__":
    main()
