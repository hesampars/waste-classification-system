#!/usr/bin/env python3
"""
Script to preprocess waste classification datasets.
"""

import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_utils import WasteDatasetPreprocessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess waste classification datasets")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help="Directory containing the datasets")
    parser.add_argument("--output-dir", type=str, default=os.path.join(config.DATA_DIR, "processed"),
                        help="Directory to save processed data")
    return parser.parse_args()

def main():
    """Main function to preprocess all datasets."""
    args = parse_args()
    
    # Create preprocessor
    preprocessor = WasteDatasetPreprocessor(args.data_dir)
    
    # Process all datasets
    splits = preprocessor.process_all_datasets()
    
    if splits:
        print("Dataset preprocessing completed successfully")
        
        # Print class distribution
        class_counts = {}
        for item in splits["train"]:
            class_name = item["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nClass distribution in training set:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} images")
    else:
        print("Dataset preprocessing failed")

if __name__ == "__main__":
    main()
