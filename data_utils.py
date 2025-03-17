#!/usr/bin/env python3
"""
Data utilities module for waste classification system.
"""

import os
import sys
import json
import random
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WasteDatasetPreprocessor:
    """Preprocessor for waste classification datasets."""
    
    def __init__(self, data_dir):
        """
        Initialize the dataset preprocessor.
        
        Args:
            data_dir: Directory containing the datasets
        """
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Class mapping for standardization
        self.class_mapping = {
            # TrashNet mapping
            "glass": "glass",
            "paper": "paper",
            "cardboard": "cardboard",
            "plastic": "plastic",
            "metal": "metal",
            "trash": "trash",
            
            # TACO mapping
            "Plastic bottle": "plastic",
            "Bottle cap": "plastic",
            "Plastic bag & wrapper": "plastic",
            "Carton": "cardboard",
            "Paper": "paper",
            "Aluminium foil": "metal",
            "Metal can": "metal",
            "Glass bottle": "glass",
            "Plastic container": "plastic",
            "Plastic utensils": "plastic",
            "Pop tab": "metal",
            "Straw": "plastic",
            "Paper cup": "paper",
            "Plastic cup": "plastic",
            "Plastic lid": "plastic",
            "Cigarette": "trash",
            "Other plastic": "plastic",
            "Other metal": "metal",
            "Other glass": "glass",
            "Other paper": "paper",
            "Unlabeled litter": "trash",
            
            # MJU-Waste mapping
            "battery": "e-waste",
            "biological": "organic",
            "brown-glass": "glass",
            "cardboard": "cardboard",
            "clothes": "textile",
            "green-glass": "glass",
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "shoes": "textile",
            "trash": "trash",
            "white-glass": "glass",
            
            # Waste-Pictures mapping
            "battery": "e-waste",
            "biological": "organic",
            "clothes": "textile",
            "e-waste": "e-waste",
            "glass": "glass",
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "textile": "textile",
            "trash": "trash",
            "mixed": "mixed",
            
            # Open Images mapping
            "Bottle": "plastic",
            "Tin can": "metal",
            "Plastic bag": "plastic",
            "Cardboard": "cardboard",
            "Paper": "paper",
            "Glass": "glass",
            "Mobile phone": "e-waste",
            "Computer": "e-waste",
            "Food": "organic",
            "Clothing": "textile"
        }
    
    def preprocess_trashnet(self):
        """
        Preprocess TrashNet dataset.
        
        Returns:
            List of processed image metadata
        """
        dataset_dir = os.path.join(self.data_dir, "trashnet")
        if not os.path.exists(dataset_dir):
            print(f"TrashNet dataset not found at: {dataset_dir}")
            return []
        
        # Find the dataset folder (might be nested)
        data_folder = None
        for root, dirs, files in os.walk(dataset_dir):
            if "glass" in dirs and "paper" in dirs and "cardboard" in dirs:
                data_folder = root
                break
        
        if not data_folder:
            print("Could not find TrashNet data folder structure")
            return []
        
        print(f"Processing TrashNet dataset from: {data_folder}")
        
        metadata = []
        
        # Process each class folder
        for class_name in os.listdir(data_folder):
            class_dir = os.path.join(data_folder, class_name)
            
            if not os.path.isdir(class_dir):
                continue
            
            # Map class name
            mapped_class = self.class_mapping.get(class_name, "trash")
            
            # Process images in this class
            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                src_path = os.path.join(class_dir, filename)
                dst_path = os.path.join(self.output_dir, f"trashnet_{filename}")
                
                # Copy file to output directory
                shutil.copy(src_path, dst_path)
                
                # Add metadata
                metadata.append({
                    "file": os.path.basename(dst_path),
                    "source": "trashnet",
                    "original_class": class_name,
                    "class": mapped_class
                })
        
        print(f"Processed {len(metadata)} images from TrashNet")
        return metadata
    
    def preprocess_taco(self):
        """
        Preprocess TACO dataset.
        
        Returns:
            List of processed image metadata
        """
        dataset_dir = os.path.join(self.data_dir, "taco")
        if not os.path.exists(dataset_dir):
            print(f"TACO dataset not found at: {dataset_dir}")
            return []
        
        # Find the annotations file
        annotations_file = None
        for root, dirs, files in os.walk(dataset_dir):
            if "annotations.json" in files:
                annotations_file = os.path.join(root, "annotations.json")
                break
        
        if not annotations_file:
            print("Could not find TACO annotations.json file")
            return []
        
        # Load annotations
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"Error loading TACO annotations: {str(e)}")
            return []
        
        print(f"Processing TACO dataset from: {os.path.dirname(annotations_file)}")
        
        metadata = []
        
        # Get image directory
        image_dir = os.path.join(os.path.dirname(annotations_file), "data")
        if not os.path.exists(image_dir):
            image_dir = os.path.dirname(annotations_file)
        
        # Process images
        for image_info in annotations["images"]:
            image_id = image_info["id"]
            filename = image_info["file_name"]
            
            # Find annotations for this image
            image_annotations = [a for a in annotations["annotations"] if a["image_id"] == image_id]
            
            if not image_annotations:
                continue
            
            # Get most common category
            category_counts = {}
            for ann in image_annotations:
                category_id = ann["category_id"]
                category_info = next((c for c in annotations["categories"] if c["id"] == category_id), None)
                
                if category_info:
                    category_name = category_info["name"]
                    category_counts[category_name] = category_counts.get(category_name, 0) + 1
            
            if not category_counts:
                continue
            
            # Get most common category
            original_class = max(category_counts.items(), key=lambda x: x[1])[0]
            
            # Map class name
            mapped_class = self.class_mapping.get(original_class, "trash")
            
            # Source and destination paths
            src_path = os.path.join(image_dir, filename)
            dst_path = os.path.join(self.output_dir, f"taco_{os.path.basename(filename)}")
            
            # Check if source file exists
            if not os.path.exists(src_path):
                continue
            
            # Copy file to output directory
            shutil.copy(src_path, dst_path)
            
            # Add metadata
            metadata.append({
                "file": os.path.basename(dst_path),
                "source": "taco",
                "original_class": original_class,
                "class": mapped_class
            })
        
        print(f"Processed {len(metadata)} images from TACO")
        return metadata
    
    def preprocess_mju_waste(self):
        """
        Preprocess MJU-Waste dataset.
        
        Returns:
            List of processed image metadata
        """
        dataset_dir = os.path.join(self.data_dir, "mju-waste")
        if not os.path.exists(dataset_dir):
            print(f"MJU-Waste dataset not found at: {dataset_dir}")
            return []
        
        # Find the dataset folder (might be nested)
        data_folder = None
        for root, dirs, files in os.walk(dataset_dir):
            if "battery" in dirs or "biological" in dirs or "cardboard" in dirs:
                data_folder = root
                break
        
        if not data_folder:
            print("Could not find MJU-Waste data folder structure")
            return []
        
        print(f"Processing MJU-Waste dataset from: {data_folder}")
        
        metadata = []
        
        # Process each class folder
        for class_name in os.listdir(data_folder):
            class_dir = os.path.join(data_folder, class_name)
            
            if not os.path.isdir(class_dir):
                continue
            
            # Map class name
            mapped_class = self.class_mapping.get(class_name, "trash")
            
            # Process images in this class
            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                src_path = os.path.join(class_dir, filename)
                dst_path = os.path.join(self.output_dir, f"mju_{filename}")
                
                # Copy file to output directory
                shutil.copy(src_path, dst_path)
                
                # Add metadata
                metadata.append({
                    "file": os.path.basename(dst_path),
                    "source": "mju-waste",
                    "original_class": class_name,
                    "class": mapped_class
                })
        
        print(f"Processed {len(metadata)} images from MJU-Waste")
        return metadata
    
    def preprocess_waste_pictures(self):
        """
        Preprocess Waste-Pictures dataset.
        
        Returns:
            List of processed image metadata
        """
        dataset_dir = os.path.join(self.data_dir, "waste-pictures")
        if not os.path.exists(dataset_dir):
            print(f"Waste-Pictures dataset not found at: {dataset_dir}")
            return []
        
        # Find the dataset folder (might be nested)
        data_folder = None
        for root, dirs, files in os.walk(dataset_dir):
            if "plastic" in dirs or "paper" in dirs or "metal" in dirs:
                data_folder = root
                break
        
        if not data_folder:
            print("Could not find Waste-Pictures data folder structure")
            return []
        
        print(f"Processing Waste-Pictures dataset from: {data_folder}")
        
        metadata = []
        
        # Process each class folder
        for class_name in os.listdir(data_folder):
            class_dir = os.path.join(data_folder, class_name)
            
            if not os.path.isdir(class_dir):
                continue
            
            # Map class name
            mapped_class = self.class_mapping.get(class_name, "trash")
            
            # Process images in this class
            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                src_path = os.path.join(class_dir, filename)
                dst_path = os.path.join(self.output_dir, f"waste_{filename}")
                
                # Copy file to output directory
                shutil.copy(src_path, dst_path)
                
                # Add metadata
                metadata.append({
                    "file": os.path.basename(dst_path),
                    "source": "waste-pictures",
                    "original_class": class_name,
                    "class": mapped_class
                })
        
        print(f"Processed {len(metadata)} images from Waste-Pictures")
        return metadata
    
    def preprocess_open_images(self):
        """
        Preprocess Open Images dataset.
        
        Returns:
            List of processed image metadata
        """
        dataset_dir = os.path.join(self.data_dir, "open-images")
        if not os.path.exists(dataset_dir):
            print(f"Open Images dataset not found at: {dataset_dir}")
            return []
        
        # Find the OIDv4 toolkit folder
        oidv4_dir = os.path.join(dataset_dir, "OIDv4_ToolKit")
        if not os.path.exists(oidv4_dir):
            print("Could not find OIDv4_ToolKit folder")
            return []
        
        # Find the downloaded images
        download_dir = os.path.join(oidv4_dir, "OID", "Dataset")
        if not os.path.exists(download_dir):
            print("Could not find downloaded Open Images dataset")
            return []
        
        print(f"Processing Open Images dataset from: {download_dir}")
        
        metadata = []
        
        # Process each class folder
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(download_dir, split)
            
            if not os.path.exists(split_dir):
                continue
            
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                
                if not os.path.isdir(class_dir):
                    continue
                
                # Map class name
                mapped_class = self.class_mapping.get(class_name, "trash")
                
                # Process images in this class
                for filename in os.listdir(class_dir):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    
                    src_path = os.path.join(class_dir, filename)
                    dst_path = os.path.join(self.output_dir, f"openimages_{filename}")
                    
                    # Copy file to output directory
                    shutil.copy(src_path, dst_path)
                    
                    # Add metadata
                    metadata.append({
                        "file": os.path.basename(dst_path),
                        "source": "open-images",
                        "original_class": class_name,
                        "class": mapped_class
                    })
        
        print(f"Processed {len(metadata)} images from Open Images")
        return metadata
    
    def create_dataset_splits(self, metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Create train/val/test splits from metadata.
        
        Args:
            metadata: List of image metadata
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Shuffle metadata
        random.shuffle(metadata)
        
        # Group by <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>