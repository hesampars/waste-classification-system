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
            if "battery" in dirs or "biological" in dirs or "clothes" in dirs:
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
                dst_path = os.path.join(self.output_dir, f"waste_pictures_{filename}")
                
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
        download_dir = os.path.join(self.data_dir, "open-images")
        if not os.path.exists(download_dir):
            print(f"Open Images dataset not found at: {download_dir}")
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
    
    def process_all_datasets(self):
        """
        Process all available datasets and create train/val/test splits.
        
        Returns:
            Dictionary with train/val/test splits
        """
        # Process each dataset and collect metadata
        all_metadata = []
        
        # Process TrashNet
        trashnet_metadata = self.preprocess_trashnet()
        all_metadata.extend(trashnet_metadata)
        
        # Process TACO
        taco_metadata = self.preprocess_taco()
        all_metadata.extend(taco_metadata)
        
        # Process MJU-Waste
        mju_waste_metadata = self.preprocess_mju_waste()
        all_metadata.extend(mju_waste_metadata)
        
        # Process Waste-Pictures
        waste_pictures_metadata = self.preprocess_waste_pictures()
        all_metadata.extend(waste_pictures_metadata)
        
        # Process Open Images
        open_images_metadata = self.preprocess_open_images()
        all_metadata.extend(open_images_metadata)
        
        # Create dataset splits
        if all_metadata:
            print(f"Total processed images: {len(all_metadata)}")
            return self.create_dataset_splits(all_metadata)
        else:
            print("No images were processed. Check dataset paths.")
            return None
    
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
        
        # Group by class
        class_groups = {}
        for item in metadata:
            class_name = item["class"]
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(item)
        
        # Create stratified splits
        train_data = []
        val_data = []
        test_data = []
        
        for class_name, items in class_groups.items():
            # Calculate split sizes
            n_items = len(items)
            n_train = int(n_items * train_ratio)
            n_val = int(n_items * val_ratio)
            
            # Split data
            train_data.extend(items[:n_train])
            val_data.extend(items[n_train:n_train+n_val])
            test_data.extend(items[n_train+n_val:])
        
        # Shuffle again
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        # Create splits dictionary
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        
        # Save splits to file
        splits_file = os.path.join(self.output_dir, "splits.json")
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Created dataset splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        print(f"Saved splits to: {splits_file}")
        
        return splits

class WasteDataset(Dataset):
    """Dataset for waste classification."""
    
    def __init__(self, data_dir, split="train", transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the processed data
            split: Data split to use (train, val, test)
            transform: Transforms to apply to images
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load splits
        splits_file = os.path.join(data_dir, "splits.json")
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        self.data = splits[split]
        
        # Get class names
        self.classes = sorted(list(set(item["class"] for item in self.data)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"Loaded {len(self.data)} images for {split} split")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, label)
        """
        item = self.data[idx]
        image_file = os.path.join(self.data_dir, item["file"])
        
        # Load image
        image = Image.open(image_file).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[item["class"]]
        
        return image, label

def get_data_loaders(data_dir, batch_size=32, image_size=224, num_workers=4):
    """
    Get data loaders for training and validation.
    
    Args:
        data_dir: Directory containing the processed data
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with train, val, and test data loaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WasteDataset(data_dir, split="train", transform=train_transform)
    val_dataset = WasteDataset(data_dir, split="val", transform=val_transform)
    test_dataset = WasteDataset(data_dir, split="test", transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Return data loaders
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "classes": train_dataset.classes
    }
