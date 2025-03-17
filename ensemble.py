#!/usr/bin/env python3
"""
Ensemble classifier module for waste classification system.
"""

import torch
import numpy as np
from PIL import Image

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.classifier import WasteClassifier

class WasteEnsembleClassifier:
    def __init__(self, model_paths, weights, classes):
        """
        Initialize the ensemble classifier.
        
        Args:
            model_paths: Dictionary of model names to paths
            weights: Dictionary of model names to weights
            classes: List of class names
        """
        self.model_paths = model_paths
        self.weights = weights
        self.classes = classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize classifiers
        self.classifiers = {}
        
        for model_name, model_path in model_paths.items():
            # Extract architecture name
            if "convnext" in model_name:
                arch = "convnext_large"
                weight_key = "convnext"
            elif "efficientnet" in model_name:
                arch = "tf_efficientnetv2_l"
                weight_key = "efficientnet"
            elif "swin" in model_name:
                arch = "swin_large_patch4_window7_224"
                weight_key = "swin"
            else:
                print(f"Unknown model architecture: {model_name}")
                continue
            
            # Initialize classifier
            classifier = WasteClassifier(arch, model_path)
            
            # Add to classifiers
            if classifier.model is not None:
                self.classifiers[weight_key] = classifier
                print(f"Added {weight_key} to ensemble with weight {weights.get(weight_key, 1.0)}")
    
    def classify(self, image):
        """
        Classify an image using the ensemble.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Dictionary with classification results
        """
        if not self.classifiers:
            return {"success": False, "error": "No classifiers available"}
        
        try:
            # Convert numpy array to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Get predictions from each classifier
            all_predictions = {}
            
            for name, classifier in self.classifiers.items():
                result = classifier.classify(image)
                
                if result["success"]:
                    all_predictions[name] = result["predictions"]
            
            if not all_predictions:
                return {"success": False, "error": "All classifiers failed"}
            
            # Combine predictions
            combined_scores = np.zeros(len(self.classes))
            
            for name, predictions in all_predictions.items():
                weight = self.weights.get(name, 1.0)
                
                for pred in predictions:
                    class_idx = self.classes.index(pred["class"])
                    combined_scores[class_idx] += pred["confidence"] * weight
            
            # Normalize scores
            total_weight = sum(self.weights.values())
            combined_scores /= total_weight
            
            # Get sorted indices
            sorted_indices = np.argsort(combined_scores)[::-1]
            
            # Create predictions list
            predictions = []
            
            for i in sorted_indices:
                predictions.append({
                    "class": self.classes[i],
                    "confidence": float(combined_scores[i])
                })
            
            return {
                "success": True,
                "predictions": predictions
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
