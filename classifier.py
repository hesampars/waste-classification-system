#!/usr/bin/env python3
"""
Classifier module for waste classification system.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WasteClassifier:
    def __init__(self, model_name, model_path=None):
        """
        Initialize the waste classifier.
        
        Args:
            model_name: Name of the model architecture
            model_path: Path to the model weights
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Import timm dynamically to avoid import errors if not installed
            import timm
            
            # Create model
            self.model = timm.create_model(model_name, pretrained=False, num_classes=len(config.WASTE_CLASSES))
            
            # Load weights if provided
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded weights from: {model_path}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set up transforms
            self.transform = self._get_transforms()
            
            print(f"Initialized classifier: {model_name}")
        
        except ImportError:
            print("Error: timm package not found. Please install with: pip install timm")
            self.model = None
        except Exception as e:
            print(f"Error initializing classifier: {str(e)}")
            self.model = None
    
    def _get_transforms(self):
        """Get image transforms for the model."""
        # Get input size based on model
        if "efficientnet" in self.model_name and "v2" in self.model_name:
            input_size = 384
        else:
            input_size = 224
        
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def classify(self, image):
        """
        Classify an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Dictionary with classification results
        """
        if self.model is None:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            # Convert numpy array to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Apply transforms
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()
            
            # Get sorted indices
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # Create predictions list
            predictions = []
            
            for i in sorted_indices:
                predictions.append({
                    "class": config.WASTE_CLASSES[i],
                    "confidence": float(probabilities[i])
                })
            
            return {
                "success": True,
                "predictions": predictions
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
