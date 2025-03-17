#!/usr/bin/env python3
"""
Configuration file for waste classification system.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration
DETECTION_MODEL = "yolov8x"
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_MAX_OBJECTS = 50

# Classification models
CLASSIFICATION_MODELS = {
    "convnext_large": os.path.join(MODEL_DIR, "classifier", "convnext_large", "best_model.pt"),
    "efficientnetv2_l": os.path.join(MODEL_DIR, "classifier", "tf_efficientnetv2_l", "best_model.pt"),
    "swin_large": os.path.join(MODEL_DIR, "classifier", "swin_large_patch4_window7_224", "best_model.pt")
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    "convnext": 0.45,
    "efficientnet": 0.35,
    "swin": 0.20
}

# Waste classes
WASTE_CLASSES = [
    "cardboard", "glass", "metal", "paper", 
    "plastic", "trash", "e-waste", "organic", "textile", "mixed"
]

# Class-specific confidence thresholds
CLASS_THRESHOLDS = {
    "cardboard": 0.65,
    "glass": 0.70,
    "metal": 0.65,
    "paper": 0.65,
    "plastic": 0.70,
    "trash": 0.75,
    "e-waste": 0.75,
    "organic": 0.65,
    "textile": 0.70,
    "mixed": 0.80
}

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    "cardboard": (101, 67, 33),  # Brown
    "glass": (225, 225, 225),  # Light Gray
    "metal": (128, 128, 128),  # Gray
    "paper": (255, 255, 240),  # Off-White
    "plastic": (0, 0, 255),  # Red
    "trash": (0, 0, 0),  # Black
    "e-waste": (0, 255, 255),  # Yellow
    "organic": (0, 128, 0),  # Green
    "textile": (255, 0, 255),  # Magenta
    "mixed": (128, 0, 128)  # Purple
}

# Dataset paths
DATASET_PATHS = {
    "trashnet": os.path.join(DATA_DIR, "trashnet"),
    "taco": os.path.join(DATA_DIR, "taco"),
    "waste-pictures": os.path.join(DATA_DIR, "waste-pictures"),
    "mju-waste": os.path.join(DATA_DIR, "mju-waste"),
    "open-images": os.path.join(DATA_DIR, "open-images")
}

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
IMAGE_SIZE = 224
