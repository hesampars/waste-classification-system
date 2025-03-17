#!/usr/bin/env python3
"""
Google Colab setup script for waste classification system.
"""

# This script will be used in Google Colab to set up the environment
# and download the repository and datasets

# Install required packages
!pip install torch torchvision timm numpy pillow opencv-python matplotlib scikit-learn tqdm requests gradio ultralytics

# Clone the repository
!git clone https://github.com/yourusername/waste-classification-system.git
%cd waste-classification-system

# Create directories
!mkdir -p data models output

# Upload datasets to Google Drive
# Instructions for the user:
# 1. Upload your dataset zip files to Google Drive
# 2. Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')

# Copy datasets from Google Drive to the project
# Adjust the paths according to your Google Drive structure
!cp /content/drive/MyDrive/waste_datasets/*.zip data/

# Extract datasets
!python scripts/download_datasets.py --skip-existing

# Preprocess datasets
!python scripts/preprocess_datasets.py

# Train models
# Uncomment the model you want to train
# !python scripts/train.py --model convnext_large
# !python scripts/train.py --model tf_efficientnetv2_l
# !python scripts/train.py --model swin_large_patch4_window7_224
# !python scripts/train.py --model all

# Run the application
# !python app.py
