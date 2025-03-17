# Waste Classification System

An advanced waste classification system designed for industrial settings with mixed waste items on conveyor belts. The system uses state-of-the-art deep learning models optimized for GPU deployment, focusing on maximizing accuracy even with imperfect imaging conditions.

## Features

- Object detection using YOLOv8x to identify and localize waste items
- Ensemble classification with multiple specialized models
- Decision fusion with weighted ensemble and temporal consistency
- User-friendly Gradio interface for real-time classification
- Optimized for T4 GPU deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/waste-classification-system.git
cd waste-classification-system

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The system uses multiple public datasets:

- TrashNet
- TACO
- Waste-Pictures
- MJU-Waste
- Google Open Images

To download and prepare the datasets:

```bash
# Download datasets
python scripts/download_datasets.py

# Preprocess datasets
python scripts/preprocess_datasets.py
```

## Training

To train the classification models:

```bash
# Train all models
python scripts/train.py --model all

# Or train a specific model
python scripts/train.py --model convnext_large
```

## Usage

To run the application:

```bash
python app.py
```

This will start a Gradio web interface where you can upload images for waste classification.

## Project Structure

- `app.py`: Main application with Gradio interface
- `config.py`: Configuration settings
- `src/`: Source code modules
  - `detector.py`: Object detection module
  - `classifier.py`: Classification module
  - `ensemble.py`: Ensemble classification module
  - `data_utils.py`: Data utilities
- `scripts/`: Utility scripts
  - `download_datasets.py`: Dataset download script
  - `preprocess_datasets.py`: Dataset preprocessing script
  - `train.py`: Model training script
- `data/`: Dataset storage (created during setup)
- `models/`: Trained model storage (created during training)
- `output/`: Output storage (created during usage)

## License

[MIT License](LICENSE)
