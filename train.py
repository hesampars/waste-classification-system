#!/usr/bin/env python3
"""
Script to train waste classification models.
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_utils import create_dataloaders

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train waste classification models")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help="Directory containing the processed data")
    parser.add_argument("--model-dir", type=str, default=config.MODEL_DIR,
                        help="Directory to save trained models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["convnext_large", "tf_efficientnetv2_l", "swin_large_patch4_window7_224", "all"],
                        help="Model architecture to train")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE,
                        help="Input image size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    return parser.parse_args()

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, model_dir, model_name, num_epochs=25):
    """
    Train a model.
    
    Args:
        model: Model to train
        dataloaders: Dictionary of train and validation dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        model_dir: Directory to save model
        model_name: Name of the model
        num_epochs: Number of training epochs
        
    Returns:
        Trained model and training history
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize variables
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Start time
    start_time = time.time()
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Update learning rate
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            # Print epoch statistics
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
                # Save model
                torch.save(best_model_wts, os.path.join(model_dir, "best_model.pt"))
                print(f"Saved best model with accuracy: {best_acc:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'history': history
            }, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        print()
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    
    # Save training history
    with open(os.path.join(model_dir, "history.json"), 'w') as f:
        # Convert numpy values to Python types
        history_serializable = {
            "train_loss": [float(x) for x in history["train_loss"]],
            "train_acc": [float(x) for x in history["train_acc"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "val_acc": [float(x) for x in history["val_acc"]]
        }
        json.dump(history_serializable, f, indent=2)
    
    # Plot training history
    plot_training_history(history, os.path.join(model_dir, "training_history.png"))
    
    return model, history

def plot_training_history(history, output_path):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, dataloader, device, classes, output_dir):
    """
    Evaluate a model on the test set.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to evaluate on
        classes: List of class names
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Initialize variables
    all_preds = []
    all_labels = []
    
    # Iterate over data
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Save classification report
    with open(os.path.join(output_dir, "classification_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    return {
        "confusion_matrix": cm,
        "classification_report": report
    }

def train_single_model(model_name, dataloaders, classes, args):
    """
    Train a single model.
    
    Args:
        model_name: Name of the model architecture
        dataloaders: Dictionary of dataloaders
        classes: List of class names
        args: Command line arguments
        
    Returns:
        Trained model
    """
    print(f"\nTraining {model_name} model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model directory
    model_dir = os.path.join(args.model_dir, "classifier", model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model
    if "efficientnet" in model_name:
        image_size = 384
    else:
        image_size = 224
    
    model = timm.create_model(model_name, pretrained=True, num_classes=len(classes))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume training if requested
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders={"train": dataloaders["train"], "val": dataloaders["val"]},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_dir=model_dir,
        model_name=model_name,
        num_epochs=args.epochs
    )
    
    # Evaluate model
    evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        classes=classes,
        output_dir=model_dir
    )
    
    return model

def main():
    """Main function to train waste classification models."""
    args = parse_args()
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Get class names
    classes = dataloaders["classes"]
    print(f"Training with {len(classes)} classes: {classes}")
    
    # Save class names
    class_file = os.path.join(args.model_dir, "classes.json")
    with open(class_file, 'w') as f:
        json.dump(classes, f, indent=2)
    
    # Train models
    if args.model == "all":
        # Train all models
        models = ["convnext_large", "tf_efficientnetv2_l", "swin_large_patch4_window7_224"]
        for model_name in models:
            train_single_model(model_name, dataloaders, classes, args)
    else:
        # Train single model
        train_single_model(args.model, dataloaders, classes, args)
    
    print("Training completed successfully")

if __name__ == "__main__":
    main()
