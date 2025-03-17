#!/usr/bin/env python3
"""
Main application for waste classification system.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
import gradio as gr

# Import local modules
import config
from src.detector import WasteObjectDetector
from src.classifier import WasteClassifier
from src.ensemble import WasteEnsembleClassifier

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Waste Classification System")
    parser.add_argument("--mode", type=str, default="app", choices=["app", "test"],
                        help="Application mode: 'app' for Gradio interface, 'test' for testing")
    parser.add_argument("--image", type=str, help="Path to test image (for test mode)")
    return parser.parse_args()

def process_image(image):
    """
    Process an image through the waste classification pipeline.
    
    Args:
        image: Input image (PIL Image or numpy array)
        
    Returns:
        Processed image with detections and classifications
    """
    # Initialize detector and classifier
    detector = WasteObjectDetector(model_name=config.DETECTION_MODEL)
    ensemble = WasteEnsembleClassifier(
        model_paths=config.CLASSIFICATION_MODELS,
        weights=config.ENSEMBLE_WEIGHTS,
        classes=config.WASTE_CLASSES
    )
    
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Make a copy for visualization
    vis_image = image_np.copy()
    
    # Detect waste objects
    detections = detector.detect(image_np)
    
    if not detections["success"]:
        return vis_image, "Detection failed: " + detections.get("error", "Unknown error")
    
    # Crop detected objects
    crops = detector.crop_detections(image_np, detections)
    
    # Classify each crop
    results = []
    
    for crop_info in crops:
        crop_img = crop_info["crop"]
        box = crop_info["box"]
        
        # Classify crop
        classification = ensemble.classify(crop_img)
        
        if classification["success"]:
            # Get top prediction
            top_class = classification["predictions"][0]["class"]
            top_conf = classification["predictions"][0]["confidence"]
            
            # Check if confidence exceeds threshold
            if top_conf >= config.CLASS_THRESHOLDS.get(top_class, 0.7):
                # Add to results
                results.append({
                    "box": box,
                    "class": top_class,
                    "confidence": top_conf
                })
                
                # Draw on visualization image
                x1, y1, x2, y2 = box
                color = config.CLASS_COLORS.get(top_class, (0, 0, 255))
                
                # Draw box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{top_class}: {top_conf:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image, f"Detected {len(results)} waste items"

def create_gradio_interface():
    """Create Gradio interface for waste classification."""
    with gr.Blocks(title="Advanced Waste Classification System") as app:
        gr.Markdown("# Advanced Waste Classification System")
        gr.Markdown("Upload an image to detect and classify waste items.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                submit_btn = gr.Button("Classify Waste")
            
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Detection Results")
                output_text = gr.Textbox(label="Results")
        
        submit_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, output_text]
        )
        
        gr.Markdown("## Instructions")
        gr.Markdown("""
        1. Upload an image containing waste items
        2. Click 'Classify Waste' to process the image
        3. View the detection results and classification
        
        The system can detect and classify the following waste types:
        - Cardboard
        - Glass
        - Metal
        - Paper
        - Plastic
        - Trash (general)
        - E-waste
        - Organic
        - Textile
        - Mixed waste
        """)
    
    return app

def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == "app":
        # Create and launch Gradio interface
        app = create_gradio_interface()
        app.launch(server_name="0.0.0.0", share=True)
    
    elif args.mode == "test" and args.image:
        # Test mode with single image
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        # Load image
        image = Image.open(args.image)
        
        # Process image
        result_image, result_text = process_image(image)
        
        # Display results
        print(result_text)
        
        # Save result image
        output_path = os.path.join(config.OUTPUT_DIR, "result.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()
