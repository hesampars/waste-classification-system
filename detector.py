#!/usr/bin/env python3
"""
Object detector module for waste classification system.
"""

import torch
import numpy as np
from PIL import Image

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WasteObjectDetector:
    def __init__(self, model_name=config.DETECTION_MODEL):
        """
        Initialize the waste object detector.
        
        Args:
            model_name: Name of the YOLOv8 model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Import YOLO dynamically to avoid import errors if not installed
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print(f"Loaded YOLOv8 model: {model_name}")
        except ImportError:
            print("Error: ultralytics package not found. Please install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"Error loading YOLOv8 model: {str(e)}")
            self.model = None
    
    def detect(self, image, conf_threshold=None, iou_threshold=None):
        """
        Detect objects in an image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Run detection
            results = self.model(
                image_np, 
                conf=conf_threshold or config.DETECTION_CONF_THRESHOLD,
                iou=iou_threshold or config.DETECTION_IOU_THRESHOLD,
                max_det=config.DETECTION_MAX_OBJECTS
            )
            
            # Process results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence
                    conf = box.conf[0].cpu().numpy()
                    
                    # Get class
                    if box.cls.shape[0] > 0:
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = result.names[cls_id]
                    else:
                        cls_name = "unknown"
                    
                    # Add to detections
                    detections.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(conf),
                        "class": cls_name
                    })
            
            return {
                "success": True,
                "detections": detections,
                "image_shape": image_np.shape[:2]  # (height, width)
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def crop_detections(self, image, detections):
        """
        Crop detected objects from an image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            detections: Detection results from detect()
            
        Returns:
            List of cropped images
        """
        if not detections["success"]:
            return []
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Crop detections
        crops = []
        
        for det in detections["detections"]:
            x1, y1, x2, y2 = det["box"]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_np.shape[1], x2)
            y2 = min(image_np.shape[0], y2)
            
            # Crop image
            crop = image_np[y1:y2, x1:x2]
            
            # Convert to PIL Image
            crop_pil = Image.fromarray(crop)
            
            # Add to crops
            crops.append({
                "crop": crop_pil,
                "box": det["box"],
                "confidence": det["confidence"],
                "class": det["class"]
            })
        
        return crops
