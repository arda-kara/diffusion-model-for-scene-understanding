"""
Object detection module using YOLOv8.
"""

import os
import torch
from ultralytics import YOLO

from scene_understanding.config import config


class ObjectDetector:
    """
    Object detector using YOLOv8.
    """
    
    def __init__(self, model_path=None, confidence_threshold=None, iou_threshold=None):
        """
        Initialize the object detector.
        
        Args:
            model_path (str, optional): Path to the YOLOv8 model. Defaults to config.DETECTION_MODEL.
            confidence_threshold (float, optional): Confidence threshold for detections. Defaults to config.DETECTION_CONFIDENCE.
            iou_threshold (float, optional): IoU threshold for NMS. Defaults to config.DETECTION_IOU_THRESHOLD.
        """
        if model_path is None:
            model_path = config.DETECTION_MODEL
        
        if confidence_threshold is None:
            confidence_threshold = config.DETECTION_CONFIDENCE
        
        if iou_threshold is None:
            iou_threshold = config.DETECTION_IOU_THRESHOLD
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLOv8 model from {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading YOLOv8 model: {e}")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        print(f"Using device: {self.device}")
    
    def detect(self, image):
        """
        Detect objects in an image.
        
        Args:
            image: Input image (PIL Image or numpy array).
            
        Returns:
            list: List of (class_label, bbox, confidence) tuples.
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get class label
                cls_id = int(box.cls.item())
                class_label = result.names[cls_id]
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence
                confidence = box.conf.item()
                
                # Add to detections
                detections.append((class_label, (x1, y1, x2, y2), confidence))
        
        return detections
    
    def detect_batch(self, images):
        """
        Detect objects in a batch of images.
        
        Args:
            images (list): List of input images.
            
        Returns:
            list: List of detection results, one per image.
        """
        # Run inference
        results = self.model(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        # Process results
        batch_detections = []
        
        for result in results:
            detections = []
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get class label
                cls_id = int(box.cls.item())
                class_label = result.names[cls_id]
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence
                confidence = box.conf.item()
                
                # Add to detections
                detections.append((class_label, (x1, y1, x2, y2), confidence))
            
            batch_detections.append(detections)
        
        return batch_detections 