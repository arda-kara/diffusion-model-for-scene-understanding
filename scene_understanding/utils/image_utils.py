"""
Utility functions for image processing.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from scene_understanding.config import config


def load_image(image_path):
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        PIL.Image: Loaded image in PIL format.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {e}")


def preprocess_image(image, target_size=None):
    """
    Preprocess an image for model input.
    
    Args:
        image (PIL.Image): Input image.
        target_size (tuple, optional): Target size for resizing. Defaults to config.IMAGE_SIZE.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE
        
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])
    
    # Apply preprocessing
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def preprocess_for_yolo(image):
    """
    Preprocess an image specifically for YOLOv8.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        numpy.ndarray: Preprocessed image in numpy format.
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_np


def preprocess_for_blip(image):
    """
    Preprocess an image specifically for BLIP captioning model.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # BLIP uses a different preprocessing pipeline
    blip_preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])
    
    # Apply preprocessing
    image_tensor = blip_preprocess(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes on an image.
    
    Args:
        image (PIL.Image): Input image.
        detections (list): List of detection results, each containing class_label, bbox, and confidence.
        
    Returns:
        PIL.Image: Image with bounding boxes drawn.
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Draw bounding boxes
    for detection in detections:
        class_label, bbox, confidence = detection
        
        # Extract coordinates
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Generate a color based on the class label (for consistent colors)
        color = get_color_for_class(class_label)
        
        # Draw rectangle
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_label}: {confidence:.2f}"
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to PIL image
    return Image.fromarray(image_np)


def get_color_for_class(class_label):
    """
    Generate a consistent color for a class label.
    
    Args:
        class_label (str): Class label.
        
    Returns:
        tuple: RGB color tuple.
    """
    # Hash the class label to get a consistent color
    hash_value = hash(class_label) % 255
    
    # Generate RGB values
    r = (hash_value * 123) % 255
    g = (hash_value * 45) % 255
    b = (hash_value * 67) % 255
    
    return (r, g, b)


def save_image(image, output_path):
    """
    Save an image to a file.
    
    Args:
        image (PIL.Image): Image to save.
        output_path (str): Path to save the image to.
        
    Returns:
        str: Path to the saved image.
    """
    image.save(output_path)
    return output_path 