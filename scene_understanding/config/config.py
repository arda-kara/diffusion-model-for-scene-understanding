"""
Configuration settings for the Scene Understanding system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "scene_understanding", "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image preprocessing settings
IMAGE_SIZE = (640, 640)  # Standard size for YOLOv8
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Object detection settings
DETECTION_MODEL = "yolov8n.pt"  # Options: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", etc.
DETECTION_CONFIDENCE = 0.5      # Minimum confidence threshold for object detection
DETECTION_IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# Captioning model settings
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"  # BLIP model
CAPTION_MAX_LENGTH = 50         # Maximum length of generated captions
CAPTION_NUM_BEAMS = 4           # Number of beams for beam search

# Relationship extraction settings
SPACY_MODEL = "en_core_web_sm"  # SpaCy model for NLP processing
RELATION_CONFIDENCE = 0.7       # Confidence threshold for extracted relationships

# Scene graph settings
GRAPH_NODE_SIZE = 2000          # Node size for visualization
GRAPH_EDGE_WIDTH = 2            # Edge width for visualization
GRAPH_FONT_SIZE = 10            # Font size for node and edge labels

# Visualization settings
VIZ_DPI = 300                   # DPI for saved visualizations
VIZ_FORMAT = "png"              # Format for saved visualizations
VIZ_SHOW_BBOXES = True          # Whether to show bounding boxes in the visualization
VIZ_SHOW_LABELS = True          # Whether to show object labels in the visualization

# Logging settings
LOG_LEVEL = "INFO"              # Logging level
LOG_FILE = os.path.join(PROJECT_ROOT, "scene_understanding.log")

# Performance settings
USE_GPU = True                  # Whether to use GPU for inference
BATCH_SIZE = 1                  # Batch size for processing multiple images 