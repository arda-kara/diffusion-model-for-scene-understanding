# Diffusion Model for Scene Understanding

A system that automatically interprets 2D images to build scene graphs describing objects, relationships, and contextual cues, using a combination of object detection, image captioning, and natural language processing techniques.

## Project Overview

This project aims to create a comprehensive scene understanding system that:

1. Detects objects in an image using state-of-the-art object detection models (YOLOv8/DETR)
2. Generates descriptive captions using vision-language models (BLIP)
3. Extracts relationships between objects using NLP techniques
4. Constructs a scene graph that represents the semantic understanding of the image
5. Provides visualization tools for human interpretation

The system is designed to be robust, with multiple fallback mechanisms to ensure reliable operation even when individual components encounter issues.

## Key Features

- **Robust Object Detection**: Identifies objects with high accuracy using YOLOv8
- **Detailed Image Captioning**: Generates descriptive captions using BLIP
- **Advanced Relationship Extraction**: Identifies spatial and semantic relationships between objects
- **Comprehensive Scene Graph Construction**: Builds a complete graph representation of the scene
- **Multiple Fallback Mechanisms**: Ensures the system continues to function even when components fail
- **Detailed Visualization**: Renders the scene graph for easy interpretation

## Architecture

The system consists of several key components:

### 1. Image Preprocessing
- Resizes and normalizes images for model input
- Applies transformations required by different models
- Handles various image formats and resolutions

### 2. Object Detection
- Uses YOLOv8 to identify objects and their bounding boxes
- Filters detections based on confidence thresholds
- Handles overlapping detections with non-maximum suppression

### 3. Captioning Module
- Uses BLIP to generate detailed scene descriptions
- Provides context beyond simple object detection
- Captures activities and relationships in the scene

### 4. Relationship Extraction
- Parses captions using SpaCy for NLP processing
- Extracts subject-relation-object triples from text
- Identifies spatial relationships based on object positions
- Includes rule-based fallbacks when NLP processing fails

### 5. Scene Graph Construction
- Builds a graph representation with objects as nodes and relationships as edges
- Integrates information from all previous components
- Provides a structured representation for downstream applications

### 6. Visualization
- Renders the scene graph using NetworkX and Matplotlib
- Displays object bounding boxes on the original image
- Shows relationships between objects with labeled edges

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/scene-understanding.git
cd scene-understanding
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install the required SpaCy model and NLTK resources:
```
python install_dependencies.py
```

Alternatively, you can install the SpaCy model manually:
```
python -m spacy download en_core_web_sm
```

4. Download model weights (if not automatically downloaded):
```
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Basic Usage

```python
from scene_understanding.core.pipeline import SceneUnderstandingPipeline

# Initialize the pipeline
pipeline = SceneUnderstandingPipeline()

# Process an image
scene_graph = pipeline.process_image("path/to/image.jpg")

# Visualize the scene graph
pipeline.visualize_graph(scene_graph, save_path="output.png")
```

### Command Line Interface

```
python -m scene_understanding.run --image path/to/image.jpg --output output_directory
```

### Advanced Usage

```python
from scene_understanding.core.pipeline import SceneUnderstandingPipeline
from scene_understanding.config.settings import Settings

# Create custom settings
custom_settings = Settings(
    detection_confidence=0.4,
    use_gpu=True,
    max_objects=15,
    extract_attributes=True
)

# Initialize the pipeline with custom settings
pipeline = SceneUnderstandingPipeline(settings=custom_settings)

# Process an image with additional options
scene_graph = pipeline.process_image(
    "path/to/image.jpg",
    detect_objects=True,
    generate_caption=True,
    extract_relationships=True
)

# Access components of the scene graph
objects = scene_graph.get_objects()
relationships = scene_graph.get_relationships()

# Export the scene graph to different formats
scene_graph.to_json("scene_graph.json")
scene_graph.to_networkx()  # Returns a NetworkX graph object
```

## Project Structure

```
scene_understanding/
├── core/                  # Core pipeline components
│   ├── pipeline.py        # Main pipeline implementation
│   ├── scene_graph.py     # Scene graph data structure
│   └── processor.py       # Image processing utilities
├── models/                # Model implementations and wrappers
│   ├── detection/         # Object detection models
│   │   └── yolo.py        # YOLOv8 implementation
│   ├── captioning/        # Image captioning models
│   │   └── blip.py        # BLIP implementation
│   └── relationship/      # Relationship extraction models
├── utils/                 # Utility functions
│   ├── nlp_utils.py       # NLP processing utilities
│   ├── image_utils.py     # Image handling utilities
│   └── graph_utils.py     # Graph manipulation utilities
├── data/                  # Data handling
│   ├── input/             # Input images
│   └── output/            # Output scene graphs and visualizations
├── visualization/         # Visualization tools
│   ├── graph_vis.py       # Scene graph visualization
│   └── image_vis.py       # Image annotation visualization
└── config/                # Configuration files
    └── settings.py        # System settings and parameters
```

## Examples

### Example 1: Kitchen Scene

![Kitchen Scene](examples/kitchen_scene.jpg)

Input: An image of a kitchen with various appliances and objects.

Output:
- Detected objects: refrigerator, oven, microwave, sink, counter, cabinets
- Extracted relationships:
  - (microwave, on, counter)
  - (sink, in, counter)
  - (oven, next to, refrigerator)

### Example 2: Outdoor Scene

![Outdoor Scene](examples/outdoor_scene.jpg)

Input: An image of a park with people and animals.

Output:
- Detected objects: person, dog, tree, bench, bicycle
- Extracted relationships:
  - (person, sitting on, bench)
  - (dog, near, person)
  - (bicycle, leaning against, tree)

## Troubleshooting

### Common Issues

#### SpaCy Model Issues

If you encounter errors related to SpaCy or NLP processing:

1. Make sure you have installed the required models and resources:
```
python install_dependencies.py
```

2. If you see errors about sentence boundaries in SpaCy, the system should automatically handle them with fallback mechanisms, but you can also try reinstalling the SpaCy model:
```
python -m pip install --upgrade spacy
python -m spacy download en_core_web_sm
```

3. Run the test script to diagnose NLP issues:
```
python test_nlp.py
```

#### Object Detection Issues

If object detection is not working correctly:

1. Ensure the YOLOv8 weights are downloaded correctly
2. Check if CUDA is available if you're using GPU acceleration
3. Try adjusting the detection confidence threshold in the settings

#### Memory Issues

If you encounter out-of-memory errors:

1. Reduce the image size in the settings
2. Use a smaller YOLOv8 model (e.g., YOLOv8n instead of YOLOv8x)
3. Process images in batches rather than all at once

## Evaluation

The system is evaluated based on:

### Quantitative Metrics
- Object detection accuracy (mAP, precision, recall)
- Caption quality (BLEU, ROUGE, CIDEr scores)
- Relationship extraction accuracy (compared to human annotations)
- Scene graph completeness (node and edge coverage)
- Processing speed and resource usage

### Qualitative Assessment
- Correctness of identified relationships
- Relevance of the scene graph to the image content
- Usefulness for downstream applications
- Handling of complex scenes and edge cases

## Future Extensions

### Short-term Improvements
- Support for more object detection models (Faster R-CNN, DETR)
- Integration with more advanced captioning models
- Improved relationship extraction with transformer-based models
- Better handling of occlusions and partially visible objects

### Long-term Vision
- 3D scene graph construction from multiple viewpoints
- Multi-modal integration (image + text + audio)
- Knowledge graph linking for common-sense reasoning
- Task & motion planning integration for robotics applications
- Temporal scene graphs for video understanding
- Fine-grained attribute extraction for objects

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project uses the following open-source libraries:
- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce Research
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model implementations
- [SpaCy](https://spacy.io/) for natural language processing
- [NetworkX](https://networkx.org/) for graph processing
- [Matplotlib](https://matplotlib.org/) and [Pillow](https://python-pillow.org/) for visualization

## Citation

If you use this project in your research, please cite:

```
@software{scene_understanding_system,
  author = {Your Name},
  title = {Diffusion Model for Scene Understanding},
  year = {2023},
  url = {https://github.com/yourusername/scene-understanding}
}
``` 