"""
Simple example demonstrating the scene understanding system.
"""

import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene_understanding.core.pipeline import SceneUnderstandingPipeline
from scene_understanding.utils.image_utils import load_image


def main():
    """
    Main function.
    """
    # Initialize the pipeline
    pipeline = SceneUnderstandingPipeline()
    
    # Path to an example image
    # Replace with your own image path
    image_path = "examples/images/living_room.jpg"
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please add an image to the examples/images directory or update the image path.")
        return
    
    # Process the image
    scene_graph, output_paths = pipeline.process_image(image_path)
    
    # Print the results
    print(f"Processed image: {image_path}")
    print(f"Outputs saved to: {output_paths}")
    
    # Display the results
    # Load the output images
    bbox_image = Image.open(output_paths["bbox_image"])
    graph_with_image = Image.open(output_paths["scene_graph_with_image"])
    
    # Display the images
    plt.figure(figsize=(10, 8))
    plt.imshow(bbox_image)
    plt.title("Object Detection")
    plt.axis("off")
    plt.show()
    
    plt.figure(figsize=(20, 10))
    plt.imshow(graph_with_image)
    plt.title("Scene Graph with Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main() 