"""
Command-line interface for the scene understanding system.
"""

import os
import argparse
import glob
from pathlib import Path

from scene_understanding.config import config
from scene_understanding.core.pipeline import SceneUnderstandingPipeline


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Scene Understanding System")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--image_dir", type=str, help="Path to directory containing input images")
    
    # Output options
    parser.add_argument("--output", type=str, default=config.OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    
    # Model options
    parser.add_argument("--detection_model", type=str, default=config.DETECTION_MODEL, help="Path to detection model")
    parser.add_argument("--caption_model", type=str, default=config.CAPTION_MODEL, help="Name of caption model")
    parser.add_argument("--detection_confidence", type=float, default=config.DETECTION_CONFIDENCE, help="Detection confidence threshold")
    
    # Visualization options
    parser.add_argument("--show", action="store_true", help="Show visualizations")
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize pipeline
    pipeline = SceneUnderstandingPipeline()
    
    # Process images
    if args.image:
        # Process a single image
        scene_graph, output_paths = pipeline.process_image(
            args.image,
            output_dir=args.output,
            visualize=not args.no_visualize
        )
        
        print(f"Processed image: {args.image}")
        print(f"Outputs saved to: {args.output}")
        
        # Show visualizations if requested
        if args.show and not args.no_visualize:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Show image with bounding boxes
            bbox_image = Image.open(output_paths["bbox_image"])
            plt.figure(figsize=(10, 8))
            plt.imshow(bbox_image)
            plt.title("Object Detection")
            plt.axis("off")
            plt.show()
            
            # Show scene graph with image
            graph_with_image = Image.open(output_paths["scene_graph_with_image"])
            plt.figure(figsize=(20, 10))
            plt.imshow(graph_with_image)
            plt.title("Scene Graph with Image")
            plt.axis("off")
            plt.show()
    
    else:
        # Process all images in a directory
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Process each image
        for image_path in image_paths:
            scene_graph, output_paths = pipeline.process_image(
                image_path,
                output_dir=args.output,
                visualize=not args.no_visualize
            )
            
            print(f"Processed image: {image_path}")
        
        print(f"All outputs saved to: {args.output}")


if __name__ == "__main__":
    main() 