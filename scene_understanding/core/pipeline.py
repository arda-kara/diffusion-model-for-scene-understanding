"""
Main pipeline for scene understanding.
"""

import os
import time
import logging
from pathlib import Path

from scene_understanding.config import config
from scene_understanding.utils.image_utils import load_image, preprocess_for_yolo, draw_bounding_boxes, save_image
from scene_understanding.models.object_detection import ObjectDetector
from scene_understanding.models.image_captioning import ImageCaptioner
from scene_understanding.models.relationship_extraction import RelationshipExtractor
from scene_understanding.models.scene_graph import SceneGraphBuilder


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)


class SceneUnderstandingPipeline:
    """
    Main pipeline for scene understanding.
    """
    
    def __init__(self, detector=None, captioner=None, relationship_extractor=None, graph_builder=None):
        """
        Initialize the scene understanding pipeline.
        
        Args:
            detector (ObjectDetector, optional): Object detector. Defaults to None.
            captioner (ImageCaptioner, optional): Image captioner. Defaults to None.
            relationship_extractor (RelationshipExtractor, optional): Relationship extractor. Defaults to None.
            graph_builder (SceneGraphBuilder, optional): Scene graph builder. Defaults to None.
        """
        logger.info("Initializing scene understanding pipeline")
        
        # Initialize components
        self.detector = detector if detector else ObjectDetector()
        self.captioner = captioner if captioner else ImageCaptioner()
        self.relationship_extractor = relationship_extractor if relationship_extractor else RelationshipExtractor()
        self.graph_builder = graph_builder if graph_builder else SceneGraphBuilder()
        
        logger.info("Pipeline initialized")
    
    def process_image(self, image_path, output_dir=None, visualize=True):
        """
        Process an image and generate a scene graph.
        
        Args:
            image_path (str): Path to the input image.
            output_dir (str, optional): Directory to save outputs. Defaults to config.OUTPUT_DIR.
            visualize (bool, optional): Whether to visualize results. Defaults to True.
            
        Returns:
            tuple: (scene_graph, output_paths) where output_paths is a dictionary of saved file paths.
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_filename = Path(image_path).stem
        
        # Initialize output paths
        output_paths = {
            "bbox_image": os.path.join(output_dir, f"{base_filename}_bbox.png"),
            "scene_graph_json": os.path.join(output_dir, f"{base_filename}_graph.json"),
            "scene_graph_viz": os.path.join(output_dir, f"{base_filename}_graph.png"),
            "scene_graph_with_image": os.path.join(output_dir, f"{base_filename}_graph_with_image.png")
        }
        
        logger.info(f"Processing image: {image_path}")
        
        # Step 1: Load and preprocess the image
        start_time = time.time()
        image = load_image(image_path)
        logger.info(f"Image loaded in {time.time() - start_time:.2f} seconds")
        
        # Step 2: Detect objects
        start_time = time.time()
        image_for_yolo = preprocess_for_yolo(image)
        detections = self.detector.detect(image_for_yolo)
        logger.info(f"Object detection completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Detected {len(detections)} objects")
        
        # Step 3: Generate caption
        start_time = time.time()
        caption = self.captioner.generate_detailed_caption(image)
        logger.info(f"Caption generation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Caption: {caption}")
        
        # Step 4: Extract relationships
        start_time = time.time()
        relationships = self.relationship_extractor.extract_relationships(caption, detections)
        logger.info(f"Relationship extraction completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Extracted {len(relationships)} relationships")
        
        # Step 5: Build scene graph
        start_time = time.time()
        scene_graph = self.graph_builder.build_graph(detections, relationships, caption)
        logger.info(f"Scene graph construction completed in {time.time() - start_time:.2f} seconds")
        
        # Step 6: Save outputs
        if visualize:
            # Draw bounding boxes on the image
            bbox_image = draw_bounding_boxes(image, detections)
            save_image(bbox_image, output_paths["bbox_image"])
            
            # Save scene graph as JSON
            self.graph_builder.save_graph(scene_graph, output_paths["scene_graph_json"])
            
            # Visualize scene graph
            self.graph_builder.visualize(scene_graph, output_paths["scene_graph_viz"])
            
            # Visualize scene graph with image
            self.graph_builder.visualize_with_image(scene_graph, image, output_paths["scene_graph_with_image"])
            
            logger.info(f"Outputs saved to {output_dir}")
        
        return scene_graph, output_paths
    
    def process_batch(self, image_paths, output_dir=None, visualize=True):
        """
        Process a batch of images.
        
        Args:
            image_paths (list): List of paths to input images.
            output_dir (str, optional): Directory to save outputs. Defaults to config.OUTPUT_DIR.
            visualize (bool, optional): Whether to visualize results. Defaults to True.
            
        Returns:
            list: List of (scene_graph, output_paths) tuples.
        """
        results = []
        
        for image_path in image_paths:
            result = self.process_image(image_path, output_dir, visualize)
            results.append(result)
        
        return results
    
    def visualize_graph(self, scene_graph, image=None, output_path=None, show=False):
        """
        Visualize a scene graph.
        
        Args:
            scene_graph (networkx.DiGraph): Scene graph.
            image (PIL.Image, optional): Original image. Defaults to None.
            output_path (str, optional): Path to save the visualization to. Defaults to None.
            show (bool, optional): Whether to display the visualization. Defaults to False.
            
        Returns:
            str: Path to the saved visualization, if output_path is provided.
        """
        if image is not None:
            return self.graph_builder.visualize_with_image(scene_graph, image, output_path, show)
        else:
            return self.graph_builder.visualize(scene_graph, output_path, show)
    
    def query_scene_graph(self, scene_graph, query_type, query_params=None):
        """
        Query a scene graph.
        
        Args:
            scene_graph (networkx.DiGraph): Scene graph.
            query_type (str): Type of query.
            query_params (dict, optional): Query parameters. Defaults to None.
            
        Returns:
            Various: Query results.
        """
        return self.graph_builder.query_graph(scene_graph, query_type, query_params) 