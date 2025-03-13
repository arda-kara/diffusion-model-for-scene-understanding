"""
Scene graph module.
"""

import os
import networkx as nx

from scene_understanding.utils.graph_utils import (
    create_scene_graph,
    add_object_to_graph,
    add_relationship_to_graph,
    save_scene_graph,
    visualize_scene_graph,
    visualize_scene_graph_with_image
)


class SceneGraphBuilder:
    """
    Builds a scene graph from detections and relationships.
    """
    
    def __init__(self):
        """
        Initialize the scene graph builder.
        """
        pass
    
    def build_graph(self, detections, relationships, caption=None):
        """
        Build a scene graph from detections and relationships.
        
        Args:
            detections (list): List of (class_label, bbox, confidence) tuples.
            relationships (list): List of (subject_id, relation, object_id, confidence) tuples.
            caption (str, optional): Scene caption. Defaults to None.
            
        Returns:
            networkx.DiGraph: Scene graph.
        """
        # Create an empty scene graph
        G = create_scene_graph()
        
        # Add caption to the scene node if provided
        if caption:
            G.nodes["Scene"]["caption"] = caption
        
        # Add objects to the graph
        for i, (class_label, bbox, confidence) in enumerate(detections):
            object_id = f"obj_{i}"
            add_object_to_graph(G, object_id, class_label, bbox, confidence)
        
        # Add relationships to the graph
        for subject_id, relation, object_id, confidence in relationships:
            add_relationship_to_graph(G, subject_id, relation, object_id, confidence)
        
        return G
    
    def save_graph(self, G, output_path):
        """
        Save a scene graph to a file.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            output_path (str): Path to save the graph to.
            
        Returns:
            str: Path to the saved graph.
        """
        return save_scene_graph(G, output_path)
    
    def visualize(self, G, output_path=None, show=False):
        """
        Visualize a scene graph.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            output_path (str, optional): Path to save the visualization to. Defaults to None.
            show (bool, optional): Whether to display the visualization. Defaults to False.
            
        Returns:
            str: Path to the saved visualization, if output_path is provided.
        """
        return visualize_scene_graph(G, output_path, show)
    
    def visualize_with_image(self, G, image, output_path=None, show=False):
        """
        Visualize a scene graph alongside the original image.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            image: Original image (PIL Image).
            output_path (str, optional): Path to save the visualization to. Defaults to None.
            show (bool, optional): Whether to display the visualization. Defaults to False.
            
        Returns:
            str: Path to the saved visualization, if output_path is provided.
        """
        return visualize_scene_graph_with_image(G, image, output_path, show)
    
    def get_objects(self, G):
        """
        Get all objects in a scene graph.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            
        Returns:
            list: List of (object_id, class_label, bbox, confidence) tuples.
        """
        objects = []
        
        for node, attrs in G.nodes(data=True):
            if attrs.get("type") == "object":
                object_id = node
                class_label = attrs.get("class_label", "")
                bbox = attrs.get("bbox", (0, 0, 0, 0))
                confidence = attrs.get("confidence", 0.0)
                
                objects.append((object_id, class_label, bbox, confidence))
        
        return objects
    
    def get_relationships(self, G):
        """
        Get all relationships in a scene graph.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            
        Returns:
            list: List of (subject_id, relation, object_id, confidence) tuples.
        """
        relationships = []
        
        for subject_id, object_id, attrs in G.edges(data=True):
            if subject_id != "Scene":  # Skip edges from the scene node
                relation = attrs.get("relation", "")
                confidence = attrs.get("confidence", 0.0)
                
                relationships.append((subject_id, relation, object_id, confidence))
        
        return relationships
    
    def get_caption(self, G):
        """
        Get the caption of a scene graph.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            
        Returns:
            str: Scene caption, or None if not available.
        """
        return G.nodes["Scene"].get("caption", None)
    
    def query_graph(self, G, query_type, query_params=None):
        """
        Query a scene graph.
        
        Args:
            G (networkx.DiGraph): Scene graph.
            query_type (str): Type of query. Options: "objects", "relationships", "object_by_id", "relationships_by_object".
            query_params (dict, optional): Query parameters. Defaults to None.
            
        Returns:
            Various: Query results.
        """
        if query_type == "objects":
            return self.get_objects(G)
        
        elif query_type == "relationships":
            return self.get_relationships(G)
        
        elif query_type == "object_by_id":
            if query_params and "object_id" in query_params:
                object_id = query_params["object_id"]
                
                if object_id in G.nodes and G.nodes[object_id].get("type") == "object":
                    attrs = G.nodes[object_id]
                    class_label = attrs.get("class_label", "")
                    bbox = attrs.get("bbox", (0, 0, 0, 0))
                    confidence = attrs.get("confidence", 0.0)
                    
                    return (object_id, class_label, bbox, confidence)
            
            return None
        
        elif query_type == "relationships_by_object":
            if query_params and "object_id" in query_params:
                object_id = query_params["object_id"]
                relationships = []
                
                # Get relationships where the object is the subject
                for _, target, attrs in G.out_edges(object_id, data=True):
                    relation = attrs.get("relation", "")
                    confidence = attrs.get("confidence", 0.0)
                    
                    relationships.append((object_id, relation, target, confidence, "subject"))
                
                # Get relationships where the object is the object
                for source, _, attrs in G.in_edges(object_id, data=True):
                    if source != "Scene":  # Skip edges from the scene node
                        relation = attrs.get("relation", "")
                        confidence = attrs.get("confidence", 0.0)
                        
                        relationships.append((source, relation, object_id, confidence, "object"))
                
                return relationships
            
            return []
        
        else:
            raise ValueError(f"Unknown query type: {query_type}") 