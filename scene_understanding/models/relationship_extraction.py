"""
Relationship extraction module.
"""

from scene_understanding.config import config
from scene_understanding.utils.nlp_utils import (
    extract_relationships_from_caption,
    generate_relationship_from_positions
)


class RelationshipExtractor:
    """
    Extracts relationships between objects in a scene.
    """
    
    def __init__(self, confidence_threshold=None):
        """
        Initialize the relationship extractor.
        
        Args:
            confidence_threshold (float, optional): Confidence threshold for extracted relationships.
                Defaults to config.RELATION_CONFIDENCE.
        """
        if confidence_threshold is None:
            confidence_threshold = config.RELATION_CONFIDENCE
        
        self.confidence_threshold = confidence_threshold
    
    def extract_from_caption(self, caption, detections):
        """
        Extract relationships from a caption.
        
        Args:
            caption (str): Image caption.
            detections (list): List of (class_label, bbox, confidence) tuples.
            
        Returns:
            list: List of (subject_id, relation, object_id, confidence) tuples.
        """
        # Extract object class labels
        object_labels = [detection[0] for detection in detections]
        
        # Extract relationships from caption
        caption_relationships = extract_relationships_from_caption(caption, object_labels)
        
        # Map relationships to object IDs
        relationships = []
        
        for subject_label, relation, object_label in caption_relationships:
            # Find subject and object in detections
            subject_id = None
            object_id = None
            
            for i, (class_label, _, _) in enumerate(detections):
                if class_label == subject_label and subject_id is None:
                    subject_id = f"obj_{i}"
                
                if class_label == object_label and object_id is None:
                    object_id = f"obj_{i}"
                
                if subject_id is not None and object_id is not None:
                    break
            
            # If both subject and object are found, add the relationship
            if subject_id is not None and object_id is not None:
                relationships.append((subject_id, relation, object_id, self.confidence_threshold))
        
        return relationships
    
    def extract_from_positions(self, detections):
        """
        Extract relationships based on the relative positions of objects.
        
        Args:
            detections (list): List of (class_label, bbox, confidence) tuples.
            
        Returns:
            list: List of (subject_id, relation, object_id, confidence) tuples.
        """
        relationships = []
        
        # Compare each pair of objects
        for i, (class_label_i, bbox_i, _) in enumerate(detections):
            for j, (class_label_j, bbox_j, _) in enumerate(detections):
                if i != j:  # Skip self-relationships
                    # Generate relationship based on positions
                    relationship = generate_relationship_from_positions(
                        class_label_i, bbox_i, class_label_j, bbox_j
                    )
                    
                    if relationship:
                        subject_label, relation, object_label = relationship
                        
                        # Check if the relationship is between the correct objects
                        if subject_label == class_label_i and object_label == class_label_j:
                            relationships.append((f"obj_{i}", relation, f"obj_{j}", self.confidence_threshold))
        
        return relationships
    
    def extract_relationships(self, caption, detections):
        """
        Extract relationships from both caption and positions.
        
        Args:
            caption (str): Image caption.
            detections (list): List of (class_label, bbox, confidence) tuples.
            
        Returns:
            list: List of (subject_id, relation, object_id, confidence) tuples.
        """
        relationships = []
        
        try:
            # Extract relationships from caption
            caption_relationships = self.extract_from_caption(caption, detections)
            relationships.extend(caption_relationships)
        except Exception as e:
            print(f"Warning: Error in caption-based relationship extraction: {e}")
        
        try:
            # Extract relationships from positions
            position_relationships = self.extract_from_positions(detections)
            
            # Add position-based relationships if they don't conflict with caption-based ones
            for subject_id, relation, object_id, confidence in position_relationships:
                # Check if this relationship already exists
                exists = False
                
                for s_id, r, o_id, _ in relationships:
                    if s_id == subject_id and o_id == object_id:
                        exists = True
                        break
                
                if not exists:
                    relationships.append((subject_id, relation, object_id, confidence))
        except Exception as e:
            print(f"Warning: Error in position-based relationship extraction: {e}")
        
        # If no relationships were found, create at least one generic relationship
        if not relationships and len(detections) >= 2:
            obj_0_id = f"obj_0"
            obj_1_id = f"obj_1"
            relationships.append((obj_0_id, "in scene with", obj_1_id, self.confidence_threshold))
        
        return relationships 