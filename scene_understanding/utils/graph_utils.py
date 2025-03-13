"""
Utility functions for graph operations.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from scene_understanding.config import config


def create_scene_graph():
    """
    Create an empty scene graph.
    
    Returns:
        networkx.DiGraph: Empty directed graph with a root "Scene" node.
    """
    G = nx.DiGraph()
    
    # Add the root "Scene" node
    G.add_node("Scene", type="root")
    
    return G


def add_object_to_graph(G, object_id, class_label, bbox, confidence):
    """
    Add an object node to the scene graph.
    
    Args:
        G (networkx.DiGraph): Scene graph.
        object_id (str): Unique identifier for the object.
        class_label (str): Class label of the object.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        confidence (float): Detection confidence.
        
    Returns:
        networkx.DiGraph: Updated scene graph.
    """
    # Add the object node
    G.add_node(
        object_id,
        type="object",
        class_label=class_label,
        bbox=bbox,
        confidence=confidence
    )
    
    # Connect the object to the scene
    G.add_edge("Scene", object_id, relation="contains")
    
    return G


def add_relationship_to_graph(G, subject_id, relation, object_id, confidence=1.0):
    """
    Add a relationship edge between two objects in the scene graph.
    
    Args:
        G (networkx.DiGraph): Scene graph.
        subject_id (str): ID of the subject node.
        relation (str): Relationship label.
        object_id (str): ID of the object node.
        confidence (float, optional): Confidence of the relationship. Defaults to 1.0.
        
    Returns:
        networkx.DiGraph: Updated scene graph.
    """
    # Add the relationship edge
    G.add_edge(
        subject_id,
        object_id,
        relation=relation,
        confidence=confidence
    )
    
    return G


def save_scene_graph(G, output_path):
    """
    Save a scene graph to a JSON file.
    
    Args:
        G (networkx.DiGraph): Scene graph.
        output_path (str): Path to save the graph to.
        
    Returns:
        str: Path to the saved graph.
    """
    # Convert the graph to a dictionary
    graph_data = {
        "nodes": [],
        "edges": []
    }
    
    # Add nodes
    for node, attrs in G.nodes(data=True):
        node_data = {"id": node}
        node_data.update(attrs)
        
        # Convert numpy arrays and other non-serializable types
        for key, value in node_data.items():
            if isinstance(value, np.ndarray):
                node_data[key] = value.tolist()
        
        graph_data["nodes"].append(node_data)
    
    # Add edges
    for source, target, attrs in G.edges(data=True):
        edge_data = {"source": source, "target": target}
        edge_data.update(attrs)
        
        # Convert numpy arrays and other non-serializable types
        for key, value in edge_data.items():
            if isinstance(value, np.ndarray):
                edge_data[key] = value.tolist()
        
        graph_data["edges"].append(edge_data)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    return output_path


def load_scene_graph(input_path):
    """
    Load a scene graph from a JSON file.
    
    Args:
        input_path (str): Path to the JSON file.
        
    Returns:
        networkx.DiGraph: Loaded scene graph.
    """
    # Load from JSON
    with open(input_path, 'r') as f:
        graph_data = json.load(f)
    
    # Create a new graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_data in graph_data["nodes"]:
        node_id = node_data.pop("id")
        G.add_node(node_id, **node_data)
    
    # Add edges
    for edge_data in graph_data["edges"]:
        source = edge_data.pop("source")
        target = edge_data.pop("target")
        G.add_edge(source, target, **edge_data)
    
    return G


def visualize_scene_graph(G, output_path=None, show=False):
    """
    Visualize a scene graph.
    
    Args:
        G (networkx.DiGraph): Scene graph.
        output_path (str, optional): Path to save the visualization to. Defaults to None.
        show (bool, optional): Whether to display the visualization. Defaults to False.
        
    Returns:
        str: Path to the saved visualization, if output_path is provided.
    """
    plt.figure(figsize=(12, 10))
    
    # Create a layout for the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if G.nodes[node].get("type") == "root":
            node_colors.append("lightblue")
            node_sizes.append(config.GRAPH_NODE_SIZE * 1.5)
        else:
            node_colors.append("lightgreen")
            node_sizes.append(config.GRAPH_NODE_SIZE)
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )
    
    # Draw edges
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        confidence = data.get("confidence", 1.0)
        edge_labels[(u, v)] = f"{relation}\n({confidence:.2f})"
    
    nx.draw_networkx_edges(
        G, pos,
        width=config.GRAPH_EDGE_WIDTH,
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=15
    )
    
    # Draw labels
    node_labels = {}
    for node in G.nodes():
        if G.nodes[node].get("type") == "root":
            node_labels[node] = "Scene"
        else:
            class_label = G.nodes[node].get("class_label", "")
            confidence = G.nodes[node].get("confidence", 1.0)
            node_labels[node] = f"{class_label}\n({confidence:.2f})"
    
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=config.GRAPH_FONT_SIZE,
        font_weight="bold"
    )
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=config.GRAPH_FONT_SIZE - 2
    )
    
    plt.title("Scene Graph", fontsize=16)
    plt.axis("off")
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches="tight")
        plt.close()
        return output_path
    
    if show:
        plt.show()
    
    plt.close()
    return None


def visualize_scene_graph_with_image(G, image, output_path=None, show=False):
    """
    Visualize a scene graph alongside the original image.
    
    Args:
        G (networkx.DiGraph): Scene graph.
        image (PIL.Image): Original image.
        output_path (str, optional): Path to save the visualization to. Defaults to None.
        show (bool, optional): Whether to display the visualization. Defaults to False.
        
    Returns:
        str: Path to the saved visualization, if output_path is provided.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display the image on the left
    ax1.imshow(np.array(image))
    ax1.set_title("Original Image", fontsize=16)
    ax1.axis("off")
    
    # Draw bounding boxes if available
    for node, attrs in G.nodes(data=True):
        if attrs.get("type") == "object" and "bbox" in attrs:
            bbox = attrs["bbox"]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Create a rectangle patch
            rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
            
            # Add label
            class_label = attrs.get("class_label", "")
            confidence = attrs.get("confidence", 1.0)
            label = f"{class_label} ({confidence:.2f})"
            ax1.text(x1, y1 - 5, label, color='r', fontsize=10, backgroundcolor='white')
    
    # Display the scene graph on the right
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if G.nodes[node].get("type") == "root":
            node_colors.append("lightblue")
            node_sizes.append(config.GRAPH_NODE_SIZE)
        else:
            node_colors.append("lightgreen")
            node_sizes.append(config.GRAPH_NODE_SIZE * 0.8)
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax2
    )
    
    # Draw edges
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        confidence = data.get("confidence", 1.0)
        edge_labels[(u, v)] = f"{relation}\n({confidence:.2f})"
    
    nx.draw_networkx_edges(
        G, pos,
        width=config.GRAPH_EDGE_WIDTH,
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        ax=ax2
    )
    
    # Draw labels
    node_labels = {}
    for node in G.nodes():
        if G.nodes[node].get("type") == "root":
            node_labels[node] = "Scene"
        else:
            class_label = G.nodes[node].get("class_label", "")
            confidence = G.nodes[node].get("confidence", 1.0)
            node_labels[node] = f"{class_label}\n({confidence:.2f})"
    
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=config.GRAPH_FONT_SIZE,
        font_weight="bold",
        ax=ax2
    )
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=config.GRAPH_FONT_SIZE - 2,
        ax=ax2
    )
    
    ax2.set_title("Scene Graph", fontsize=16)
    ax2.axis("off")
    
    plt.tight_layout()
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches="tight")
        plt.close()
        return output_path
    
    if show:
        plt.show()
    
    plt.close()
    return None 