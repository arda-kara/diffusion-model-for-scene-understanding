"""
Test script for NLP utilities in the scene understanding system.
This script tests the functionality of the NLP utilities and provides detailed diagnostics.
"""

import sys
import os
import spacy
from scene_understanding.utils.nlp_utils import (
    extract_subject_verb_object,
    extract_spatial_relationships,
    extract_relationships_from_caption,
    find_matching_object
)

def print_spacy_info():
    """Print information about the SpaCy installation and model."""
    print("\n=== SpaCy Information ===")
    print(f"SpaCy version: {spacy.__version__}")
    
    try:
        nlp = spacy.load("en_core_web_sm")
        print(f"Model loaded: en_core_web_sm")
        print(f"Available pipeline components: {', '.join(nlp.pipe_names)}")
        
        # Check for critical components
        if 'parser' in nlp.pipe_names:
            print("✓ Parser component is available (required for dependency parsing)")
        else:
            print("✗ Parser component is NOT available - this will affect relationship extraction")
            
        if 'tagger' in nlp.pipe_names:
            print("✓ Tagger component is available (required for POS tagging)")
        else:
            print("✗ Tagger component is NOT available - this will affect relationship extraction")
            
        if 'sentencizer' in nlp.pipe_names or 'parser' in nlp.pipe_names:
            print("✓ Sentence boundary detection is available")
        else:
            print("✗ Sentence boundary detection is NOT available - this will affect caption processing")
        
        # Test basic functionality
        test_sentence = "The cat sits on the mat."
        doc = nlp(test_sentence)
        print("\nTest sentence parsing:")
        print(f"Sentence: {test_sentence}")
        print("Tokens:", [token.text for token in doc])
        print("POS tags:", [token.pos_ for token in doc])
        print("Dependencies:", [(token.text, token.dep_, token.head.text) for token in doc])
        print("Sentences:", [sent.text for sent in doc.sents])
        
    except Exception as e:
        print(f"Error loading SpaCy model: {e}")
        print("Please run 'python install_dependencies.py' to install the required model.")

def test_extract_subject_verb_object():
    """Test the extract_subject_verb_object function."""
    print("\n=== Testing extract_subject_verb_object ===")
    
    test_sentences = [
        "The cat sits on the mat.",
        "A person is riding a bicycle on the street.",
        "The dog is chasing the ball in the park."
    ]
    
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        try:
            triples = extract_subject_verb_object(sentence)
            if triples:
                print("Extracted SVO triples:")
                for subj, verb, obj in triples:
                    print(f"  - Subject: '{subj}', Verb: '{verb}', Object: '{obj}'")
            else:
                print("No SVO triples extracted.")
                
            # Detailed analysis
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(sentence)
            print("\nDetailed parsing:")
            for token in doc:
                print(f"Token: '{token.text}', POS: {token.pos_}, Dep: {token.dep_}, Head: '{token.head.text}'")
                
            # Find potential subjects and objects
            subjects = [token.text for token in doc if token.dep_ in ('nsubj', 'nsubjpass')]
            objects = [token.text for token in doc if token.dep_ in ('dobj', 'pobj')]
            verbs = [token.text for token in doc if token.pos_ == 'VERB']
            
            print(f"Potential subjects: {subjects}")
            print(f"Potential objects: {objects}")
            print(f"Potential verbs: {verbs}")
            
        except Exception as e:
            print(f"Error during extraction: {e}")

def test_extract_spatial_relationships():
    """Test the extract_spatial_relationships function."""
    print("\n=== Testing extract_spatial_relationships ===")
    
    test_sentences = [
        "The cat is on the mat.",
        "The book is under the table.",
        "A person is standing next to a car."
    ]
    
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        try:
            relationships = extract_spatial_relationships(sentence)
            if relationships:
                print("Extracted spatial relationships:")
                for subj, rel, obj in relationships:
                    print(f"  - Subject: '{subj}', Relation: '{rel}', Object: '{obj}'")
            else:
                print("No spatial relationships extracted.")
                
            # Detailed analysis
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(sentence)
            print("\nDetailed parsing:")
            for token in doc:
                print(f"Token: '{token.text}', POS: {token.pos_}, Dep: {token.dep_}, Head: '{token.head.text}'")
                
            # Find potential spatial prepositions
            spatial_preps = [token.text for token in doc if token.pos_ == 'ADP' and token.dep_ == 'prep']
            print(f"Potential spatial prepositions: {spatial_preps}")
            
        except Exception as e:
            print(f"Error during extraction: {e}")

def test_extract_relationships_from_caption():
    """Test the extract_relationships_from_caption function."""
    print("\n=== Testing extract_relationships_from_caption ===")
    
    test_captions = [
        "A person riding a bicycle on the street.",
        "A dog chasing a ball in the park.",
        "A cat sitting on a windowsill watching birds."
    ]
    
    test_objects = [
        ["person", "bicycle", "street"],
        ["dog", "ball", "park"],
        ["cat", "windowsill", "birds"]
    ]
    
    for caption, objects in zip(test_captions, test_objects):
        print(f"\nCaption: {caption}")
        print(f"Objects: {objects}")
        try:
            relationships = extract_relationships_from_caption(caption, objects)
            if relationships:
                print("Extracted relationships:")
                for subj, rel, obj in relationships:
                    print(f"  - Subject: '{subj}', Relation: '{rel}', Object: '{obj}'")
            else:
                print("No relationships extracted.")
                
            # Test object matching
            print("\nTesting object matching:")
            for word in caption.split():
                for obj in objects:
                    match = find_matching_object(word, objects)
                    if match:
                        print(f"Word '{word}' matched with object '{match}'")
            
        except Exception as e:
            print(f"Error during extraction: {e}")

def main():
    """Main function."""
    print("=== NLP Utilities Test ===")
    
    # Print SpaCy information
    print_spacy_info()
    
    # Test SVO extraction
    test_extract_subject_verb_object()
    
    # Test spatial relationship extraction
    test_extract_spatial_relationships()
    
    # Test relationship extraction from caption
    test_extract_relationships_from_caption()
    
    print("\n=== Test completed ===")
    print("If you encountered issues, please run 'python install_dependencies.py' to ensure all dependencies are correctly installed.")

if __name__ == "__main__":
    main() 