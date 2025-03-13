"""
Utility functions for NLP operations.
"""

import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load SpaCy model
try:
    # Try to load the full model with all components
    nlp = spacy.load("en_core_web_sm")
    print("Loaded SpaCy model with all components")
    
    # Ensure the model has the necessary components
    if not nlp.has_pipe('parser'):
        print("Warning: SpaCy model does not have a parser component")
    
    # Add sentencizer if not present (for sentence boundary detection)
    if not nlp.has_pipe('sentencizer') and not nlp.has_pipe('parser'):
        nlp.add_pipe('sentencizer')
        print("Added sentencizer to SpaCy model")
        
except OSError:
    # If the model is not installed, provide instructions
    print("SpaCy model 'en_core_web_sm' not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    # Create a blank model as fallback
    nlp = spacy.blank("en")
    # Add the sentencizer component to handle sentence boundaries
    nlp.add_pipe('sentencizer')
    print("Using blank SpaCy model with sentencizer")

# Print the pipeline components for debugging
print(f"SpaCy pipeline components: {', '.join(nlp.pipe_names)}")

def preprocess_text(text):
    """
    Preprocess text for NLP operations.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_entities(text):
    """
    Extract named entities from text.
    
    Args:
        text (str): Input text.
        
    Returns:
        list: List of (entity_text, entity_type) tuples.
    """
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def extract_noun_phrases(text):
    """
    Extract noun phrases from text.
    
    Args:
        text (str): Input text.
        
    Returns:
        list: List of noun phrases.
    """
    try:
        doc = nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases
    except Exception as e:
        print(f"Error extracting noun phrases: {e}")
        return []


# Simple rule-based fallback parser for subject-verb-object extraction
def simple_svo_extraction(text):
    """
    Simple rule-based extraction of subject-verb-object triples.
    
    Args:
        text (str): Input text.
        
    Returns:
        list: List of (subject, verb, object) tuples.
    """
    words = text.lower().split()
    triples = []
    
    # Common verbs to look for
    common_verbs = ["is", "are", "sits", "sitting", "stands", "standing", 
                   "lies", "lying", "placed", "located", "positioned", 
                   "contains", "holds", "has", "rides", "riding", 
                   "drives", "driving", "walks", "walking", "runs", "running"]
    
    # Common prepositions
    prepositions = ["on", "in", "at", "by", "near", "next", "to", "beside", 
                   "above", "below", "under", "over", "behind", "front", 
                   "inside", "outside", "between", "with"]
    
    # Look for simple patterns like "X is on Y" or "X sits on Y"
    for i in range(len(words) - 2):
        # Check for "X verb" pattern
        if i+1 < len(words) and words[i+1] in common_verbs:
            subject = words[i]
            verb = words[i+1]
            
            # Check for "X verb Y" pattern (direct object)
            if i+2 < len(words) and words[i+2] not in prepositions:
                obj = words[i+2]
                triples.append((subject, verb, obj))
            
            # Check for "X verb prep Y" pattern (prepositional object)
            elif i+3 < len(words) and words[i+2] in prepositions:
                relation = words[i+2]
                obj = words[i+3]
                triples.append((subject, relation, obj))
    
    # Look for patterns like "X on Y" (implied verb)
    for i in range(len(words) - 2):
        if words[i+1] in prepositions:
            subject = words[i]
            relation = words[i+1]
            obj = words[i+2]
            triples.append((subject, relation, obj))
    
    return triples


def extract_subject_verb_object(text):
    """
    Extract subject-verb-object triples from text.
    
    Args:
        text (str): Input text.
        
    Returns:
        list: List of (subject, verb, object) tuples.
    """
    triples = []
    
    try:
        # Try to process with SpaCy
        doc = nlp(text)
        
        # Process each sentence
        for sent in doc.sents:
            sent_triples = []
            
            # Find all verbs
            for token in sent:
                if token.pos_ == "VERB":
                    # Find subject
                    subjects = []
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            # Get the full noun phrase for the subject
                            subject_span = get_span_for_token(child)
                            subjects.append(subject_span)
                    
                    # Find direct objects
                    direct_objects = []
                    for child in token.children:
                        if child.dep_ == "dobj":
                            # Get the full noun phrase for the object
                            object_span = get_span_for_token(child)
                            direct_objects.append(object_span)
                    
                    # Find prepositional objects
                    prep_objects = []
                    for child in token.children:
                        if child.dep_ == "prep":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    # Get the full noun phrase for the object
                                    object_span = get_span_for_token(grandchild)
                                    prep_objects.append((child.text, object_span))
                    
                    # Create triples with direct objects
                    for subject in subjects:
                        for dobj in direct_objects:
                            sent_triples.append((subject, token.text, dobj))
                    
                    # Create triples with prepositional objects
                    for subject in subjects:
                        for prep, pobj in prep_objects:
                            sent_triples.append((subject, prep, pobj))
            
            triples.extend(sent_triples)
        
    except Exception as e:
        print(f"Error in SpaCy parsing: {e}")
    
    # If SpaCy parsing didn't work, try the simple rule-based approach
    if not triples:
        print("SpaCy parsing didn't extract any triples, using rule-based fallback")
        triples = simple_svo_extraction(text)
    
    return triples


def get_span_for_token(token):
    """
    Get the full noun phrase span for a token.
    
    Args:
        token (spacy.tokens.Token): The token to get the span for.
        
    Returns:
        str: The text of the span.
    """
    # Start with the token itself
    span_tokens = [token]
    
    # Add compound words
    for child in token.children:
        if child.dep_ in ["compound", "amod", "det"]:
            span_tokens.append(child)
    
    # Sort by position in the sentence
    span_tokens.sort(key=lambda t: t.i)
    
    # Join the tokens
    return " ".join(t.text for t in span_tokens)


# Simple rule-based fallback for spatial relationship extraction
def simple_spatial_extraction(text, objects):
    """
    Simple rule-based extraction of spatial relationships.
    
    Args:
        text (str): Input text.
        objects (list): List of object class labels.
        
    Returns:
        list: List of (subject, relation, object) tuples.
    """
    words = text.lower().split()
    relationships = []
    
    # Define spatial prepositions
    spatial_preps = [
        "on", "in", "at", "by", "near", "next", "to", "beside", "above", "below",
        "under", "over", "behind", "front", "inside", "outside", "between"
    ]
    
    # Look for patterns like "X on Y" or "X is on Y"
    for i, word in enumerate(words):
        if word in spatial_preps:
            # Look for objects before and after the preposition
            before_idx = max(0, i-3)
            after_idx = min(len(words)-1, i+3)
            
            before_text = ' '.join(words[before_idx:i])
            after_text = ' '.join(words[i+1:after_idx+1])
            
            for obj1 in objects:
                for obj2 in objects:
                    if obj1 != obj2:
                        # Check if obj1 is before the preposition and obj2 is after
                        if obj1.lower() in before_text.lower() and obj2.lower() in after_text.lower():
                            relationships.append((obj1, word, obj2))
    
    return relationships


def extract_spatial_relationships(text, objects=None):
    """
    Extract spatial relationships from text.
    
    Args:
        text (str): Input text.
        objects (list, optional): List of object class labels. If None, extracts relationships
                                 without matching to specific objects.
        
    Returns:
        list: List of (subject, relation, object) tuples.
    """
    relationships = []
    
    # Define spatial prepositions
    spatial_preps = [
        "on", "in", "at", "by", "near", "next to", "beside", "above", "below",
        "under", "over", "behind", "in front of", "inside", "outside", "between"
    ]
    
    try:
        # Parse the text with SpaCy
        doc = nlp(text)
        
        # Process each sentence
        for sent in doc.sents:
            # Find preposition phrases
            for token in sent:
                if token.text.lower() in spatial_preps or token.lemma_.lower() in spatial_preps:
                    # Find the object of the preposition
                    for child in token.children:
                        if child.dep_ == "pobj":
                            obj_span = get_span_for_token(child)
                            
                            # Find the subject (what the object is related to)
                            current = token
                            while current.head != current:  # Traverse up until root
                                current = current.head
                                if current.pos_ in ["NOUN", "PROPN"]:
                                    subject_span = get_span_for_token(current)
                                    
                                    # If objects list is provided, match with detected objects
                                    if objects:
                                        subject_match = find_matching_object(subject_span, objects)
                                        obj_match = find_matching_object(obj_span, objects)
                                        
                                        if subject_match and obj_match:
                                            relationships.append((subject_match, token.text, obj_match))
                                    else:
                                        # If no objects list, just use the extracted spans
                                        relationships.append((subject_span, token.text, obj_span))
                                    
                                    break
    
    except Exception as e:
        print(f"Error in spatial relationship extraction: {e}")
    
    # If no relationships were found and objects are provided, use the simple rule-based approach
    if not relationships and objects:
        print("SpaCy parsing didn't extract any spatial relationships, using rule-based fallback")
        relationships = simple_spatial_extraction(text, objects)
    
    # If still no relationships and we have at least 2 objects, create a generic one
    if not relationships and objects and len(objects) >= 2:
        # Just add a generic "near" relationship between the first two objects
        relationships.append((objects[0], "near", objects[1]))
    
    return relationships


def find_matching_object(text, objects):
    """
    Find a matching object from the list of detected objects.
    
    Args:
        text (str): Text to match.
        objects (list): List of object class labels.
        
    Returns:
        str: Matching object class label, or None if no match is found.
    """
    if not text or not objects:
        return None
        
    text = text.lower()
    
    # Direct match
    for obj in objects:
        if text == obj.lower():
            return obj
    
    # Check if text contains any of the objects
    for obj in objects:
        if obj.lower() in text:
            return obj
    
    # Check if any object contains the text
    for obj in objects:
        if text in obj.lower():
            return obj
    
    # Check for word overlap
    text_words = set(text.split())
    for obj in objects:
        obj_words = set(obj.lower().split())
        if text_words.intersection(obj_words):
            return obj
    
    return None


def extract_relationships_from_caption(caption, objects):
    """
    Extract relationships from a caption.
    
    Args:
        caption (str): Image caption.
        objects (list): List of object class labels.
        
    Returns:
        list: List of (subject, relation, object) tuples.
    """
    relationships = []
    
    try:
        # Extract subject-verb-object triples
        svo_triples = extract_subject_verb_object(caption)
        
        # Process SVO triples
        for subj, verb, obj in svo_triples:
            subj_match = find_matching_object(subj, objects)
            obj_match = find_matching_object(obj, objects)
            
            if subj_match and obj_match and subj_match != obj_match:
                relationships.append((subj_match, verb, obj_match))
        
        # Extract spatial relationships
        spatial_rels = extract_spatial_relationships(caption, objects)
        
        # Add spatial relationships that aren't already in the list
        for subj, rel, obj in spatial_rels:
            if not any(r[0] == subj and r[2] == obj for r in relationships):
                relationships.append((subj, rel, obj))
    
    except Exception as e:
        print(f"Warning: Error in relationship extraction: {e}. Using fallback method.")
    
    # If no relationships were found through NLP, try direct text matching
    if not relationships:
        print("No relationships found through NLP, trying direct text matching")
        
        # Look for direct mentions of objects in the caption
        caption_lower = caption.lower()
        
        # Check for common relationship patterns
        for obj1 in objects:
            for obj2 in objects:
                if obj1 != obj2:
                    obj1_lower = obj1.lower()
                    obj2_lower = obj2.lower()
                    
                    # Check if both objects are mentioned in the caption
                    if obj1_lower in caption_lower and obj2_lower in caption_lower:
                        # Check for common spatial relationships
                        for relation in ["on", "in", "at", "by", "near", "next to", "beside", "above", "below"]:
                            if relation in caption_lower:
                                # Check if the relation is between these two objects
                                rel_idx = caption_lower.find(relation)
                                obj1_idx = caption_lower.find(obj1_lower)
                                obj2_idx = caption_lower.find(obj2_lower)
                                
                                # Simple heuristic: if obj1 is before relation and obj2 is after
                                if obj1_idx < rel_idx < obj2_idx:
                                    relationships.append((obj1, relation, obj2))
                                    break
                        
                        # If no specific relation was found, use a generic one
                        if not any(r[0] == obj1 and r[2] == obj2 for r in relationships):
                            # Check for verbs like "sitting", "standing", etc.
                            for verb in ["sitting", "standing", "lying", "placed"]:
                                if verb in caption_lower:
                                    relationships.append((obj1, verb, obj2))
                                    break
    
    # If still no relationships and we have at least 2 objects, create generic relationships
    if not relationships and len(objects) >= 2:
        # Create simple relationships between pairs of objects
        for i in range(len(objects) - 1):
            relationships.append((objects[i], "related to", objects[i + 1]))
    
    # If no relationships were found, add at least one generic relationship
    if not relationships and len(objects) >= 2:
        relationships.append((objects[0], "in scene with", objects[1]))
    
    return relationships


def generate_relationship_from_positions(obj1, bbox1, obj2, bbox2):
    """
    Generate a spatial relationship based on the relative positions of two objects.
    
    Args:
        obj1 (str): Class label of the first object.
        bbox1 (tuple): Bounding box of the first object (x1, y1, x2, y2).
        obj2 (str): Class label of the second object.
        bbox2 (tuple): Bounding box of the second object (x1, y1, x2, y2).
        
    Returns:
        tuple: (subject, relation, object) tuple, or None if no clear relationship.
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate centers
    center_x1 = (x1_1 + x2_1) / 2
    center_y1 = (y1_1 + y2_1) / 2
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Check for containment (one object inside another)
    if (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2):
        return (obj1, "contains", obj2)
    
    if (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1):
        return (obj2, "contains", obj1)
    
    # Check for above/below
    if abs(center_x1 - center_x2) < (x2_1 - x1_1 + x2_2 - x1_2) / 2:
        if center_y1 < center_y2:
            return (obj1, "above", obj2)
        else:
            return (obj1, "below", obj2)
    
    # Check for left/right
    if abs(center_y1 - center_y2) < (y2_1 - y1_1 + y2_2 - y1_2) / 2:
        if center_x1 < center_x2:
            return (obj1, "left of", obj2)
        else:
            return (obj1, "right of", obj2)
    
    # Check for near
    distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
    max_dim = max(x2_1 - x1_1, y2_1 - y1_1, x2_2 - x1_2, y2_2 - y1_2)
    
    if distance < max_dim * 2:
        return (obj1, "near", obj2)
    
    return None 