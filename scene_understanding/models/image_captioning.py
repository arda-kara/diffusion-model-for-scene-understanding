"""
Image captioning module using BLIP.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from scene_understanding.config import config
from scene_understanding.utils.image_utils import preprocess_for_blip


class ImageCaptioner:
    """
    Image captioner using BLIP.
    """
    
    def __init__(self, model_name=None, max_length=None, num_beams=None):
        """
        Initialize the image captioner.
        
        Args:
            model_name (str, optional): Name of the BLIP model. Defaults to config.CAPTION_MODEL.
            max_length (int, optional): Maximum length of generated captions. Defaults to config.CAPTION_MAX_LENGTH.
            num_beams (int, optional): Number of beams for beam search. Defaults to config.CAPTION_NUM_BEAMS.
        """
        if model_name is None:
            model_name = config.CAPTION_MODEL
        
        if max_length is None:
            max_length = config.CAPTION_MAX_LENGTH
        
        if num_beams is None:
            num_beams = config.CAPTION_NUM_BEAMS
        
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Load the model
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            print(f"Loaded BLIP model from {model_name}")
        except Exception as e:
            raise ValueError(f"Error loading BLIP model: {e}")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def generate_caption(self, image, prompt=None):
        """
        Generate a caption for an image.
        
        Args:
            image: Input image (PIL Image).
            prompt (str, optional): Optional text prompt to guide caption generation.
            
        Returns:
            str: Generated caption.
        """
        # Process the image
        if prompt is None:
            # Default caption generation
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        else:
            # Conditional caption generation
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # Decode the caption
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    
    def generate_detailed_caption(self, image):
        """
        Generate a detailed caption for an image, focusing on relationships.
        
        Args:
            image: Input image (PIL Image).
            
        Returns:
            str: Detailed caption.
        """
        # Generate a basic caption
        basic_caption = self.generate_caption(image)
        
        # Generate a more detailed caption with a specific prompt
        detailed_prompt = "Describe the relationships between objects in this image."
        detailed_caption = self.generate_caption(image, prompt=detailed_prompt)
        
        # Combine the captions
        combined_caption = f"{basic_caption} {detailed_caption}"
        
        return combined_caption
    
    def generate_spatial_caption(self, image):
        """
        Generate a caption focusing on spatial relationships.
        
        Args:
            image: Input image (PIL Image).
            
        Returns:
            str: Spatial relationship caption.
        """
        # Generate a caption with a specific prompt for spatial relationships
        spatial_prompt = "Describe the spatial layout and positioning of objects in this image."
        spatial_caption = self.generate_caption(image, prompt=spatial_prompt)
        
        return spatial_caption
    
    def generate_multiple_captions(self, image, prompts):
        """
        Generate multiple captions for an image with different prompts.
        
        Args:
            image: Input image (PIL Image).
            prompts (list): List of text prompts.
            
        Returns:
            list: List of generated captions.
        """
        captions = []
        
        # Generate a caption for each prompt
        for prompt in prompts:
            caption = self.generate_caption(image, prompt=prompt)
            captions.append(caption)
        
        return captions 