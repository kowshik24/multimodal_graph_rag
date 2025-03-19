from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel

class FigureExtractor:
    def __init__(self, config):
        self.config = config
        self.clip_processor = CLIPProcessor.from_pretrained(config.models["image"]["name"])
        self.clip_model = CLIPModel.from_pretrained(config.models["image"]["name"])
        
    def extract_figures(self, page, doc) -> list:
        """Extract figures and their captions from a page."""
        figures = []
        
        # Get all images from the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            if base_image:
                # Convert to PIL Image
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Find caption
                caption = self._find_figure_caption(page, img)
                
                # Generate image embedding
                embedding = self._generate_image_embedding(image)
                
                figures.append({
                    "image": image,
                    "bbox": img[1:5],
                    "caption": caption,
                    "embedding": embedding,
                    "page_num": page.number
                })
                
        return figures
    
    def _find_figure_caption(self, page, img) -> str:
        """Find the caption associated with an image using proximity and patterns."""
        caption = ""
        img_bbox = img[1:5]
        
        # Get text blocks near the image
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                if self._is_caption_candidate(block, img_bbox):
                    text = block["text"].strip()
                    if self._is_caption_text(text):
                        caption = text
                        break
                        
        return caption
    
    def _is_caption_candidate(self, block, img_bbox) -> bool:
        """Check if a text block is a potential caption for the image."""
        block_bbox = block["bbox"]
        
        # Check if block is below or above the image within a threshold
        vertical_distance = min(
            abs(block_bbox[1] - img_bbox[3]),  # Distance to bottom of image
            abs(block_bbox[3] - img_bbox[1])   # Distance to top of image
        )
        
        # Check horizontal overlap
        horizontal_overlap = (
            max(0, min(block_bbox[2], img_bbox[2]) - max(block_bbox[0], img_bbox[0]))
        )
        
        return (vertical_distance < self.config.caption_distance_threshold and
                horizontal_overlap > 0)
    
    def _is_caption_text(self, text: str) -> bool:
        """Check if text matches caption patterns."""
        text_lower = text.lower()
        caption_patterns = [
            "figure", "fig.", "fig", "image", "illustration"
        ]
        
        return any(pattern in text_lower for pattern in caption_patterns)
    
    def _generate_image_embedding(self, image: Image) -> torch.Tensor:
        """Generate embedding for an image using CLIP."""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            
        return image_features.squeeze().numpy()