import fitz
from PIL import Image
import io
from transformers import AutoModelForObjectDetection, AutoProcessor, DetrImageProcessor
import logging
import torch

class MultimodalDocumentProcessor:
    def __init__(self, config):
        self.config = config.get("models", {}) if isinstance(config, dict) else config
        
        # Access model names safely with defaults
        table_model = self.config.get("table_detection", {}).get("name", "microsoft/table-transformer-detection")
        self.table_detector = AutoModelForObjectDetection.from_pretrained(table_model)
        self.table_processor = AutoProcessor.from_pretrained(table_model)
        
        structure_model = self.config.get("table_structure", {}).get("name", "microsoft/table-transformer-structure-recognition")
        self.table_structure_recognizer = AutoModelForObjectDetection.from_pretrained(structure_model)

    def process_document(self, file_path):
        """Process document and extract multimodal elements."""
        document = {"text_blocks": [], "tables": [], "figures": []}
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            # Extract text with positioning
            text_blocks = self._extract_text_blocks(page)
            document["text_blocks"].extend(text_blocks)
            
            # Extract tables
            tables = self._extract_tables(page)
            document["tables"].extend(tables)
            
            # Extract figures
            figures = self._extract_figures(page, doc)
            document["figures"].extend(figures)
            
        return document

    def _extract_text_blocks(self, page):
        """Extract text blocks with spatial information."""
        blocks = []
        page_dict = page.get_text("dict")
        
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                try:
                    text = ""
                    # Extract text from spans if available
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text += span.get("text", "")
                    # Fallback to direct text if available
                    if not text and "text" in block:
                        text = block["text"]
                    
                    if text:
                        blocks.append({
                            "text": text,
                            "bbox": block.get("bbox", [0, 0, 0, 0]),
                            "page_num": page.number
                        })
                except Exception as e:
                    logging.warning(f"Error processing text block on page {page.number}: {str(e)}")
                    continue
        
        return blocks

    def _extract_tables(self, page):
        """Extract tables from a page using the table detection model."""
        tables = []
        try:
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Prepare image for model
            inputs = self.table_processor(images=img, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = self.table_detector(**inputs)
            
            # Process results - using correct post-processing method
            predictions = outputs.logits.softmax(-1)
            boxes = outputs.pred_boxes
            
            # Get scores and convert boxes to image scale
            scores = predictions.max(-1).values
            scaled_boxes = self.table_processor.post_process(outputs, target_sizes=[(img.size[1], img.size[0])])[0]['boxes']
            
            # Filter predictions with confidence > 0.7
            for score, box in zip(scores[0], scaled_boxes[0]):
                if score > 0.7:
                    tables.append({
                        "confidence": score.item(),
                        "bbox": box.tolist(),
                        "page_num": page.number
                    })
                    
        except Exception as e:
            logging.error(f"Error detecting tables on page {page.number}: {str(e)}")
            
        return tables

    def _extract_figures(self, page, doc):
        """Extract figures from a page."""
        figures = []
        try:
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                base_image = doc.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    figures.append({
                        "bbox": list(image_info[1:5]),  # x0, y0, x1, y1
                        "image": image,
                        "page_num": page.number,
                        "metadata": {
                            "colorspace": base_image["colorspace"],
                            "width": base_image["width"],
                            "height": base_image["height"]
                        }
                    })
                    
        except Exception as e:
            logging.error(f"Error extracting figures on page {page.number}: {str(e)}")
            
        return figures