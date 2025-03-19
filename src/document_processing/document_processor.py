import fitz
from PIL import Image
import io
from transformers import AutoModelForObjectDetection, AutoProcessor
import logging

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