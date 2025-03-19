import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection

class TableExtractor:
    def __init__(self, config):
        self.config = config
        self.detector = AutoModelForObjectDetection.from_pretrained(
            config.models["table_detection"]["name"]
        )
        self.structure_recognizer = AutoModelForObjectDetection.from_pretrained(
            config.models["table_structure"]["name"]
        )
        
    def extract_tables(self, page_image: Image) -> list:
        """Extract tables from a page image."""
        tables = []
        
        # Detect tables in the page
        table_boxes = self._detect_tables(page_image)
        
        for box in table_boxes:
            # Extract table region
            table_region = self._crop_table_region(page_image, box)
            
            # Recognize table structure
            structure = self._recognize_structure(table_region)
            
            # Extract cells and headers
            cells, headers = self._extract_cells(structure)
            
            # Convert to structured format
            table_data = self._structure_table_data(cells, headers)
            
            tables.append({
                "bbox": box.tolist(),
                "content": table_data,
                "headers": headers,
                "cells": cells
            })
            
        return tables
    
    def _detect_tables(self, image: Image) -> torch.Tensor:
        """Detect table regions in the image."""
        # Prepare image for the model
        inputs = self.detector.processor(images=image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = self.detector(**inputs)
        
        # Filter predictions by confidence
        confident_detections = []
        for score, label, box in zip(outputs.scores, outputs.labels, outputs.boxes):
            if score > self.config.models["table_detection"]["confidence_threshold"]:
                confident_detections.append(box)
                
        return torch.stack(confident_detections) if confident_detections else torch.tensor([])
    
    def _recognize_structure(self, table_image: Image) -> dict:
        """Recognize the structure of a table region."""
        inputs = self.structure_recognizer.processor(images=table_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.structure_recognizer(**inputs)
            
        return self._process_structure_outputs(outputs)
    
    def _process_structure_outputs(self, outputs) -> dict:
        """Process structure recognition outputs into a structured format."""
        cells = []
        rows = []
        cols = []
        
        # Process detected cells and their relationships
        for score, label, box in zip(outputs.scores, outputs.labels, outputs.boxes):
            if score > self.config.models["table_detection"]["confidence_threshold"]:
                cell_type = self.structure_recognizer.config.id2label[label.item()]
                cells.append({
                    "bbox": box.tolist(),
                    "type": cell_type
                })
                
        return {
            "cells": cells,
            "rows": self._identify_rows(cells),
            "cols": self._identify_columns(cells)
        }