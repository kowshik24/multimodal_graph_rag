from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import spacy
from typing import List, Dict, Tuple
import re

class EntityExtractor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("jean-baptiste/roberta-large-ner-english")
        self.model = AutoModelForTokenClassification.from_pretrained("jean-baptiste/roberta-large-ner-english")
        self.nlp = spacy.load("en_core_web_sm")
        self._entity_counter = 0  # Add counter for generating unique IDs
    
    def _generate_entity_id(self) -> str:
        """Generate a unique entity ID."""
        self._entity_counter += 1
        return f"entity_{self._entity_counter}"

    def extract_entities(self, chunks: List[Dict]) -> List[Dict]:
        """Extract entities from document chunks."""
        entities = []
        
        for chunk in chunks:
            # Extract named entities
            named_entities = self._extract_named_entities(chunk["text"])
            
            # Extract technical entities
            technical_entities = self._extract_technical_entities(chunk["text"])
            
            # Merge and deduplicate entities
            chunk_entities = self._merge_entities(named_entities, technical_entities)
            
            # Add chunk reference and entity ID
            for entity in chunk_entities:
                entity["chunk_id"] = chunk["id"]
                entity["id"] = self._generate_entity_id()  # Add unique ID
                entities.append(entity)
                
        return self._deduplicate_entities(entities)
    
    def _extract_named_entities(self, text: str) -> List[Dict]:
        """Extract named entities using RoBERTa model."""
        entities = []
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1)
            
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        current_entity = {"text": "", "type": "", "start": 0}
        
        for idx, (token, pred) in enumerate(zip(tokens, predictions[0])):
            entity_label = self.model.config.id2label[pred.item()]
            
            if entity_label.startswith("B-"):
                if current_entity["text"]:
                    entities.append(current_entity.copy())
                current_entity = {
                    "text": token,
                    "type": entity_label[2:],
                    "start": idx
                }
            elif entity_label.startswith("I-") and current_entity["text"]:
                current_entity["text"] += " " + token
                
        if current_entity["text"]:
            entities.append(current_entity)
            
        return entities
    
    def _extract_technical_entities(self, text: str) -> List[Dict]:
        """Extract technical entities using spaCy and custom patterns."""
        doc = self.nlp(text)
        entities = []
        
        # Custom patterns for technical entities
        patterns = [
            (r"(?i)\b[a-z_][a-z0-9_]*\([^)]*\)", "FUNCTION"),
            (r"(?i)\b(class|interface)\s+[A-Z][a-zA-Z0-9_]*", "CLASS"),
            (r"(?i)\b[A-Z][A-Z0-9_]*\b", "CONSTANT"),
            (r"(?i)\b(https?://|www\.)[^\s]+", "URL")
        ]
        
        # Extract using patterns
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start()
                })
                
        return entities
    
    def _merge_entities(self, named_entities: List[Dict], technical_entities: List[Dict]) -> List[Dict]:
        """Merge named entities and technical entities, handling overlaps."""
        merged = named_entities + technical_entities
        # Sort by start position to handle overlaps
        merged.sort(key=lambda x: x["start"])
        
        # Remove overlapping entities, keeping the longer one
        result = []
        if not merged:
            return result
            
        current = merged[0]
        for next_entity in merged[1:]:
            current_end = current["start"] + len(current["text"])
            # If there's no overlap, add current to result and move to next
            if next_entity["start"] >= current_end:
                result.append(current)
                current = next_entity
            else:
                # If there's overlap, keep the longer entity
                if len(next_entity["text"]) > len(current["text"]):
                    current = next_entity
        
        result.append(current)
        return result
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities across chunks."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a tuple of identifying features
            entity_key = (entity["text"].lower(), entity["type"])
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
                
        return unique_entities