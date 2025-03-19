from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import spacy
from typing import List, Dict, Tuple

class EntityExtractor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("jean-baptiste/roberta-large-ner-english")
        self.model = AutoModelForTokenClassification.from_pretrained("jean-baptiste/roberta-large-ner-english")
        self.nlp = spacy.load("en_core_web_sm")
        
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
            
            # Add chunk reference
            for entity in chunk_entities:
                entity["chunk_id"] = chunk["id"]
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