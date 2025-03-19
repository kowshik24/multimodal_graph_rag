import spacy
import networkx as nx
from typing import List, Dict, Tuple

class RelationshipExtractor:
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_relationships(self, entities: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Extract relationships between entities."""
        relationships = []
        
        # Build entity index for efficient lookup
        entity_index = self._build_entity_index(entities)
        
        for chunk in chunks:
            # Extract syntactic relationships
            syntactic_rels = self._extract_syntactic_relationships(
                chunk["text"],
                entity_index
            )
            relationships.extend(syntactic_rels)
            
            # Extract semantic relationships
            semantic_rels = self._extract_semantic_relationships(
                chunk["text"],
                entity_index
            )
            relationships.extend(semantic_rels)
            
            # Extract cross-modal relationships
            if "tables" in chunk or "figures" in chunk:
                cross_modal_rels = self._extract_cross_modal_relationships(
                    chunk,
                    entity_index
                )
                relationships.extend(cross_modal_rels)
                
        return self._deduplicate_relationships(relationships)
    
    def _build_entity_index(self, entities: List[Dict]) -> Dict:
        """Build an index of entities for efficient lookup."""
        index = {}
        for entity in entities:
            key = entity["text"].lower()
            if key not in index:
                index[key] = []
            index[key].append(entity)
        return index
    
    def _extract_syntactic_relationships(self, text: str, entity_index: Dict) -> List[Dict]:
        """Extract relationships based on syntactic patterns."""
        relationships = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            # Extract subject-verb-object relationships
            for token in sent:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = token.text.lower()
                    verb = token.head.text
                    
                    # Find direct object
                    for child in token.head.children:
                        if child.dep_ == "dobj":
                            obj = child.text.lower()
                            
                            # Check if subject and object are known entities
                            if subject in entity_index and obj in entity_index:
                                relationships.append({
                                    "source": entity_index[subject][0]["id"],
                                    "target": entity_index[obj][0]["id"],
                                    "type": verb,
                                    "confidence": 1.0
                                })
                                
        return relationships
    
    def _extract_semantic_relationships(self, text: str, entity_index: Dict) -> List[Dict]:
        """Extract relationships based on semantic patterns and co-occurrence."""
        relationships = []
        doc = self.nlp(text)
        
        # Window-based co-occurrence
        window_size = 5
        tokens = [token.text.lower() for token in doc]
        
        for i, token in enumerate(tokens):
            if token in entity_index:
                window_start = max(0, i - window_size)
                window_end = min(len(tokens), i + window_size + 1)
                
                for j in range(window_start, window_end):
                    if i != j and tokens[j] in entity_index:
                        # Calculate co-occurrence strength
                        distance = abs(i - j)
                        strength = 1.0 - (distance / window_size)
                        
                        relationships.append({
                            "source": entity_index[token][0]["id"],
                            "target": entity_index[tokens[j]][0]["id"],
                            "type": "co-occurs",
                            "confidence": strength
                        })
                        
        return relationships