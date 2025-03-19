from transformers import AutoTokenizer, AutoModel
import spacy
import numpy as np

class SemanticContextPreservingChunker:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["models"]["text_embedding"]["name"])  
        self.model = AutoModel.from_pretrained(config["models"]["text_embedding"]["name"])
        self.nlp = spacy.load("en_core_web_sm")
        
    def chunk_document(self, document):
        """Create semantic chunks while preserving context."""
        chunks = []
        
        # Process text blocks
        sections = self._identify_sections(document["text_blocks"])
        for section in sections:
            section_chunks = self._create_semantic_chunks(section)
            chunks.extend(section_chunks)
            
        # Process tables and figures
        self._process_tables(document["tables"], chunks)
        self._process_figures(document["figures"], chunks)
        
        return self._link_chunks(chunks)

    def _create_semantic_chunks(self, section):
        """Create chunks based on semantic boundaries."""
        doc = self.nlp(section["text"])
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent.text))
            
            if current_tokens + sent_tokens > self.config["max_chunk_size"]:
                if current_chunk:
                    chunks.append(self._create_chunk_object(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens
                
        if current_chunk:
            chunks.append(self._create_chunk_object(current_chunk))
            
        return chunks