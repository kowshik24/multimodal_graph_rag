from transformers import AutoTokenizer, AutoModel
import spacy
import numpy as np
import re

class SemanticContextPreservingChunker:
    def __init__(self, config):
        self.config = config
        # Handle both nested and flat config structures
        model_name = (config.get("models", {}).get("text_embedding", {}).get("name") or 
                     config.get("text_embedding_model", "sentence-transformers/all-mpnet-base-v2"))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Initialize spacy with download if needed
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.max_chunk_size = config.get("chunking", {}).get("max_chunk_size", 512)
        
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

    def _identify_sections(self, text_blocks):
        """Identify logical sections from text blocks."""
        sections = []
        current_section = []
        
        for block in text_blocks:
            text = block["text"]
            # Check if block starts a new section (e.g., headers)
            if self._is_section_header(text):
                if current_section:
                    sections.append({
                        "text": "\n".join(b["text"] for b in current_section),
                        "blocks": current_section
                    })
                current_section = []
            current_section.append(block)
            
        # Add final section if exists
        if current_section:
            sections.append({
                "text": "\n".join(b["text"] for b in current_section),
                "blocks": current_section
            })
            
        return sections
        
    def _is_section_header(self, text):
        """Check if text appears to be a section header."""
        # Basic heuristics for identifying headers
        text = text.strip()
        if not text:
            return False
            
        # Check for common header patterns
        header_patterns = [
            r"^[0-9]+\.[0-9]*\s+[A-Z]",  # Numbered sections
            r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,4}$",  # Title Case
            r"^[A-Z\s]{4,}$"  # ALL CAPS
        ]
        
        return any(re.match(pattern, text) for pattern in header_patterns)

    def _create_semantic_chunks(self, section):
        """Create chunks based on semantic boundaries."""
        doc = self.nlp(section["text"])
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent.text))
            
            if current_tokens + sent_tokens > self.max_chunk_size:
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