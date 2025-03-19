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

    def _create_chunk_object(self, sentences):
        """Create a chunk object from a list of sentences."""
        text = " ".join([sent.text for sent in sentences])
        return {
            "id": self._generate_chunk_id(),
            "text": text,
            "metadata": {
                "start": sentences[0].start_char,
                "end": sentences[-1].end_char
            }
        }

    def _process_tables(self, tables, chunks):
        """Process and add tables as chunks."""
        for idx, table in enumerate(tables):
            chunks.append({
                "id": f"table_{idx}",
                "text": self._table_to_text(table),
                "type": "table",
                "metadata": {
                    "bbox": table.get("bbox", []),
                    "headers": table.get("headers", []),
                    "cells": table.get("cells", [])
                }
            })

    def _process_figures(self, figures, chunks):
        """Process and add figures as chunks."""
        for idx, figure in enumerate(figures):
            chunks.append({
                "id": f"figure_{idx}",
                "text": figure.get("caption", ""),
                "type": "figure",
                "metadata": {
                    "bbox": figure.get("bbox", []),
                    "image": figure.get("image", None)
                }
            })

    def _link_chunks(self, chunks):
        """Link chunks based on semantic relationships and proximity."""
        # Add relationships between consecutive chunks
        for i in range(len(chunks) - 1):
            chunks[i]["next_chunk"] = chunks[i + 1]["id"]
            chunks[i + 1]["prev_chunk"] = chunks[i]["id"]
            
        return chunks

    def _generate_chunk_id(self):
        """Generate a unique chunk ID."""
        # Use a simple counter for now - could be made more sophisticated
        if not hasattr(self, '_chunk_counter'):
            self._chunk_counter = 0
        self._chunk_counter += 1
        return f"chunk_{self._chunk_counter}"

    def _table_to_text(self, table):
        """Convert table content to text representation."""
        text_parts = []
        
        # Add headers if present
        if "headers" in table:
            text_parts.append(" | ".join(str(h) for h in table["headers"]))
            
        # Add cell contents
        if "cells" in table:
            for row in table["cells"]:
                text_parts.append(" | ".join(str(cell) for cell in row))
                
        return "\n".join(text_parts) if text_parts else ""