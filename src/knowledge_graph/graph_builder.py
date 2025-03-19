import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

class KnowledgeGraphBuilder:
    def __init__(self, config):
        self.config = config.get("models", {}) if isinstance(config, dict) else config
        
        # Access model names with defaults
        text_model = self.config.get("text_embedding", {}).get("name", "sentence-transformers/all-mpnet-base-v2")
        image_model = self.config.get("image", {}).get("name", "openai/clip-vit-base-patch32")
        
        self.text_embedder = SentenceTransformer(text_model)
        self.image_processor = CLIPProcessor.from_pretrained(image_model)
        self.image_model = CLIPModel.from_pretrained(image_model)
        
    def build_graph(self, chunks, entities, relationships=None):
        """Build knowledge graph from document elements."""
        graph = nx.DiGraph()
        relationships = relationships or []
        
        # Add nodes for chunks
        self._add_chunk_nodes(graph, chunks)
        
        # Add entity nodes
        self._add_entity_nodes(graph, entities)
        
        # Add relationships
        self._add_relationships(graph, relationships)
        
        # Generate embeddings for nodes
        self._generate_embeddings(graph)
        
        return graph
        
    def _add_chunk_nodes(self, graph, chunks):
        """Add document chunks as nodes."""
        for chunk in chunks:
            graph.add_node(
                f"chunk_{chunk['id']}", 
                type="chunk",
                content=chunk["text"],
                metadata=chunk["metadata"]
            )
            
    def _add_entity_nodes(self, graph, entities):
        """Add entity nodes to the graph."""
        for entity in entities:
            graph.add_node(
                f"entity_{entity['id']}", 
                type="entity",
                content=entity["text"],
                entity_type=entity["type"],
                metadata=entity.get("metadata", {})
            )
            
            # Link entity to its source chunk if available
            if "chunk_id" in entity:
                graph.add_edge(
                    f"chunk_{entity['chunk_id']}", 
                    f"entity_{entity['id']}", 
                    type="contains"
                )
                
    def _add_relationships(self, graph, relationships):
        """Add relationships as edges between nodes."""
        for rel in relationships:
            source_type = "entity" if "entity" in rel["source"] else "chunk"
            target_type = "entity" if "entity" in rel["target"] else "chunk"
            
            graph.add_edge(
                f"{source_type}_{rel['source']}", 
                f"{target_type}_{rel['target']}", 
                type=rel["type"],
                confidence=rel.get("confidence", 1.0)
            )
            
    def _generate_embeddings(self, graph):
        """Generate embeddings for different node types."""
        embedding_dim = self.config.get("embedding", {}).get("text_dimension", 768)
        
        for node, data in graph.nodes(data=True):
            content = data.get("content", "")
            
            if not content:
                continue
                
            try:
                if data["type"] in ["chunk", "entity", "text"]:
                    embedding = self.text_embedder.encode(content)
                elif data["type"] == "table":
                    embedding = self._generate_table_embedding(content)
                elif data["type"] == "figure":
                    embedding = self._generate_image_embedding(content)
                else:
                    embedding = self.text_embedder.encode(str(content))
                
                # Ensure consistent dimensions
                if len(embedding) != embedding_dim:
                    logging.warning(f"Embedding dimension mismatch for node {node}. Skipping.")
                    continue
                    
                graph.nodes[node]["embedding"] = embedding
            except Exception as e:
                logging.error(f"Error generating embedding for node {node}: {str(e)}")
                continue
                
    def _generate_table_embedding(self, table_content):
        """Generate embedding for table content."""
        # Convert table content to string representation
        table_text = self._table_to_text(table_content)
        return self.text_embedder.encode(table_text)
        
    def _generate_image_embedding(self, image):
        """Generate embedding for image using CLIP."""
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.image_model.get_image_features(**inputs)
        return outputs.squeeze().detach().numpy()
        
    def _table_to_text(self, table_content):
        """Convert table content to text representation."""
        text_parts = []
        
        # Add headers if present
        if "headers" in table_content:
            text_parts.append(" | ".join(table_content["headers"]))
            
        # Add cell contents
        if "cells" in table_content:
            for row in table_content["cells"]:
                text_parts.append(" | ".join(str(cell) for cell in row))
                
        return "\n".join(text_parts)