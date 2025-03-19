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
            
    def _generate_embeddings(self, graph):
        """Generate embeddings for different node types."""
        for node, data in graph.nodes(data=True):
            if data["type"] == "text":
                embedding = self.text_embedder.encode(data["content"])
            elif data["type"] == "table":
                embedding = self._generate_table_embedding(data["content"])
            elif data["type"] == "figure":
                embedding = self._generate_image_embedding(data["content"])
            graph.nodes[node]["embedding"] = embedding