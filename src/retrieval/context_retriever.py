import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import torch
import logging

class ContextAwareRetriever:
    def __init__(self, knowledge_graph, config):
        self.knowledge_graph = knowledge_graph
        self.config = config
        # Use the same model as used in graph building
        embedding_model = (config.get("models", {})
                         .get("text_embedding", {})
                         .get("name", "sentence-transformers/all-mpnet-base-v2"))
        self.embedder = SentenceTransformer(embedding_model)
        
    def retrieve(self, query, top_k=5):
        """Retrieve relevant context using hybrid search."""
        query_embedding = self.embedder.encode(query)
        
        # Initial vector similarity search
        initial_candidates = self._vector_search(query_embedding, top_k * 2)
        
        # Expand through graph
        expanded_candidates = self._graph_expansion(initial_candidates)
        
        # Score and rank candidates
        scored_candidates = self._hybrid_scoring(
            expanded_candidates, 
            query_embedding
        )
        
        # Select top results
        return self._select_top_results(scored_candidates, top_k)
        
    def _vector_search(self, query_embedding, k):
        """Perform vector similarity search."""
        similarities = []
        
        for node, data in self.knowledge_graph.nodes(data=True):
            if "embedding" not in data:
                continue
                
            try:
                node_embedding = data["embedding"]
                # Ensure embeddings have same dimensions
                if len(query_embedding) != len(node_embedding):
                    logging.warning(f"Embedding dimension mismatch for node {node}. "
                                 f"Expected {len(query_embedding)}, got {len(node_embedding)}")
                    continue
                    
                similarity = 1 - cosine(query_embedding, node_embedding)
                similarities.append((node, similarity))
            except Exception as e:
                logging.error(f"Error computing similarity for node {node}: {str(e)}")
                continue
                
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def _graph_expansion(self, initial_candidates):
        """Expand initial candidates through graph connections."""
        expanded = set(node for node, _ in initial_candidates)
        
        # For each initial candidate
        for node, _ in initial_candidates:
            # Add neighbors to expanded set
            neighbors = set(self.knowledge_graph.neighbors(node))
            expanded.update(neighbors)
            
        return list(expanded)
    
    def _hybrid_scoring(self, candidates, query_embedding):
        """Score candidates using vector similarity and graph metrics."""
        scored = []
        
        for node in candidates:
            node_data = self.knowledge_graph.nodes[node]
            if "embedding" not in node_data:
                continue
                
            # Vector similarity score
            similarity = 1 - cosine(query_embedding, node_data["embedding"])
            
            # Graph-based importance (using degree as a simple metric)
            importance = len(list(self.knowledge_graph.neighbors(node))) / len(self.knowledge_graph)
            
            # Combine scores (weighted sum)
            final_score = 0.7 * similarity + 0.3 * importance
            scored.append((node, final_score))
            
        return scored
        
    def _select_top_results(self, scored_candidates, k):
        """Select top k results from scored candidates."""
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:k]