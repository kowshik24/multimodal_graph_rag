import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

class ContextAwareRetriever:
    def __init__(self, knowledge_graph, config):
        self.knowledge_graph = knowledge_graph
        self.config = config
        # Handle config as dictionary
        embedding_model = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
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
            if "embedding" in data:
                similarity = 1 - cosine(query_embedding, data["embedding"])
                similarities.append((node, similarity))
                
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]