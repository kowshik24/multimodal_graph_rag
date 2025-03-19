from typing import List, Dict
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContextAssembler:
    def __init__(self, config):
        self.config = config
        
    def assemble_context(self, 
                        retrieved_nodes: List[str],
                        knowledge_graph: nx.DiGraph,
                        query_embedding: np.ndarray,
                        max_tokens: int = 4000) -> Dict:
        """Assemble retrieved nodes into a coherent context."""
        # Get node data
        nodes_data = self._get_nodes_data(retrieved_nodes, knowledge_graph)
        
        # Sort by relevance and type
        sorted_nodes = self._sort_nodes(nodes_data, query_embedding)
        
        # Group by type
        grouped_nodes = self._group_nodes(sorted_nodes)
        
        # Assemble context while respecting token limit
        context = self._assemble_with_token_limit(grouped_nodes, max_tokens)
        
        return context
    
    def _get_nodes_data(self, nodes: List[str], graph: nx.DiGraph) -> List[Dict]:
        """Get full data for retrieved nodes."""
        nodes_data = []
        for node in nodes:
            data = graph.nodes[node]
            data["id"] = node
            nodes_data.append(data)
        return nodes_data
    
    def _sort_nodes(self, nodes_data: List[Dict], query_embedding: np.ndarray) -> List[Dict]:
        """Sort nodes by relevance and type priority."""
        # Calculate relevance scores
        for node in nodes_data:
            if "embedding" in node:
                node["relevance_score"] = cosine_similarity(
                    node["embedding"], 
                    query_embedding
                )
            else:
                node["relevance_score"] = 0.0
                
        # Define type priorities
        type_priority = {
            "text": 1,
            "table": 2,
            "figure": 2,
            "entity": 3
        }
        
        # Sort by type priority and relevance
        return sorted(
            nodes_data,
            key=lambda x: (
                type_priority.get(x.get("type", ""), 999),
                -x["relevance_score"]
            )
        )
    
    def _group_nodes(self, nodes: List[Dict]) -> Dict[str, List[Dict]]:
        """Group nodes by their type."""
        grouped = {
            "text": [],
            "table": [],
            "figure": [],
            "entity": []
        }
        
        for node in nodes:
            node_type = node.get("type", "text")
            if node_type in grouped:
                grouped[node_type].append(node)
                
        return grouped
    
    def _assemble_with_token_limit(self, 
                                 grouped_nodes: Dict[str, List[Dict]], 
                                 max_tokens: int) -> Dict:
        """Assemble context while respecting token limit."""
        context = {
            "text": [],
            "tables": [],
            "figures": [],
            "entities": [],
            "total_tokens": 0
        }
        
        # Add text chunks first
        remaining_tokens = max_tokens
        for node in grouped_nodes["text"]:
            tokens = self._count_tokens(node["content"])
            if remaining_tokens >= tokens:
                context["text"].append(node["content"])
                remaining_tokens -= tokens
            else:
                break
                
        # Add tables and figures
        for table in grouped_nodes["table"][:2]:  # Limit to 2 most relevant tables
            context["tables"].append({
                "content": table["content"],
                "caption": table.get("caption", "")
            })
            
        for figure in grouped_nodes["figure"][:2]:  # Limit to 2 most relevant figures
            context["figures"].append({
                "image": figure["image"],
                "caption": figure.get("caption", "")
            })
            
        # Add relevant entities
        context["entities"] = [
            {"text": node["text"], "type": node["type"]}
            for node in grouped_nodes["entity"][:5]  # Limit to 5 most relevant entities
        ]
        
        context["total_tokens"] = max_tokens - remaining_tokens
        return context
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using model's tokenizer."""
        return len(self.tokenizer.encode(text))