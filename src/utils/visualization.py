import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List
import plotly.graph_objects as go

class GraphVisualizer:
    def __init__(self, config):
        self.config = config
        
    def visualize_knowledge_graph(self, 
                                graph: nx.DiGraph,
                                highlight_nodes: List[str] = None,
                                output_path: str = None):
        """Visualize the knowledge graph using Plotly."""
        # Create position layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Prepare node traces
        node_types = set(nx.get_node_attributes(graph, 'type').values())
        node_traces = {}
        
        for node_type in node_types:
            nodes = [n for n, attr in graph.nodes(data=True) 
                    if attr.get('type') == node_type]
            
            x = [pos[node][0] for node in nodes]
            y = [pos[node][1] for node in nodes]
            
            node_traces[node_type] = go.Scatter(
                x=x,
                y=y,
                mode='markers+text',
                name=node_type,
                text=[self._get_node_label(graph, node) for node in nodes],
                hoverinfo='text',
                marker=dict(
                    size=20,
                    color=self._get_color_for_type(node_type)
                )
            )
            
        # Prepare edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2].get('type', ''))
            
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace] + list(node_traces.values()),
            layout=go.Layout(
                showlegend=True,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if output_path:
            fig.write_html(output_path)
        return fig
    
    def visualize_retrieval_path(self,
                               graph: nx.DiGraph,
                               path: List[str],
                               output_path: str = None):
        """Visualize the retrieval path in the knowledge graph."""
        # Create a subgraph of the retrieval path
        path_edges = list(zip(path[:-1], path[1:]))
        subgraph = graph.subgraph(path)
        
        # Create position layout
        pos = nx.spring_layout(subgraph)
        
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color='lightblue',
            node_size=1000
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=path_edges,
            edge_color='r',
            width=2
        )
        
        # Add labels
        labels = {node: self._get_node_label(graph, node) for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels)
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def _get_node_label(self, graph: nx.DiGraph, node: str) -> str:
        """Get displayable label for a node."""
        attr = graph.nodes[node]
        if 'text' in attr:
            return attr['text'][:20] + '...' if len(attr['text']) > 20 else attr['text']
        return str(node)
    
    def _get_color_for_type(self, node_type: str) -> str:
        """Get color for node type."""
        color_map = {
            'text': '#1f77b4',
            'table': '#2ca02c',
            'figure': '#ff7f0e',
            'entity': '#d62728'
        }
        return color_map.get(node_type, '#7f7f7f')