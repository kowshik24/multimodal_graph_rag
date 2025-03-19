import torch
import numpy as np
from typing import Union, List

def normalize_embedding(embedding: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Normalize embedding vector to unit length."""
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    return embedding / np.linalg.norm(embedding)

def combine_embeddings(embeddings: List[np.ndarray], 
                      weights: List[float] = None) -> np.ndarray:
    """Combine multiple embeddings with optional weights."""
    if weights is None:
        weights = [1.0] * len(embeddings)
    
    weighted_embeddings = [w * e for w, e in zip(weights, embeddings)]
    combined = np.sum(weighted_embeddings, axis=0)
    return normalize_embedding(combined)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))