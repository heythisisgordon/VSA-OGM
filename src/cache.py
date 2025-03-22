"""Vector caching for VSA-OGM."""

import torch
from typing import Dict, Tuple, List, Any, Optional

from .functional import power, bind_batch

class VectorCache:
    """
    Cache for VSA vectors to avoid redundant computation.
    
    This class implements a cache for VSA vectors to avoid redundant computation
    of the same vectors, improving performance for repeated operations.
    """
    
    def __init__(
        self, 
        xy_axis_vectors: torch.Tensor, 
        length_scale: float, 
        device: torch.device, 
        grid_resolution: float = 0.1, 
        max_size: int = 10000
    ):
        """
        Initialize the vector cache.
        
        Args:
            xy_axis_vectors: Axis vectors for VSA operations
            length_scale: Length scale for power operation
            device: Device to store tensors on
            grid_resolution: Resolution for discretizing points for caching
            max_size: Maximum number of vectors to cache
        """
        self.xy_axis_vectors = xy_axis_vectors
        self.length_scale = length_scale
        self.device = device
        self.grid_resolution = grid_resolution
        self.max_size = max_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_batch_vectors(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get or compute VSA vectors for a batch of points in parallel.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            
        Returns:
            Tensor of shape [N, vsa_dimensions] containing VSA vectors
        """
        # Discretize points to grid resolution for caching
        keys = torch.floor(points / self.grid_resolution).long()
        
        # Initialize result tensor
        result = torch.zeros((points.shape[0], self.xy_axis_vectors[0].shape[0]), 
                            device=self.device)
        
        # Identify points not in cache
        missing_indices = []
        
        for i, key in enumerate(keys):
            key_tuple = (key[0].item(), key[1].item())
            if key_tuple in self.cache:
                result[i] = self.cache[key_tuple]
                self.cache_hits += 1
            else:
                missing_indices.append(i)
                self.cache_misses += 1
        
        if missing_indices:
            # Compute missing vectors in parallel
            missing_points = points[missing_indices]
            
            # Compute x and y vectors for all missing points at once
            x_vectors = power(self.xy_axis_vectors[0], missing_points[:, 0], self.length_scale)
            y_vectors = power(self.xy_axis_vectors[1], missing_points[:, 1], self.length_scale)
            
            # Bind vectors in batch
            missing_vectors = bind_batch([x_vectors, y_vectors], self.device)
            
            # Update cache and result
            for idx, i in enumerate(missing_indices):
                key_tuple = (keys[i][0].item(), keys[i][1].item())
                
                # Simple cache management: if cache is full, don't add new items
                if len(self.cache) < self.max_size:
                    self.cache[key_tuple] = missing_vectors[idx]
                
                result[i] = missing_vectors[idx]
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }
    
    def clear(self) -> None:
        """Clear the cache to free memory"""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
