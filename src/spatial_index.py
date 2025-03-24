"""
Spatial indexing for efficient point cloud queries.

This module provides a spatial indexing system for efficient range queries on point clouds.
It uses a grid-based approach to quickly find points within a specified distance.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Union


class SpatialIndex:
    """
    Grid-based spatial index for efficient range queries on point clouds.
    
    This class provides methods to efficiently find points within a specified range
    by dividing the space into a grid and only checking points in relevant grid cells.
    """
    
    def __init__(self, 
                 points: Union[np.ndarray, torch.Tensor], 
                 cell_size: float = 1.0,
                 device: str = "cpu") -> None:
        """
        Initialize the spatial index with a set of points.
        
        Args:
            points: Point cloud as array of shape (N, D) where N is number of points
                   and D is dimensionality (typically 2 or 3)
            cell_size: Size of each grid cell (smaller cells = faster queries but more memory)
            device: Device to store tensors on ("cpu" or "cuda")
        """
        # Convert points to torch tensor if needed
        if isinstance(points, np.ndarray):
            self.points = torch.from_numpy(points).to(device)
        else:
            self.points = points.to(device)
            
        self.device = device
        self.cell_size = cell_size
        self.dimensionality = self.points.shape[1]
        
        # Compute bounds
        self.min_bounds = torch.min(self.points, dim=0)[0]
        self.max_bounds = torch.max(self.points, dim=0)[0]
        
        # Create grid
        self._build_grid()
        
    def _build_grid(self) -> None:
        """Build the spatial grid index."""
        # Calculate grid dimensions
        grid_size = torch.ceil((self.max_bounds - self.min_bounds) / self.cell_size).long()
        self.grid_size = grid_size
        
        # Calculate cell indices for each point
        cell_indices = torch.floor((self.points - self.min_bounds) / self.cell_size).long()
        
        # Ensure indices are within bounds
        for d in range(self.dimensionality):
            cell_indices[:, d] = torch.clamp(cell_indices[:, d], 0, grid_size[d] - 1)
        
        # Create a unique cell ID for each point
        if self.dimensionality == 2:
            self.point_cell_ids = cell_indices[:, 0] * grid_size[1] + cell_indices[:, 1]
        elif self.dimensionality == 3:
            self.point_cell_ids = (cell_indices[:, 0] * grid_size[1] + cell_indices[:, 1]) * grid_size[2] + cell_indices[:, 2]
        else:
            # For higher dimensions, use a more general approach
            multiplier = torch.ones(self.dimensionality, device=self.device, dtype=torch.long)
            for d in range(1, self.dimensionality):
                multiplier[d] = multiplier[d-1] * grid_size[d-1]
            self.point_cell_ids = torch.sum(cell_indices * multiplier, dim=1)
        
        # Create a dictionary-like structure mapping cell IDs to point indices
        unique_cell_ids, inverse_indices = torch.unique(self.point_cell_ids, return_inverse=True)
        
        self.cell_to_points = {}
        for i, cell_id in enumerate(unique_cell_ids):
            self.cell_to_points[cell_id.item()] = torch.where(self.point_cell_ids == cell_id)[0]
    
    def range_query(self, 
                    center: Union[np.ndarray, torch.Tensor, List[float], Tuple[float, ...]], 
                    radius: float) -> torch.Tensor:
        """
        Find all points within a specified radius of a center point.
        
        Args:
            center: Center point of the query
            radius: Radius of the query
            
        Returns:
            Tensor of indices of points within the radius
        """
        # Convert center to tensor if needed
        if isinstance(center, (list, tuple, np.ndarray)):
            center = torch.tensor(center, device=self.device, dtype=self.points.dtype)
        
        # Calculate the grid cells that need to be checked
        min_cell = torch.floor((center - radius - self.min_bounds) / self.cell_size).long()
        max_cell = torch.ceil((center + radius - self.min_bounds) / self.cell_size).long()
        
        # Ensure cell indices are within bounds
        for d in range(self.dimensionality):
            min_cell[d] = torch.clamp(min_cell[d], 0, self.grid_size[d] - 1)
            max_cell[d] = torch.clamp(max_cell[d], 0, self.grid_size[d] - 1)
        
        # Collect points from all cells in range
        candidate_indices = []
        
        # For 2D case, optimize the cell iteration
        if self.dimensionality == 2:
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    cell_id = i * self.grid_size[1] + j
                    if cell_id.item() in self.cell_to_points:
                        candidate_indices.append(self.cell_to_points[cell_id.item()])
        # For 3D case
        elif self.dimensionality == 3:
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    for k in range(min_cell[2], max_cell[2] + 1):
                        cell_id = (i * self.grid_size[1] + j) * self.grid_size[2] + k
                        if cell_id.item() in self.cell_to_points:
                            candidate_indices.append(self.cell_to_points[cell_id.item()])
        # General case for higher dimensions (less efficient)
        else:
            # This is a simplified approach for higher dimensions
            # A more efficient implementation would use recursive iteration
            cell_ranges = [range(min_cell[d], max_cell[d] + 1) for d in range(self.dimensionality)]
            from itertools import product
            for cell_idx in product(*cell_ranges):
                cell_id = 0
                for d, idx in enumerate(cell_idx):
                    multiplier = 1
                    for d2 in range(d):
                        multiplier *= self.grid_size[d2]
                    cell_id += idx * multiplier
                if cell_id in self.cell_to_points:
                    candidate_indices.append(self.cell_to_points[cell_id])
        
        if not candidate_indices:
            return torch.tensor([], device=self.device, dtype=torch.long)
            
        # Combine all candidate indices
        if len(candidate_indices) == 1:
            candidates = candidate_indices[0]
        else:
            candidates = torch.cat(candidate_indices)
        
        # Calculate distances to center
        distances = torch.sqrt(torch.sum((self.points[candidates] - center) ** 2, dim=1))
        
        # Filter by radius
        within_radius = candidates[distances <= radius]
        
        return within_radius
    
    def k_nearest(self, 
                 center: Union[np.ndarray, torch.Tensor, List[float], Tuple[float, ...]], 
                 k: int = 1) -> torch.Tensor:
        """
        Find the k nearest points to a center point.
        
        Args:
            center: Center point of the query
            k: Number of nearest neighbors to find
            
        Returns:
            Tensor of indices of the k nearest points
        """
        # Convert center to tensor if needed
        if isinstance(center, (list, tuple, np.ndarray)):
            center = torch.tensor(center, device=self.device, dtype=self.points.dtype)
        
        # Start with a small radius and expand until we find at least k points
        radius = self.cell_size
        indices = self.range_query(center, radius)
        
        while indices.shape[0] < k and radius < torch.max(self.max_bounds - self.min_bounds):
            radius *= 2
            indices = self.range_query(center, radius)
        
        # If we found more than k points, select the k closest
        if indices.shape[0] > k:
            distances = torch.sqrt(torch.sum((self.points[indices] - center) ** 2, dim=1))
            _, idx = torch.topk(distances, k, largest=False)
            indices = indices[idx]
        
        return indices
    
    def update_points(self, 
                      points: Union[np.ndarray, torch.Tensor], 
                      rebuild: bool = True) -> None:
        """
        Update the point cloud and optionally rebuild the spatial index.
        
        Args:
            points: New point cloud
            rebuild: Whether to rebuild the spatial index immediately
        """
        # Convert points to torch tensor if needed
        if isinstance(points, np.ndarray):
            self.points = torch.from_numpy(points).to(self.device)
        else:
            self.points = points.to(self.device)
        
        # Rebuild the grid if requested
        if rebuild:
            self._build_grid()
    
    def get_points_in_quadrant(self, 
                              min_bounds: Union[np.ndarray, torch.Tensor, List[float], Tuple[float, ...]], 
                              max_bounds: Union[np.ndarray, torch.Tensor, List[float], Tuple[float, ...]]) -> torch.Tensor:
        """
        Get all points within a specified quadrant (rectangular region).
        
        Args:
            min_bounds: Minimum bounds of the quadrant (lower-left corner)
            max_bounds: Maximum bounds of the quadrant (upper-right corner)
            
        Returns:
            Tensor of indices of points within the quadrant
        """
        # Convert bounds to tensors if needed
        if isinstance(min_bounds, (list, tuple, np.ndarray)):
            min_bounds = torch.tensor(min_bounds, device=self.device, dtype=self.points.dtype)
        if isinstance(max_bounds, (list, tuple, np.ndarray)):
            max_bounds = torch.tensor(max_bounds, device=self.device, dtype=self.points.dtype)
        
        # Calculate the grid cells that need to be checked
        min_cell = torch.floor((min_bounds - self.min_bounds) / self.cell_size).long()
        max_cell = torch.ceil((max_bounds - self.min_bounds) / self.cell_size).long()
        
        # Ensure cell indices are within bounds
        for d in range(self.dimensionality):
            min_cell[d] = torch.clamp(min_cell[d], 0, self.grid_size[d] - 1)
            max_cell[d] = torch.clamp(max_cell[d], 0, self.grid_size[d] - 1)
        
        # Collect points from all cells in range
        candidate_indices = []
        
        # For 2D case, optimize the cell iteration
        if self.dimensionality == 2:
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    cell_id = i * self.grid_size[1] + j
                    if cell_id.item() in self.cell_to_points:
                        candidate_indices.append(self.cell_to_points[cell_id.item()])
        # For 3D case
        elif self.dimensionality == 3:
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    for k in range(min_cell[2], max_cell[2] + 1):
                        cell_id = (i * self.grid_size[1] + j) * self.grid_size[2] + k
                        if cell_id.item() in self.cell_to_points:
                            candidate_indices.append(self.cell_to_points[cell_id.item()])
        
        if not candidate_indices:
            return torch.tensor([], device=self.device, dtype=torch.long)
            
        # Combine all candidate indices
        if len(candidate_indices) == 1:
            candidates = candidate_indices[0]
        else:
            candidates = torch.cat(candidate_indices)
        
        # Filter points to ensure they're actually within the quadrant
        mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=self.device)
        for d in range(self.dimensionality):
            mask &= (self.points[candidates, d] >= min_bounds[d])
            mask &= (self.points[candidates, d] <= max_bounds[d])
        
        return candidates[mask]
