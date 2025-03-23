"""Spatial indexing for VSA-OGM."""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class AdaptiveSpatialIndex:
    """
    Adaptive grid-based spatial index for efficient point cloud processing.
    
    This class implements an adaptive grid-based spatial index that adjusts
    cell size based on point density for efficient range queries and provides
    functionality for region safety checking.
    """
    
    def __init__(
        self, 
        points: torch.Tensor, 
        labels: torch.Tensor, 
        min_resolution: float, 
        max_resolution: float, 
        device: torch.device
    ):
        """
        Initialize an adaptive grid-based spatial index.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
            min_resolution: Minimum resolution of the grid cells
            max_resolution: Maximum resolution of the grid cells
            device: Device to store tensors on
        """
        self.device = device
        self.points = points
        self.labels = labels
        
        # Handle empty point cloud
        if points.shape[0] == 0:
            self.cell_size = min_resolution
            self.grid = {}
            self.cell_indices = torch.zeros((0, 2), device=device).long()
            return
        
        # Determine optimal cell size based on point density
        self.cell_size = self._optimize_cell_size(points, min_resolution, max_resolution)
        
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float().to(device)
        
        if points.device != device:
            points = points.to(device)
        
        # Compute grid cell indices for each point
        self.cell_indices = torch.floor(points / self.cell_size).long()
        
        # Create dictionary mapping from cell indices to point indices
        self.grid = {}
        for i, (x, y) in enumerate(self.cell_indices):
            key = (x.item(), y.item())
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(i)
    
    def _optimize_cell_size(self, points: torch.Tensor, min_resolution: float, max_resolution: float) -> float:
        """
        Determine optimal cell size based on point distribution.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            min_resolution: Minimum resolution of the grid cells
            max_resolution: Maximum resolution of the grid cells
            
        Returns:
            Appropriate cell size
        """
        # Calculate point density
        if isinstance(points, torch.Tensor):
            x_min, y_min = points.min(dim=0).values
            x_max, y_max = points.max(dim=0).values
            x_range = x_max - x_min
            y_range = y_max - y_min
        else:  # numpy array
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x_range = x_max - x_min
            y_range = y_max - y_min
        
        area = x_range * y_range
        point_density = points.shape[0] / area if area > 0 else 1.0
        
        # Calculate adaptive cell size based on point density
        # Higher density -> smaller cells, lower density -> larger cells
        # We aim for approximately 10-50 points per cell on average
        target_points_per_cell = 25
        point_density_tensor = torch.tensor(point_density, device=self.device)
        ideal_cell_size = torch.sqrt(target_points_per_cell / point_density_tensor)
        
        # Clamp to min/max resolution
        cell_size = torch.clamp(ideal_cell_size, min=min_resolution, max=max_resolution)
        
        return cell_size.item()
    
    def query_range(self, center: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find all points within a given radius of center using squared distances.
        
        Args:
            center: [x, y] coordinates of query center
            radius: Search radius
            
        Returns:
            Tuple of (points, labels) tensors for points within the radius
        """
        # Convert center to tensor if it's not already
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, device=self.device)
        
        # Calculate cell range to search (using squared distances)
        squared_radius = radius * radius
        radius_cells = int(radius / self.cell_size) + 1
        center_cell = torch.floor(center / self.cell_size).long()
        
        # Collect point indices from relevant cells
        indices = []
        for i in range(-radius_cells, radius_cells + 1):
            for j in range(-radius_cells, radius_cells + 1):
                key = (center_cell[0].item() + i, center_cell[1].item() + j)
                if key in self.grid:
                    indices.extend(self.grid[key])
        
        if not indices:
            return torch.zeros((0, 2), device=self.device), torch.zeros(0, device=self.device)
        
        # Get candidate points
        candidate_indices = torch.tensor(indices, device=self.device)
        candidate_points = self.points[candidate_indices]
        
        # Compute squared distances efficiently
        squared_diffs = candidate_points - center.unsqueeze(0)
        squared_distances = torch.sum(squared_diffs * squared_diffs, dim=1)
        
        # Filter points within radius using squared distance
        mask = squared_distances <= squared_radius
        result_indices = candidate_indices[mask]
        
        return self.points[result_indices], self.labels[result_indices]
    
    def is_region_free(self, bounds: List[float], safety_margin: float) -> bool:
        """
        Check if a region is free of occupied points with a safety margin.
        
        Args:
            bounds: Region bounds [x_min, x_max, y_min, y_max]
            safety_margin: Minimum distance from occupied points
            
        Returns:
            True if region is free, False otherwise
        """
        # Expand bounds by safety margin
        expanded_bounds = [
            bounds[0] - safety_margin,
            bounds[1] + safety_margin,
            bounds[2] - safety_margin,
            bounds[3] + safety_margin
        ]
        
        # Find all cells that intersect with the expanded bounds
        min_cell_x = int(expanded_bounds[0] / self.cell_size)
        max_cell_x = int(expanded_bounds[1] / self.cell_size) + 1
        min_cell_y = int(expanded_bounds[2] / self.cell_size)
        max_cell_y = int(expanded_bounds[3] / self.cell_size) + 1
        
        # Check all cells in the expanded bounds
        for cell_x in range(min_cell_x, max_cell_x):
            for cell_y in range(min_cell_y, max_cell_y):
                key = (cell_x, cell_y)
                if key in self.grid:
                    # Check if any occupied points are within safety margin
                    for idx in self.grid[key]:
                        if self.labels[idx] == 1:  # Occupied point
                            point = self.points[idx]
                            
                            # Calculate minimum distance to bounds
                            dx = max(bounds[0] - point[0], 0, point[0] - bounds[1])
                            dy = max(bounds[2] - point[1], 0, point[1] - bounds[3])
                            dist = (dx**2 + dy**2)**0.5
                            
                            if dist < safety_margin:
                                return False
    
        return True
