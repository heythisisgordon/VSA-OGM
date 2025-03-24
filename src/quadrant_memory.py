"""
Quadrant-based memory system for Vector Symbolic Architecture (VSA).

This module implements a quadrant-based memory system that divides the space into
quadrants and maintains memory vectors for each quadrant. It provides methods for
updating quadrant memories with new points and retrieving similarity scores.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Union, Optional

from src.vector_ops import bind, power, normalize, make_unitary, similarity


class QuadrantMemory:
    """
    Quadrant-based memory system for Vector Symbolic Architecture (VSA).
    
    This class divides the space into quadrants and maintains memory vectors for
    each quadrant. It provides methods for updating quadrant memories with new
    points and retrieving similarity scores.
    """
    
    def __init__(self, 
                 world_bounds: Tuple[float, float, float, float],
                 quadrant_size: int,
                 vector_dim: int = 1024,
                 length_scale: float = 1.0,
                 device: str = "cpu") -> None:
        """
        Initialize the quadrant memory system.
        
        Args:
            world_bounds: Tuple of (xmin, xmax, ymin, ymax) defining the world boundaries
            quadrant_size: Number of quadrants along each axis (total quadrants = quadrant_size^2)
            vector_dim: Dimensionality of the VSA vectors
            length_scale: Length scale for fractional binding
            device: Device to perform calculations on ("cpu" or "cuda")
        """
        self.world_bounds = world_bounds
        self.quadrant_size = quadrant_size
        self.vector_dim = vector_dim
        self.length_scale = length_scale
        self.device = device
        
        # Calculate world dimensions
        self.world_width = world_bounds[1] - world_bounds[0]
        self.world_height = world_bounds[3] - world_bounds[2]
        
        # Calculate quadrant dimensions
        self.quadrant_width = self.world_width / quadrant_size
        self.quadrant_height = self.world_height / quadrant_size
        
        # Initialize axis vectors
        self._init_axis_vectors()
        
        # Build quadrant structure
        self._build_quadrants()
        
    def _init_axis_vectors(self) -> None:
        """Initialize the axis vectors for x and y dimensions."""
        self.axis_vectors = torch.zeros((2, self.vector_dim), device=self.device)
        
        # Create orthogonal basis vectors for x and y axes
        self.axis_vectors[0] = make_unitary(self.vector_dim, self.device)
        self.axis_vectors[1] = make_unitary(self.vector_dim, self.device)
        
    def _build_quadrants(self) -> None:
        """Build the quadrant structure and initialize memory vectors."""
        # Calculate quadrant boundaries
        x_bounds = torch.linspace(
            self.world_bounds[0], 
            self.world_bounds[1], 
            self.quadrant_size + 1, 
            device=self.device
        )
        
        y_bounds = torch.linspace(
            self.world_bounds[2], 
            self.world_bounds[3], 
            self.quadrant_size + 1, 
            device=self.device
        )
        
        self.quadrant_bounds = (x_bounds, y_bounds)
        
        # Calculate quadrant centers
        x_centers = (x_bounds[:-1] + x_bounds[1:]) / 2
        y_centers = (y_bounds[:-1] + y_bounds[1:]) / 2
        
        # Create grid of centers
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing='ij')
        self.quadrant_centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Initialize memory vectors for each quadrant
        total_quadrants = self.quadrant_size ** 2
        self.occupied_memory = torch.zeros((total_quadrants, self.vector_dim), device=self.device)
        self.empty_memory = torch.zeros((total_quadrants, self.vector_dim), device=self.device)
        
        # Initialize quadrant indices for fast lookup
        self.quadrant_indices = {}
        for i in range(self.quadrant_size):
            for j in range(self.quadrant_size):
                # Calculate quadrant index
                idx = i * self.quadrant_size + j
                
                # Store quadrant bounds
                self.quadrant_indices[(i, j)] = {
                    'index': idx,
                    'bounds': (
                        (x_bounds[i], x_bounds[i+1]),
                        (y_bounds[j], y_bounds[j+1])
                    ),
                    'center': (x_centers[i], y_centers[j])
                }
    
    def get_quadrant_index(self, point: Union[np.ndarray, torch.Tensor, Tuple[float, float]]) -> int:
        """
        Get the index of the quadrant containing the given point.
        
        Args:
            point: The point coordinates (x, y)
            
        Returns:
            Index of the quadrant containing the point
        """
        # Convert point to tensor if needed
        if isinstance(point, (tuple, list, np.ndarray)):
            point = torch.tensor(point, device=self.device)
            
        # Ensure point is within world bounds
        if (point[0] < self.world_bounds[0] or point[0] > self.world_bounds[1] or
            point[1] < self.world_bounds[2] or point[1] > self.world_bounds[3]):
            raise ValueError(f"Point {point} is outside world bounds {self.world_bounds}")
            
        # Calculate quadrant indices
        x_idx = int((point[0] - self.world_bounds[0]) / self.quadrant_width)
        y_idx = int((point[1] - self.world_bounds[2]) / self.quadrant_height)
        
        # Handle edge case
        if x_idx == self.quadrant_size:
            x_idx = self.quadrant_size - 1
        if y_idx == self.quadrant_size:
            y_idx = self.quadrant_size - 1
            
        # Calculate linear index
        return x_idx * self.quadrant_size + y_idx
    
    def encode_point(self, point: Union[np.ndarray, torch.Tensor, Tuple[float, float]]) -> torch.Tensor:
        """
        Encode a point into a hyperdimensional vector.
        
        Args:
            point: The point coordinates (x, y)
            
        Returns:
            Encoded vector representation of the point
        """
        # Convert point to tensor if needed
        if isinstance(point, (tuple, list, np.ndarray)):
            point = torch.tensor(point, device=self.device)
            
        # Encode each dimension
        x_vector = power(self.axis_vectors[0], point[0], self.length_scale)
        y_vector = power(self.axis_vectors[1], point[1], self.length_scale)
        
        # Bind vectors together
        return bind([x_vector, y_vector], self.device)
    
    def update_with_point(self, 
                         point: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
                         is_occupied: bool = True) -> None:
        """
        Update the memory with a new point.
        
        Args:
            point: The point coordinates (x, y)
            is_occupied: Whether the point is occupied (True) or empty (False)
        """
        # Get quadrant index
        quadrant_idx = self.get_quadrant_index(point)
        
        # Encode point
        point_vector = self.encode_point(point)
        
        # Update appropriate memory
        if is_occupied:
            self.occupied_memory[quadrant_idx] += point_vector
        else:
            self.empty_memory[quadrant_idx] += point_vector
    
    def update_with_points(self, 
                          points: Union[np.ndarray, torch.Tensor], 
                          labels: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update the memory with multiple points.
        
        Args:
            points: Array of point coordinates with shape (N, 2)
            labels: Binary labels (1=occupied, 0=empty) with shape (N,)
        """
        # Convert to tensors if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(self.device)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(self.device)
            
        # Process occupied points
        occupied_points = points[labels == 1]
        for point in occupied_points:
            self.update_with_point(point, is_occupied=True)
            
        # Process empty points
        empty_points = points[labels == 0]
        for point in empty_points:
            self.update_with_point(point, is_occupied=False)
    
    def normalize_memories(self) -> None:
        """Normalize all memory vectors to unit length."""
        # Normalize occupied memory vectors
        for i in range(len(self.occupied_memory)):
            if torch.norm(self.occupied_memory[i]) > 0:
                self.occupied_memory[i] = normalize(self.occupied_memory[i])
                
        # Normalize empty memory vectors
        for i in range(len(self.empty_memory)):
            if torch.norm(self.empty_memory[i]) > 0:
                self.empty_memory[i] = normalize(self.empty_memory[i])
    
    def query_point(self, 
                   point: Union[np.ndarray, torch.Tensor, Tuple[float, float]]) -> Dict[str, torch.Tensor]:
        """
        Query the memory for a specific point.
        
        Args:
            point: The point coordinates (x, y)
            
        Returns:
            Dictionary with similarity scores for occupied and empty memories
        """
        # Get quadrant index
        quadrant_idx = self.get_quadrant_index(point)
        
        # Encode point
        point_vector = self.encode_point(point)
        
        # Calculate similarities
        occupied_sim = similarity(point_vector, self.occupied_memory[quadrant_idx])
        empty_sim = similarity(point_vector, self.empty_memory[quadrant_idx])
        
        return {
            'occupied': occupied_sim,
            'empty': empty_sim,
            'quadrant_idx': quadrant_idx
        }
    
    def query_grid(self, 
                  resolution: float) -> Dict[str, torch.Tensor]:
        """
        Query the memory for a grid of points.
        
        Args:
            resolution: Resolution of the grid (distance between points)
            
        Returns:
            Dictionary with grid coordinates and similarity scores
        """
        # Create grid of points
        x_coords = torch.arange(
            self.world_bounds[0], 
            self.world_bounds[1], 
            resolution, 
            device=self.device
        )
        
        y_coords = torch.arange(
            self.world_bounds[2], 
            self.world_bounds[3], 
            resolution, 
            device=self.device
        )
        
        # Create meshgrid
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        grid_shape = xx.shape
        
        # Flatten grid for processing
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Initialize result arrays
        occupied_sim = torch.zeros(len(points), device=self.device)
        empty_sim = torch.zeros(len(points), device=self.device)
        
        # Query each point
        for i, point in enumerate(points):
            result = self.query_point(point)
            occupied_sim[i] = result['occupied']
            empty_sim[i] = result['empty']
        
        # Reshape results to grid
        occupied_grid = occupied_sim.reshape(grid_shape)
        empty_grid = empty_sim.reshape(grid_shape)
        
        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'occupied': occupied_grid,
            'empty': empty_grid
        }
    
    def get_quadrant_centers(self) -> torch.Tensor:
        """
        Get the centers of all quadrants.
        
        Returns:
            Tensor of shape (num_quadrants, 2) with quadrant centers
        """
        return self.quadrant_centers
    
    def get_quadrant_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the boundaries of all quadrants.
        
        Returns:
            Tuple of tensors (x_bounds, y_bounds) defining quadrant boundaries
        """
        return self.quadrant_bounds
    
    def get_memory_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Get all memory vectors.
        
        Returns:
            Dictionary with occupied and empty memory vectors
        """
        return {
            'occupied': self.occupied_memory,
            'empty': self.empty_memory
        }
