"""
Sequential processing of point clouds.

This module implements sequential processing of point clouds by sampling observation
points on a grid and simulating sensor readings at each location.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Union, Optional, Callable
from tqdm import tqdm

from src.spatial_index import SpatialIndex


class SequentialProcessor:
    """
    Sequential processor for point clouds.
    
    This class processes point clouds sequentially by sampling observation points
    on a grid and simulating sensor readings at each location.
    """
    
    def __init__(self, 
                 world_bounds: Tuple[float, float, float, float],
                 sample_resolution: float = 1.0,
                 sensor_range: float = 10.0,
                 device: str = "cpu") -> None:
        """
        Initialize the sequential processor.
        
        Args:
            world_bounds: Tuple of (xmin, xmax, ymin, ymax) defining the world boundaries
            sample_resolution: Distance between sample points on the grid
            sensor_range: Maximum range of the simulated sensor
            device: Device to perform calculations on ("cpu" or "cuda")
        """
        self.world_bounds = world_bounds
        self.sample_resolution = sample_resolution
        self.sensor_range = sensor_range
        self.device = device
        
        # Calculate world dimensions
        self.world_width = world_bounds[1] - world_bounds[0]
        self.world_height = world_bounds[3] - world_bounds[2]
        
        # Generate grid of sample positions
        self._generate_sample_positions()
        
    def _generate_sample_positions(self) -> None:
        """Generate a grid of sample positions."""
        # Create grid of sample positions
        x_coords = torch.arange(
            self.world_bounds[0] + self.sample_resolution / 2, 
            self.world_bounds[1], 
            self.sample_resolution, 
            device=self.device
        )
        
        y_coords = torch.arange(
            self.world_bounds[2] + self.sample_resolution / 2, 
            self.world_bounds[3], 
            self.sample_resolution, 
            device=self.device
        )
        
        # Create meshgrid
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Flatten grid for processing
        self.sample_positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
    def process_point_cloud(self, 
                           points: Union[np.ndarray, torch.Tensor],
                           labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           process_fn: Callable = None,
                           show_progress: bool = True) -> None:
        """
        Process a point cloud sequentially.
        
        Args:
            points: Point cloud as array of shape (N, D) where N is number of points
                   and D is dimensionality (typically 2)
            labels: Optional labels for each point (1=occupied, 0=empty)
            process_fn: Function to call for each sample position with visible points
            show_progress: Whether to show a progress bar
        """
        # Convert points to tensor if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(self.device)
            
        # Convert labels to tensor if needed
        if labels is not None and isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(self.device)
            
        # Create spatial index for efficient range queries
        spatial_index = SpatialIndex(points, cell_size=self.sensor_range/2, device=self.device)
        
        # Process each sample position
        iterator = tqdm(self.sample_positions) if show_progress else self.sample_positions
        
        for sample_pos in iterator:
            # Find points within sensor range
            visible_indices = spatial_index.range_query(sample_pos, self.sensor_range)
            
            if len(visible_indices) > 0:
                visible_points = points[visible_indices]
                
                # If labels are provided, get labels for visible points
                visible_labels = None
                if labels is not None:
                    visible_labels = labels[visible_indices]
                
                # Call process function if provided
                if process_fn is not None:
                    process_fn(sample_pos, visible_points, visible_labels)
    
    def simulate_sensor_readings(self, 
                               points: Union[np.ndarray, torch.Tensor],
                               sample_position: Union[np.ndarray, torch.Tensor, Tuple[float, float]],
                               num_rays: int = 360,
                               noise_std: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Simulate sensor readings from a sample position.
        
        Args:
            points: Point cloud as array of shape (N, D)
            sample_position: Position from which to simulate sensor readings
            num_rays: Number of rays to cast (e.g., 360 for 1-degree resolution)
            noise_std: Standard deviation of Gaussian noise to add to readings
            
        Returns:
            Dictionary with ray angles and distances
        """
        # Convert points and sample position to tensor if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(self.device)
        if isinstance(sample_position, (tuple, list, np.ndarray)):
            sample_position = torch.tensor(sample_position, device=self.device)
            
        # Create spatial index for efficient range queries
        spatial_index = SpatialIndex(points, cell_size=self.sensor_range/2, device=self.device)
        
        # Find points within sensor range
        visible_indices = spatial_index.range_query(sample_position, self.sensor_range)
        
        if len(visible_indices) == 0:
            # No points visible, return maximum range for all rays
            angles = torch.linspace(0, 2*np.pi, num_rays, device=self.device)
            distances = torch.ones(num_rays, device=self.device) * self.sensor_range
            return {'angles': angles, 'distances': distances}
        
        visible_points = points[visible_indices]
        
        # Calculate vectors from sample position to visible points
        vectors = visible_points - sample_position
        
        # Calculate distances and angles to visible points
        distances = torch.sqrt(torch.sum(vectors**2, dim=1))
        angles = torch.atan2(vectors[:, 1], vectors[:, 0])
        
        # Normalize angles to [0, 2π)
        angles = (angles + 2*np.pi) % (2*np.pi)
        
        # Initialize ray distances with maximum range
        ray_angles = torch.linspace(0, 2*np.pi, num_rays, device=self.device)
        ray_distances = torch.ones(num_rays, device=self.device) * self.sensor_range
        
        # Find closest point for each ray
        for i, ray_angle in enumerate(ray_angles):
            # Find points close to this ray angle
            angle_diff = torch.abs(angles - ray_angle)
            angle_diff = torch.min(angle_diff, 2*np.pi - angle_diff)  # Handle wraparound
            
            # Consider points within a small angle threshold
            angle_threshold = np.pi / num_rays
            close_points = angle_diff < angle_threshold
            
            if torch.any(close_points):
                # Find closest point
                closest_idx = torch.argmin(distances[close_points])
                ray_distances[i] = distances[close_points][closest_idx]
                
        # Add noise if specified
        if noise_std > 0:
            ray_distances += torch.randn_like(ray_distances) * noise_std
            ray_distances = torch.clamp(ray_distances, 0, self.sensor_range)
            
        return {'angles': ray_angles, 'distances': ray_distances}
    
    def generate_observation_points(self, 
                                   custom_resolution: Optional[float] = None,
                                   custom_bounds: Optional[Tuple[float, float, float, float]] = None) -> torch.Tensor:
        """
        Generate a custom grid of observation points.
        
        Args:
            custom_resolution: Optional custom resolution for the grid
            custom_bounds: Optional custom bounds for the grid
            
        Returns:
            Tensor of shape (N, 2) with observation points
        """
        # Use custom or default resolution
        resolution = custom_resolution if custom_resolution is not None else self.sample_resolution
        
        # Use custom or default bounds
        bounds = custom_bounds if custom_bounds is not None else self.world_bounds
        
        # Create grid of sample positions
        x_coords = torch.arange(
            bounds[0] + resolution / 2, 
            bounds[1], 
            resolution, 
            device=self.device
        )
        
        y_coords = torch.arange(
            bounds[2] + resolution / 2, 
            bounds[3], 
            resolution, 
            device=self.device
        )
        
        # Create meshgrid
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Flatten grid for processing
        return torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    def get_sample_positions(self) -> torch.Tensor:
        """
        Get the grid of sample positions.
        
        Returns:
            Tensor of shape (N, 2) with sample positions
        """
        return self.sample_positions
    
    def points_to_labels(self, 
                        points: Union[np.ndarray, torch.Tensor],
                        occupied_indices: Union[np.ndarray, torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Convert point indices to binary labels.
        
        Args:
            points: Point cloud as array of shape (N, D)
            occupied_indices: Indices of points that are occupied
            
        Returns:
            Binary labels (1=occupied, 0=empty) for each point
        """
        # Convert points to tensor if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(self.device)
            
        # Convert occupied_indices to tensor if needed
        if isinstance(occupied_indices, (list, np.ndarray)):
            occupied_indices = torch.tensor(occupied_indices, device=self.device)
            
        # Initialize all labels as empty (0)
        labels = torch.zeros(points.shape[0], device=self.device)
        
        # Set occupied points to 1
        labels[occupied_indices] = 1
        
        return labels
