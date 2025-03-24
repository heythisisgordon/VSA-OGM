"""
Main VSA-OGM mapper implementation.

This module provides the main VSAMapper class, which serves as the entry point
for the Sequential VSA-OGM system and coordinates interactions between components.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Union, Optional, Any

from src.config import Config
from src.quadrant_memory import QuadrantMemory
from src.sequential_processor import SequentialProcessor
from src.entropy import EntropyExtractor
from src.spatial_index import SpatialIndex


class VSAMapper:
    """
    Main entry point for the Sequential VSA-OGM system.
    
    This class coordinates the interactions between the different components
    of the system, providing a high-level API for processing point clouds and
    generating occupancy grid maps.
    """
    
    def __init__(self, 
                 world_bounds: Tuple[float, float, float, float],
                 config: Optional[Union[Dict[str, Any], Config]] = None) -> None:
        """
        Initialize the VSA mapper.
        
        Args:
            world_bounds: Tuple of (xmin, xmax, ymin, ymax) defining the world boundaries
            config: Optional configuration dictionary or Config object
        """
        # Initialize configuration
        if config is None:
            self.config = Config()
        elif isinstance(config, dict):
            self.config = Config(config)
        else:
            self.config = config
            
        self.world_bounds = world_bounds
        self.device = self.config.get("system", "device")
        
        # Initialize components
        self._init_components()
        
        # Initialize state variables
        self.points = None
        self.labels = None
        self.occupancy_grid = None
        self.entropy_grid = None
        self.classification = None
        
    def _init_components(self) -> None:
        """Initialize system components."""
        # Initialize quadrant memory
        self.quadrant_memory = QuadrantMemory(
            world_bounds=self.world_bounds,
            quadrant_size=self.config.get("quadrant", "size"),
            vector_dim=self.config.get("vsa", "dimensions"),
            length_scale=self.config.get("vsa", "length_scale"),
            device=self.device
        )
        
        # Initialize sequential processor
        self.sequential_processor = SequentialProcessor(
            world_bounds=self.world_bounds,
            sample_resolution=self.config.get("sequential", "sample_resolution"),
            sensor_range=self.config.get("sequential", "sensor_range"),
            device=self.device
        )
        
        # Initialize entropy extractor
        self.entropy_extractor = EntropyExtractor(
            disk_radius=self.config.get("entropy", "disk_radius"),
            occupied_threshold=self.config.get("entropy", "occupied_threshold"),
            empty_threshold=self.config.get("entropy", "empty_threshold"),
            device=self.device
        )
        
    def process_point_cloud(self, 
                           points: Union[np.ndarray, torch.Tensor],
                           labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """
        Process a full point cloud in sequential manner.
        
        Args:
            points: Point cloud as array of shape (N, D) where N is number of points
                   and D is dimensionality (typically 2)
            labels: Optional labels for each point (1=occupied, 0=empty)
        """
        # Store points and labels
        self.points = points
        self.labels = labels
        
        # If labels are not provided, assume all points are occupied
        if labels is None:
            if isinstance(points, np.ndarray):
                self.labels = np.ones(points.shape[0], dtype=np.int32)
            else:
                self.labels = torch.ones(points.shape[0], dtype=torch.int32, device=self.device)
        
        # Process point cloud sequentially
        self.process_incrementally()
        
    def process_incrementally(self, 
                             sample_resolution: Optional[float] = None,
                             sensor_range: Optional[float] = None) -> None:
        """
        Process the point cloud incrementally from sample positions.
        
        Args:
            sample_resolution: Optional custom resolution for sampling (overrides config)
            sensor_range: Optional custom sensor range (overrides config)
        """
        if self.points is None:
            raise ValueError("No point cloud to process. Call process_point_cloud first.")
            
        # Use custom or config values
        resolution = sample_resolution if sample_resolution is not None else self.config.get("sequential", "sample_resolution")
        range_val = sensor_range if sensor_range is not None else self.config.get("sequential", "sensor_range")
        
        # Update sequential processor if needed
        if (sample_resolution is not None or sensor_range is not None):
            self.sequential_processor = SequentialProcessor(
                world_bounds=self.world_bounds,
                sample_resolution=resolution,
                sensor_range=range_val,
                device=self.device
            )
        
        # Define processing function for each sample position
        def process_fn(sample_pos, visible_points, visible_labels):
            self.quadrant_memory.update_with_points(visible_points, visible_labels)
            
        # Process point cloud sequentially
        self.sequential_processor.process_point_cloud(
            self.points,
            self.labels,
            process_fn=process_fn,
            show_progress=self.config.get("system", "show_progress")
        )
        
        # Normalize memory vectors
        self.quadrant_memory.normalize_memories()
        
        # Generate maps
        self._generate_maps()
        
    def _generate_maps(self, resolution: Optional[float] = None) -> None:
        """
        Generate occupancy and entropy maps.
        
        Args:
            resolution: Optional resolution for the grid (overrides config)
        """
        # Use custom or config resolution
        res = resolution if resolution is not None else self.config.get("sequential", "sample_resolution")
        
        # Query grid from quadrant memory
        grid_results = self.quadrant_memory.query_grid(res)
        
        # Apply Born rule to convert similarity scores to probabilities
        occupied_probs = self.entropy_extractor.apply_born_rule(grid_results['occupied'])
        empty_probs = self.entropy_extractor.apply_born_rule(grid_results['empty'])
        
        # Extract features using entropy
        features = self.entropy_extractor.extract_features(occupied_probs, empty_probs)
        
        # Store results
        self.entropy_grid = features['global_entropy']
        self.classification = features['classification']
        self.occupancy_grid = self.entropy_extractor.get_occupancy_grid(self.classification)
        
        # Store grid coordinates for visualization
        self.grid_coords = {
            'x': grid_results['x_coords'],
            'y': grid_results['y_coords']
        }
        
    def get_occupancy_grid(self) -> Dict[str, torch.Tensor]:
        """
        Get the current occupancy grid.
        
        Returns:
            Dictionary with occupancy grid and coordinates
        """
        if self.occupancy_grid is None:
            raise ValueError("No occupancy grid available. Process a point cloud first.")
            
        return {
            'grid': self.occupancy_grid,
            'x_coords': self.grid_coords['x'],
            'y_coords': self.grid_coords['y']
        }
        
    def get_entropy_grid(self) -> Dict[str, torch.Tensor]:
        """
        Get the entropy grid for visualization.
        
        Returns:
            Dictionary with entropy grid and coordinates
        """
        if self.entropy_grid is None:
            raise ValueError("No entropy grid available. Process a point cloud first.")
            
        return {
            'grid': self.entropy_grid,
            'x_coords': self.grid_coords['x'],
            'y_coords': self.grid_coords['y']
        }
        
    def get_classification(self) -> Dict[str, torch.Tensor]:
        """
        Get the classification grid.
        
        Returns:
            Dictionary with classification grid and coordinates
        """
        if self.classification is None:
            raise ValueError("No classification available. Process a point cloud first.")
            
        return {
            'grid': self.classification,
            'x_coords': self.grid_coords['x'],
            'y_coords': self.grid_coords['y']
        }
        
    def query_point(self, 
                   point: Union[np.ndarray, torch.Tensor, Tuple[float, float]]) -> Dict[str, torch.Tensor]:
        """
        Query the memory for a specific point.
        
        Args:
            point: The point coordinates (x, y)
            
        Returns:
            Dictionary with similarity scores and classification
        """
        # Query quadrant memory
        result = self.quadrant_memory.query_point(point)
        
        # Apply Born rule to convert similarity scores to probabilities
        occupied_prob = self.entropy_extractor.apply_born_rule(result['occupied'])
        empty_prob = self.entropy_extractor.apply_born_rule(result['empty'])
        
        # Calculate global entropy
        global_entropy = occupied_prob - empty_prob
        
        # Classify point
        if global_entropy > self.config.get("entropy", "occupied_threshold"):
            classification = 1  # Occupied
        elif global_entropy < -self.config.get("entropy", "empty_threshold"):
            classification = -1  # Empty
        else:
            classification = 0  # Unknown
            
        return {
            'occupied_similarity': result['occupied'],
            'empty_similarity': result['empty'],
            'occupied_probability': occupied_prob,
            'empty_probability': empty_prob,
            'global_entropy': global_entropy,
            'classification': classification
        }
        
    def save_config(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration file
        """
        self.config.save(filepath)
        
    def get_quadrant_centers(self) -> torch.Tensor:
        """
        Get the centers of all quadrants.
        
        Returns:
            Tensor of shape (num_quadrants, 2) with quadrant centers
        """
        return self.quadrant_memory.get_quadrant_centers()
    
    def get_quadrant_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the boundaries of all quadrants.
        
        Returns:
            Tuple of tensors (x_bounds, y_bounds) defining quadrant boundaries
        """
        return self.quadrant_memory.get_quadrant_bounds()
    
    def get_memory_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Get all memory vectors.
        
        Returns:
            Dictionary with occupied and empty memory vectors
        """
        return self.quadrant_memory.get_memory_vectors()
    
    def get_sample_positions(self) -> torch.Tensor:
        """
        Get the grid of sample positions.
        
        Returns:
            Tensor of shape (N, 2) with sample positions
        """
        return self.sequential_processor.get_sample_positions()
