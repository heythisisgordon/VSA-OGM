"""Core VSA-OGM mapper implementation."""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import os
import time

from .functional import bind, power, SSPGenerator
from .functional import bind_batch

class VSAMapper:
    """
    Vector Symbolic Architecture Mapper for Occupancy Grid Mapping.
    
    This class implements the core VSA-OGM algorithm, converting point clouds
    to occupancy grid maps using vector symbolic operations.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the VSA mapper.
        
        Args:
            config: Configuration dictionary with the following keys:
                - world_bounds: World bounds [x_min, x_max, y_min, y_max]
                - resolution: Grid resolution in meters
                - axis_resolution: Resolution for axis vectors
                - vsa_dimensions: Dimensionality of VSA vectors
                - quadrant_hierarchy: List of quadrant hierarchy levels
                - length_scale: Length scale for power operation
                - use_query_normalization: Whether to normalize query vectors
                - decision_thresholds: Thresholds for decision making
            device: Device to use for computation
        """
        self.config = config
        # Default to CUDA when available
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract configuration parameters
        self.world_bounds = config["world_bounds"]
        self.resolution = config.get("resolution", 0.1)
        self.axis_resolution = config.get("axis_resolution", 0.5)
        self.vsa_dimensions = config.get("vsa_dimensions", 16000)
        self.quadrant_hierarchy = config.get("quadrant_hierarchy", [4])
        self.length_scale = config.get("length_scale", 2.0)
        self.use_query_normalization = config.get("use_query_normalization", True)
        self.decision_thresholds = config.get("decision_thresholds", [-0.99, 0.99])
        self.verbose = config.get("verbose", False)
        
        # Initialize core components
        self.environment_dimensionality = 2
        self.world_bounds_norm = (
            self.world_bounds[1] - self.world_bounds[0],
            self.world_bounds[3] - self.world_bounds[2]
        )
        
        # Initialize dependencies
        self.pdist = torch.nn.PairwiseDistance()
        self.ssp_generator = SSPGenerator(
            dimensionality=self.vsa_dimensions,
            device=self.device,
            length_scale=self.length_scale
        )
        
        # Initialize empty variables for class methods
        self.quadrant_axis_bounds = []
        self.quadrant_centers = []
        self.quadrant_indices_x = None
        self.quadrant_indices_y = None
        self.occupied_quadrant_memory_vectors = None
        self.empty_quadrant_memory_vectors = None
        self.xy_axis_linspace = []
        self.xy_axis_vectors = None
        self.xy_axis_matrix = None
        self.xy_axis_occupied_heatmap = None
        self.xy_axis_empty_heatmap = None
        self.xy_axis_class_matrix = None
        
        # Build the mapper components
        self._build_quadrant_hierarchy()
        self._build_quadrant_memory_hierarchy()
        self._build_quadrant_indices()
        self._build_xy_axis_linspace()
        self._build_xy_axis_vectors()
        self._build_xy_axis_matrix()
        self._build_xy_axis_heatmaps()
        self._build_xy_axis_class_matrices()
        
    def process_observation(
        self,
        points: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """
        Process a point cloud observation.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
        """
        # Convert to torch tensors if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).int()
        
        # Move to device if needed
        if points.device != self.device:
            points = points.to(self.device)
        
        if labels.device != self.device:
            labels = labels.to(self.device)
        
        # Normalize the point cloud to the world bounds
        normalized_points = points.clone()
        normalized_points[:, 0] -= self.world_bounds[0]
        normalized_points[:, 1] -= self.world_bounds[2]
        
        # Calculate quadrant memories for each new point
        # using a multipoint L2 distance calculation
        ups = normalized_points.unsqueeze(1)
        qcm = self.quadrant_centers[0]
        qcm = qcm.unsqueeze(0)
        qcm = qcm.repeat(ups.shape[0], 1, 1)
        dists = self.pdist(ups, qcm)
        closest_quads = torch.argmin(dists, dim=1)
        
        # Process occupied and empty points using batched operations
        occupied_points = normalized_points[labels == 1]
        occupied_points_closest_quads = closest_quads[labels == 1]
        
        if len(occupied_points) > 0:
            # Process occupied points in batches
            batch_size = 1000  # Adjust based on GPU memory
            num_batches = (len(occupied_points) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(occupied_points))
                
                batch_points = occupied_points[start_idx:end_idx]
                batch_quads = occupied_points_closest_quads[start_idx:end_idx]
                
                # Compute vectors for this batch
                x_vectors = power(self.xy_axis_vectors[0], batch_points[:, 0], self.length_scale)
                y_vectors = power(self.xy_axis_vectors[1], batch_points[:, 1], self.length_scale)
                
                # Bind vectors
                batch_result = bind_batch([x_vectors, y_vectors], self.device)
                
                # Update memory vectors
                for j, quad_idx in enumerate(batch_quads):
                    self.occupied_quadrant_memory_vectors[quad_idx] += batch_result[j]
        
        empty_points = normalized_points[labels == 0]
        empty_points_closest_quads = closest_quads[labels == 0]
        
        if len(empty_points) > 0:
            # Process empty points in batches
            batch_size = 1000  # Adjust based on GPU memory
            num_batches = (len(empty_points) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(empty_points))
                
                batch_points = empty_points[start_idx:end_idx]
                batch_quads = empty_points_closest_quads[start_idx:end_idx]
                
                # Compute vectors for this batch
                x_vectors = power(self.xy_axis_vectors[0], batch_points[:, 0], self.length_scale)
                y_vectors = power(self.xy_axis_vectors[1], batch_points[:, 1], self.length_scale)
                
                # Bind vectors
                batch_result = bind_batch([x_vectors, y_vectors], self.device)
                
                # Update memory vectors
                for j, quad_idx in enumerate(batch_quads):
                    self.empty_quadrant_memory_vectors[quad_idx] += batch_result[j]
        
        # Calculate the probabilities of occupancy and empty for each point in the xy axis matrix
        # Process quadrants in parallel where possible
        
        # Normalize memory vectors if needed
        if self.use_query_normalization:
            # Avoid division by zero
            occupied_norms = torch.linalg.norm(self.occupied_quadrant_memory_vectors, dim=1)
            occupied_norms = torch.clamp(occupied_norms, min=1e-10)
            normalized_occupied_vectors = self.occupied_quadrant_memory_vectors / occupied_norms.unsqueeze(1)
            
            empty_norms = torch.linalg.norm(self.empty_quadrant_memory_vectors, dim=1)
            empty_norms = torch.clamp(empty_norms, min=1e-10)
            normalized_empty_vectors = self.empty_quadrant_memory_vectors / empty_norms.unsqueeze(1)
        else:
            normalized_occupied_vectors = self.occupied_quadrant_memory_vectors
            normalized_empty_vectors = self.empty_quadrant_memory_vectors
        
        # Process quadrants
        counter = 0
        for j, y_lower in enumerate(self.quadrant_indices_y[:-1]):
            for i, x_lower in enumerate(self.quadrant_indices_x[:-1]):
                x_upper = self.quadrant_indices_x[i + 1]
                y_upper = self.quadrant_indices_y[j + 1]
                
                xy_axis_matrix_quad = self.xy_axis_matrix[x_lower:x_upper, y_lower:y_upper, :]
                
                # Calculate occupied heatmap
                qv_occupied = normalized_occupied_vectors[counter, :]
                occupied_heatmap = torch.tensordot(
                    qv_occupied,
                    xy_axis_matrix_quad,
                    dims=([0], [2])
                )
                
                # Calculate empty heatmap
                qv_empty = normalized_empty_vectors[counter, :]
                empty_heatmap = torch.tensordot(
                    qv_empty,
                    xy_axis_matrix_quad,
                    dims=([0], [2])
                )
                
                # Normalize heatmaps
                max_occupied = torch.max(torch.abs(occupied_heatmap))
                if max_occupied > 0:
                    occupied_heatmap /= max_occupied
                
                max_empty = torch.max(torch.abs(empty_heatmap))
                if max_empty > 0:
                    empty_heatmap /= max_empty
                
                # Update heatmaps
                self.xy_axis_occupied_heatmap[x_lower:x_upper, y_lower:y_upper] = occupied_heatmap
                self.xy_axis_empty_heatmap[x_lower:x_upper, y_lower:y_upper] = empty_heatmap
                
                counter += 1
        
        # Transpose and normalize final heatmaps
        self.xy_axis_occupied_heatmap = self.xy_axis_occupied_heatmap.T
        self.xy_axis_occupied_heatmap = torch.nan_to_num(self.xy_axis_occupied_heatmap)
        max_occupied = torch.max(torch.abs(self.xy_axis_occupied_heatmap))
        if max_occupied > 0:
            self.xy_axis_occupied_heatmap = self.xy_axis_occupied_heatmap / max_occupied
        
        self.xy_axis_empty_heatmap = self.xy_axis_empty_heatmap.T
        self.xy_axis_empty_heatmap = torch.nan_to_num(self.xy_axis_empty_heatmap)
        max_empty = torch.max(torch.abs(self.xy_axis_empty_heatmap))
        if max_empty > 0:
            self.xy_axis_empty_heatmap = self.xy_axis_empty_heatmap / max_empty
        
        # Update the class matrix based on the decision thresholds
        self._update_class_matrix()
        
    def get_occupancy_grid(self) -> torch.Tensor:
        """
        Get the current occupancy grid.
        
        Returns:
            Tensor of shape [H, W] containing occupancy probabilities
        """
        return self.xy_axis_occupied_heatmap
    
    def get_empty_grid(self) -> torch.Tensor:
        """
        Get the current empty grid.
        
        Returns:
            Tensor of shape [H, W] containing empty probabilities
        """
        return self.xy_axis_empty_heatmap
    
    def get_class_grid(self) -> torch.Tensor:
        """
        Get the current class grid.
        
        Returns:
            Tensor of shape [H, W] containing class labels (-1=empty, 0=unknown, 1=occupied)
        """
        return self.xy_axis_class_matrix
    
    def _update_class_matrix(self) -> None:
        """
        Update the class matrix based on the occupied and empty heatmaps.
        """
        # Initialize with unknown (0)
        self.xy_axis_class_matrix = torch.zeros(
            self.xy_axis_occupied_heatmap.shape,
            device=self.device
        )
        
        # Set occupied (1) where occupied heatmap > upper threshold
        self.xy_axis_class_matrix[self.xy_axis_occupied_heatmap > self.decision_thresholds[1]] = 1
        
        # Set empty (-1) where empty heatmap > upper threshold
        self.xy_axis_class_matrix[self.xy_axis_empty_heatmap > self.decision_thresholds[1]] = -1
    
    def _build_quadrant_hierarchy(self) -> None:
        """
        Build the quadrant hierarchy.
        """
        if self.verbose:
            print("Building quadrant hierarchy...")
        
        for level, size in enumerate(self.quadrant_hierarchy):
            self._build_quadrant_level(level, size)
    
    def _build_quadrant_level(self, level: int, size: int) -> None:
        """
        Build a specific level of the quadrant hierarchy.
        
        Args:
            level: The level in the hierarchy
            size: The size of the quadrant grid
        """
        # Calculate the quadrant size
        size_x_meters = self.world_bounds_norm[0] / size
        size_y_meters = self.world_bounds_norm[1] / size
        
        # Calculate the quadrant boundaries
        qb_x = torch.linspace(0, self.world_bounds_norm[0], size + 1, device=self.device)
        qb_y = torch.linspace(0, self.world_bounds_norm[1], size + 1, device=self.device)
        
        self.quadrant_axis_bounds.append((qb_x, qb_y))
        
        # Calculate the quadrant centers
        qcs_x = torch.linspace(0, self.world_bounds_norm[0], 2 * size + 1, device=self.device)[1::2]
        qcs_y = torch.linspace(0, self.world_bounds_norm[1], 2 * size + 1, device=self.device)[1::2]
        
        qcmg = torch.meshgrid(qcs_x, qcs_y, indexing="xy")
        qcs = torch.stack(qcmg, dim=2)
        qcs = qcs.reshape((size ** 2, 2))
        qcs = qcs.to(self.device)
        
        self.quadrant_centers.append(qcs)
    
    def _build_quadrant_memory_hierarchy(self) -> None:
        """
        Build the memory hierarchy for the quadrants.
        """
        if self.verbose:
            print("Building quadrant memory hierarchy...")
        
        # Initialize memory vectors for occupied and empty quadrants
        self.occupied_quadrant_memory_vectors = torch.zeros(
            size=(
                self.quadrant_hierarchy[0] ** self.environment_dimensionality,
                self.vsa_dimensions
            ),
            device=self.device
        )
        self.empty_quadrant_memory_vectors = torch.clone(self.occupied_quadrant_memory_vectors)
    
    def _build_quadrant_indices(self) -> None:
        """
        Build the quadrant indices based on the quadrant axis bounds.
        """
        if self.verbose:
            print("Building quadrant indices...")
        
        quadrant_indices_x = self.quadrant_axis_bounds[0][0] / self.axis_resolution
        quadrant_indices_y = self.quadrant_axis_bounds[0][1] / self.axis_resolution
        
        self.quadrant_indices_x = quadrant_indices_x.to(torch.int)
        self.quadrant_indices_y = quadrant_indices_y.to(torch.int)
    
    def _build_xy_axis_linspace(self) -> None:
        """
        Build the x and y axis linspace for the XY axis.
        """
        if self.verbose:
            print("Building XY axis linspace...")
        
        xal_steps = int(self.world_bounds_norm[0] / self.axis_resolution)
        yal_steps = int(self.world_bounds_norm[1] / self.axis_resolution)
        
        xa = torch.linspace(
            start=0,
            end=self.world_bounds_norm[0],
            steps=(2 * xal_steps + 1),
            device=self.device
        )
        ya = torch.linspace(
            start=0,
            end=self.world_bounds_norm[1],
            steps=(2 * yal_steps + 1),
            device=self.device
        )
        
        # Extract the centers from the axis linspace
        xac = xa[1::2]
        yac = ya[1::2]
        
        self.xy_axis_linspace = (xac, yac)
    
    def _build_xy_axis_vectors(self) -> None:
        """
        Build the XY axis vectors using the SSP generator.
        """
        if self.verbose:
            print("Building XY axis vectors...")
        
        self.xy_axis_vectors = self.ssp_generator.generate(
            self.environment_dimensionality
        )
    
    def _build_xy_axis_matrix(self) -> None:
        """
        Build the XY axis matrix using GPU acceleration.
        """
        if self.verbose:
            print("Building XY axis matrix...")

        x_shape = self.xy_axis_linspace[0].shape[0]
        y_shape = self.xy_axis_linspace[1].shape[0]

        # Create a meshgrid of all x,y coordinates
        x_coords = self.xy_axis_linspace[0].unsqueeze(1).repeat(1, y_shape)
        y_coords = self.xy_axis_linspace[1].unsqueeze(0).repeat(x_shape, 1)
        
        # Reshape to [num_points, 2]
        coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        
        # Process in batches to avoid GPU memory issues
        batch_size = 1000  # Reduced batch size to avoid OOM errors
        num_batches = (coords.shape[0] + batch_size - 1) // batch_size
        
        # Initialize the output matrix
        self.xy_axis_matrix = torch.zeros(
            (x_shape, y_shape, self.vsa_dimensions),
            device=self.device
        )
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, coords.shape[0])
            batch_coords = coords[start_idx:end_idx]
            
            # Compute vectors for this batch
            x_vectors = power(self.xy_axis_vectors[0], batch_coords[:, 0], self.length_scale)
            y_vectors = power(self.xy_axis_vectors[1], batch_coords[:, 1], self.length_scale)
            
            # Bind vectors
            batch_result = bind_batch([x_vectors, y_vectors], self.device)
            
            # Place results in the output matrix
            batch_indices = torch.arange(start_idx, end_idx, device=self.device)
            x_indices = batch_indices // y_shape
            y_indices = batch_indices % y_shape
            
            for j in range(len(batch_indices)):
                self.xy_axis_matrix[x_indices[j], y_indices[j], :] = batch_result[j]
        
        if self.verbose:
            print("Finished building XY axis matrix.")
    
    def _build_xy_axis_heatmaps(self) -> None:
        """
        Build the XY axis heatmaps.
        """
        if self.verbose:
            print("Building XY axis heatmaps...")
        
        self.xy_axis_occupied_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
        
        self.xy_axis_empty_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
    
    def _build_xy_axis_class_matrices(self) -> None:
        """
        Build the XY axis class matrices.
        """
        if self.verbose:
            print("Building XY axis class matrices...")
        
        self.xy_axis_class_matrix = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
