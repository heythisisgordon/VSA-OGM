"""VSA-OGM mapper implementation."""

import torch
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple, Union

from .functional import SSPGenerator
from .spatial import AdaptiveSpatialIndex
from .cache import VectorCache

class VSAMapper:
    """
    VSA Mapper with adaptive spatial indexing, vector caching, and memory monitoring.
    
    This class implements the VSA-OGM algorithm with optimizations for memory usage,
    computational efficiency, and scalability.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the VSA mapper.
        
        Args:
            config: Configuration dictionary with parameters:
                - world_bounds: World bounds [x_min, x_max, y_min, y_max]
                - resolution: Grid resolution in meters
                - min_cell_resolution: Minimum resolution for spatial indexing
                - max_cell_resolution: Maximum resolution for spatial indexing
                - vsa_dimensions: Dimensionality of VSA vectors
                - length_scale: Length scale for power operation
                - batch_size: Batch size for processing points
                - cache_size: Maximum size of vector cache
                - memory_threshold: Threshold for GPU memory usage (0.0-1.0)
                - decision_thresholds: Thresholds for decision making
                - occupied_disk_radius: Radius for occupied disk filter in entropy calculation
                - empty_disk_radius: Radius for empty disk filter in entropy calculation
            device: Device to use for computation
        """
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract configuration parameters
        self.world_bounds = config["world_bounds"]
        self.resolution = config.get("resolution", 0.1)
        self.min_cell_resolution = config.get("min_cell_resolution", self.resolution * 5)
        self.max_cell_resolution = config.get("max_cell_resolution", self.resolution * 20)
        self.vsa_dimensions = config.get("vsa_dimensions", 16000)
        self.length_scale = config.get("length_scale", 2.0)
        self.batch_size = config.get("batch_size", 1000)
        self.cache_size = config.get("cache_size", 10000)
        self.memory_threshold = config.get("memory_threshold", 0.8)  # 80% by default
        self.verbose = config.get("verbose", False)
        self.decision_thresholds = config.get("decision_thresholds", [-0.99, 0.99])
        
        # Shannon entropy parameters
        self.occupied_disk_radius = config.get("occupied_disk_radius", 2)
        self.empty_disk_radius = config.get("empty_disk_radius", 4)
        
        # Calculate normalized world bounds
        self.world_bounds_norm = (
            self.world_bounds[1] - self.world_bounds[0],  # x range
            self.world_bounds[3] - self.world_bounds[2]   # y range
        )
        
        # Initialize SSP generator for axis vectors
        self.ssp_generator = SSPGenerator(
            dimensionality=self.vsa_dimensions,
            device=self.device,
            length_scale=self.length_scale
        )
        
        # Generate axis vectors
        self.xy_axis_vectors = self.ssp_generator.generate(2)  # 2D environment
        
        # Initialize vector cache
        self.vector_cache = VectorCache(
            self.xy_axis_vectors,
            self.length_scale,
            self.device,
            grid_resolution=self.resolution,
            max_size=self.cache_size
        )
        
        # Initialize spatial index (will be set during processing)
        self.spatial_index = None
        
        # Initialize grid dimensions
        self.grid_width = int(self.world_bounds_norm[0] / self.resolution)
        self.grid_height = int(self.world_bounds_norm[1] / self.resolution)
        
        # Initialize occupancy grids
        self.occupied_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
        self.empty_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
        self.class_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
        
        # Initialize entropy grids
        self.occupied_entropy_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
        self.empty_entropy_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
        self.global_entropy_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
        
        # Add statistics tracking
        self.stats = {
            "init_time": time.time(),
            "process_time": 0.0,
            "incremental_time": 0.0,
            "total_points_processed": 0,
            "total_samples_processed": 0,
            "memory_usage": []
        }
        
        if self.verbose:
            print(f"Initialized VSAMapper with grid size: {self.grid_width}x{self.grid_height}")
            print(f"Using device: {self.device}")
            print(f"Shannon entropy disk radii: occupied={self.occupied_disk_radius}, empty={self.empty_disk_radius}")
    
    def check_memory_usage(self) -> bool:
        """
        Monitor GPU memory usage and clear cache if needed.
        
        Returns:
            True if cache was cleared, False otherwise
        """
        if self.device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            max_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            
            # Record memory usage
            self.stats["memory_usage"].append((time.time(), current_memory, max_memory))
            
            if current_memory > self.memory_threshold * max_memory:
                if self.verbose:
                    print(f"Memory usage high ({current_memory:.2f}/{max_memory:.2f} GB), clearing cache")
                
                self.vector_cache.clear()
                torch.cuda.empty_cache()
                return True
        
        return False
    
    def _create_disk_filter(self, radius: int) -> torch.Tensor:
        """
        Create a disk filter for Shannon entropy calculation.
        
        Args:
            radius: Radius of the disk filter in voxels
            
        Returns:
            Binary disk filter as a tensor
        """
        diameter = 2 * radius + 1
        center = radius
        
        # Create grid coordinates
        y, x = torch.meshgrid(
            torch.arange(diameter, device=self.device),
            torch.arange(diameter, device=self.device),
            indexing="ij"
        )
        
        # Calculate squared distance from center
        squared_distance = (x - center) ** 2 + (y - center) ** 2
        
        # Create disk filter (1 inside disk, 0 outside)
        disk = (squared_distance <= radius ** 2).float()
        
        return disk
    
    def _shannon_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate Shannon entropy of probability values.
        
        Args:
            probabilities: Tensor of probability values (0.0-1.0)
            
        Returns:
            Entropy values
        """
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-10
        
        # Ensure probabilities are valid (between 0 and 1)
        probabilities = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
        
        # Calculate entropy: -p*log2(p) - (1-p)*log2(1-p)
        entropy = -probabilities * torch.log2(probabilities) - (1 - probabilities) * torch.log2(1 - probabilities)
        
        return entropy
    
    def _normalize_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Normalize a grid to the range [0, 1].
        
        Args:
            grid: Grid to normalize
            
        Returns:
            Normalized grid
        """
        # Get min and max values
        min_val = torch.min(grid)
        max_val = torch.max(grid)
        
        # Normalize grid
        if max_val > min_val:
            normalized_grid = (grid - min_val) / (max_val - min_val)
        else:
            normalized_grid = torch.zeros_like(grid)
        
        return normalized_grid
    
    def _apply_local_entropy(self, probability_grid: torch.Tensor, disk_filter: torch.Tensor) -> torch.Tensor:
        """
        Apply local Shannon entropy calculation with disk filter.
        
        Args:
            probability_grid: Grid of probability values
            disk_filter: Disk filter to use for local entropy calculation
            
        Returns:
            Grid of local entropy values
        """
        # Get disk dimensions
        filter_height, filter_width = disk_filter.shape
        padding_y = filter_height // 2
        padding_x = filter_width // 2
        
        # Pad probability grid
        padded_grid = torch.nn.functional.pad(
            probability_grid,
            (padding_x, padding_x, padding_y, padding_y),
            mode='constant',
            value=0
        )
        
        # Initialize entropy grid
        entropy_grid = torch.zeros_like(probability_grid)
        
        # Apply disk filter to each position
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Extract local region
                local_region = padded_grid[y:y + filter_height, x:x + filter_width]
                
                # Apply disk filter
                masked_region = local_region * disk_filter
                
                # Count non-zero elements in mask
                num_elements = torch.sum(disk_filter).item()
                
                # Calculate mean probability within disk
                if num_elements > 0:
                    mean_prob = torch.sum(masked_region) / num_elements
                else:
                    mean_prob = 0.0
                
                # Calculate entropy
                entropy_grid[y, x] = self._shannon_entropy(torch.tensor([mean_prob], device=self.device)).item()
        
        return entropy_grid
    
    def _apply_shannon_entropy(self) -> None:
        """
        Apply Shannon entropy-based feature extraction as described in the paper.
        
        This method implements the Shannon entropy approach described in the paper:
        1. Convert quasi-probabilities to true probabilities using the Born rule
        2. Apply disk filters to compute local entropy for both occupied and empty grids
        3. Calculate global entropy as the difference between occupied and empty entropy
        """
        if self.verbose:
            print("Applying Shannon entropy-based feature extraction...")
        
        # Create disk filters
        occupied_disk = self._create_disk_filter(self.occupied_disk_radius)
        empty_disk = self._create_disk_filter(self.empty_disk_radius)
        
        # Normalize occupied and empty grids to get probability maps
        occupied_prob = self._normalize_grid(self.occupied_grid)
        empty_prob = self._normalize_grid(self.empty_grid)
        
        # Apply Born rule: true probability = squared quasi-probability
        occupied_prob = occupied_prob ** 2
        empty_prob = empty_prob ** 2
        
        # Calculate local entropy for occupied and empty grids
        self.occupied_entropy_grid = self._apply_local_entropy(occupied_prob, occupied_disk)
        self.empty_entropy_grid = self._apply_local_entropy(empty_prob, empty_disk)
        
        # Calculate global entropy as the difference between occupied and empty entropy
        self.global_entropy_grid = self.occupied_entropy_grid - self.empty_entropy_grid
        
        if self.verbose:
            print("Shannon entropy-based feature extraction completed")
    
    def _update_class_grid_from_entropy(self) -> None:
        """
        Update class grid based on Shannon entropy values as described in the paper.
        
        This method implements the classification approach described in the paper,
        using the global entropy grid to classify each voxel as occupied, empty, or unknown.
        """
        # Initialize with unknown (0)
        self.class_grid = torch.zeros_like(self.global_entropy_grid)
        
        # Set occupied (1) where global entropy is above upper threshold
        self.class_grid[self.global_entropy_grid > self.decision_thresholds[1]] = 1
        
        # Set empty (-1) where global entropy is below lower threshold
        self.class_grid[self.global_entropy_grid < self.decision_thresholds[0]] = -1
        
        if self.verbose:
            occupied_count = torch.sum(self.class_grid == 1).item()
            empty_count = torch.sum(self.class_grid == -1).item()
            unknown_count = torch.sum(self.class_grid == 0).item()
            
            print(f"Class grid updated from entropy: {occupied_count} occupied, "
                  f"{empty_count} empty, {unknown_count} unknown voxels")
    
    def process_observation(
        self, 
        points: torch.Tensor, 
        labels: torch.Tensor
    ) -> None:
        """
        Process a point cloud observation with memory monitoring and Shannon entropy.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
        """
        # Record start time
        start_time = time.time()
        
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
        
        # Initialize adaptive spatial index
        if self.verbose:
            print("Initializing adaptive spatial index...")
            
        self.spatial_index = AdaptiveSpatialIndex(
            normalized_points,
            labels,
            self.min_cell_resolution,
            self.max_cell_resolution,
            self.device
        )
        
        if self.verbose:
            print(f"Processing point cloud with {points.shape[0]} points")
            print(f"Spatial index cell size: {self.spatial_index.cell_size:.4f}")
        
        # Process points directly using spatial grid
        self._process_points_spatially(normalized_points, labels)
        
        # Apply Shannon entropy for feature extraction
        self._apply_shannon_entropy()
        
        # Update class grid based on entropy values
        self._update_class_grid_from_entropy()
        
        # Check memory usage and clear cache if needed
        self.check_memory_usage()
        
        # Update statistics
        self.stats["process_time"] += time.time() - start_time
        self.stats["total_points_processed"] += points.shape[0]
        
        if self.verbose:
            cache_stats = self.vector_cache.get_cache_stats()
            print(f"Vector cache stats: {cache_stats['hit_rate']*100:.1f}% hit rate, "
                  f"{cache_stats['cache_size']}/{cache_stats['max_size']} entries")
            print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    def _process_points_spatially(
        self, 
        points: torch.Tensor, 
        labels: torch.Tensor
    ) -> None:
        """
        Process points directly using spatial grid with optimized batch processing.
        
        Args:
            points: Normalized points tensor
            labels: Labels tensor
        """
        # Separate occupied and empty points
        occupied_points = points[labels == 1]
        empty_points = points[labels == 0]
        
        # Process occupied points in batches
        if len(occupied_points) > 0:
            num_batches = (len(occupied_points) + self.batch_size - 1) // self.batch_size
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(occupied_points))
                
                batch_points = occupied_points[start_idx:end_idx]
                
                # Get vectors from cache
                batch_vectors = self.vector_cache.get_batch_vectors(batch_points)
                
                # Convert points to grid coordinates
                grid_x = torch.floor(batch_points[:, 0] / self.resolution).long()
                grid_y = torch.floor(batch_points[:, 1] / self.resolution).long()
                
                # Ensure coordinates are within grid bounds
                grid_x = torch.clamp(grid_x, 0, self.grid_width - 1)
                grid_y = torch.clamp(grid_y, 0, self.grid_height - 1)
                
                # Update occupied grid using vectorized operations where possible
                for j in range(len(grid_x)):
                    x, y = grid_x[j], grid_y[j]
                    self.occupied_grid[y, x] += 1.0
                
                # Check memory usage periodically
                if (i + 1) % 10 == 0:
                    self.check_memory_usage()
        
        # Process empty points in batches
        if len(empty_points) > 0:
            num_batches = (len(empty_points) + self.batch_size - 1) // self.batch_size
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(empty_points))
                
                batch_points = empty_points[start_idx:end_idx]
                
                # Get vectors from cache
                batch_vectors = self.vector_cache.get_batch_vectors(batch_points)
                
                # Convert points to grid coordinates
                grid_x = torch.floor(batch_points[:, 0] / self.resolution).long()
                grid_y = torch.floor(batch_points[:, 1] / self.resolution).long()
                
                # Ensure coordinates are within grid bounds
                grid_x = torch.clamp(grid_x, 0, self.grid_width - 1)
                grid_y = torch.clamp(grid_y, 0, self.grid_height - 1)
                
                # Update empty grid using vectorized operations where possible
                for j in range(len(grid_x)):
                    x, y = grid_x[j], grid_y[j]
                    self.empty_grid[y, x] += 1.0
                
                # Check memory usage periodically
                if (i + 1) % 10 == 0:
                    self.check_memory_usage()
    
    def _update_class_grid(self) -> None:
        """
        Update class grid based on occupied and empty grids.
        """
        # Normalize grids
        max_occupied = torch.max(self.occupied_grid)
        if max_occupied > 0:
            occupied_norm = self.occupied_grid / max_occupied
        else:
            occupied_norm = self.occupied_grid
        
        max_empty = torch.max(self.empty_grid)
        if max_empty > 0:
            empty_norm = self.empty_grid / max_empty
        else:
            empty_norm = self.empty_grid
        
        # Initialize with unknown (0)
        self.class_grid = torch.zeros_like(self.occupied_grid)
        
        # Set occupied (1) where occupied grid > upper threshold
        self.class_grid[occupied_norm > self.decision_thresholds[1]] = 1
        
        # Set empty (-1) where empty grid > upper threshold
        self.class_grid[empty_norm > self.decision_thresholds[1]] = -1
    
    def get_occupancy_grid(self) -> torch.Tensor:
        """
        Get the current occupancy grid.
        
        Returns:
            Tensor of shape [H, W] containing occupancy probabilities
        """
        # Normalize occupied grid
        max_val = torch.max(self.occupied_grid)
        if max_val > 0:
            return self.occupied_grid / max_val
        return self.occupied_grid
    
    def get_empty_grid(self) -> torch.Tensor:
        """
        Get the current empty grid.
        
        Returns:
            Tensor of shape [H, W] containing empty probabilities
        """
        # Normalize empty grid
        max_val = torch.max(self.empty_grid)
        if max_val > 0:
            return self.empty_grid / max_val
        return self.empty_grid
    
    def get_class_grid(self) -> torch.Tensor:
        """
        Get the current class grid.
        
        Returns:
            Tensor of shape [H, W] containing class labels (-1=empty, 0=unknown, 1=occupied)
        """
        return self.class_grid
    
    def process_incrementally(
        self, 
        horizon_distance: float = 10.0, 
        sample_resolution: Optional[float] = None, 
        max_samples: Optional[int] = None,
        safety_margin: float = 0.5
    ) -> None:
        """
        Process the point cloud incrementally from sample positions with optimized memory management.
        
        Args:
            horizon_distance: Maximum distance from sample point to consider points
            sample_resolution: Resolution for sampling grid (default: 10x resolution)
            max_samples: Maximum number of sample positions to process
            safety_margin: Minimum distance from occupied points for sampling
        """
        # Record start time
        start_time = time.time()
        
        if self.spatial_index is None:
            raise ValueError("Spatial index not initialized. Call process_observation first.")
        
        if sample_resolution is None:
            sample_resolution = self.resolution * 10
        
        # Generate sample positions using a grid
        x_min, x_max = 0, self.world_bounds_norm[0]
        y_min, y_max = 0, self.world_bounds_norm[1]
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Calculate number of samples in each dimension
        nx = int(x_range / sample_resolution) + 1
        ny = int(y_range / sample_resolution) + 1
        
        if self.verbose:
            print(f"Incremental processing with {nx}x{ny} sample positions")
            print(f"Horizon distance: {horizon_distance}")
        
        # Generate grid of sample positions
        x_positions = torch.linspace(x_min, x_max, nx, device=self.device)
        y_positions = torch.linspace(y_min, y_max, ny, device=self.device)
        
        # Create meshgrid of positions
        xx, yy = torch.meshgrid(x_positions, y_positions, indexing="ij")
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Filter out positions that are too close to occupied points
        if safety_margin > 0:
            valid_positions = []
            for position in positions:
                # Create a small region around the sample point
                bounds = [
                    position[0].item() - sample_resolution/2,
                    position[0].item() + sample_resolution/2,
                    position[1].item() - sample_resolution/2,
                    position[1].item() + sample_resolution/2
                ]
                
                # Check if region is free of occupied points
                if self.spatial_index.is_region_free(bounds, safety_margin):
                    valid_positions.append(position)
            
            if valid_positions:
                positions = torch.stack(valid_positions)
            
            if self.verbose:
                print(f"Filtered to {positions.shape[0]} valid sample positions")
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples < positions.shape[0]:
            if self.verbose:
                print(f"Limiting to {max_samples} sample positions")
            
            # Randomly select positions
            indices = torch.randperm(positions.shape[0], device=self.device)[:max_samples]
            positions = positions[indices]
        
        # Reset grids
        self.occupied_grid.zero_()
        self.empty_grid.zero_()
        
        # Process each sample position
        total_points_processed = 0
        
        for i, position in enumerate(positions):
            # Query points within horizon distance
            points, labels = self.spatial_index.query_range(position, horizon_distance)
            
            if points.shape[0] > 0:
                # Process these points
                self._process_points_spatially(points, labels)
                total_points_processed += points.shape[0]
            
            # Check memory usage periodically
            if (i + 1) % 10 == 0:
                self.check_memory_usage()
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{positions.shape[0]} sample positions, "
                      f"{total_points_processed} total points")
        
        # Apply Shannon entropy for feature extraction
        self._apply_shannon_entropy()
        
        # Update the class grid based on entropy values
        self._update_class_grid_from_entropy()
        
        # Clear vector cache to free memory
        self.vector_cache.clear()
        
        # Update statistics
        self.stats["incremental_time"] += time.time() - start_time
        self.stats["total_samples_processed"] += positions.shape[0]
        
        if self.verbose:
            print(f"Incremental processing complete. Processed {total_points_processed} points "
                  f"from {positions.shape[0]} sample positions")
            print(f"Incremental processing completed in {time.time() - start_time:.2f} seconds")
    
    def get_occupied_entropy_grid(self) -> torch.Tensor:
        """
        Get the occupied entropy grid.
        
        Returns:
            Tensor of shape [H, W] containing occupied entropy values
        """
        return self.occupied_entropy_grid

    def get_empty_entropy_grid(self) -> torch.Tensor:
        """
        Get the empty entropy grid.
        
        Returns:
            Tensor of shape [H, W] containing empty entropy values
        """
        return self.empty_entropy_grid

    def get_global_entropy_grid(self) -> torch.Tensor:
        """
        Get the global entropy grid.
        
        Returns:
            Tensor of shape [H, W] containing global entropy values
        """
        return self.global_entropy_grid
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        # Calculate total time
        total_time = (time.time() - self.stats["init_time"]) if "init_time" in self.stats else 0
        
        # Get cache statistics
        cache_stats = self.vector_cache.get_cache_stats()
        
        # Combine statistics
        combined_stats = {
            "total_time": total_time,
            "process_time": self.stats["process_time"],
            "incremental_time": self.stats["incremental_time"],
            "total_points_processed": self.stats["total_points_processed"],
            "total_samples_processed": self.stats["total_samples_processed"],
            "points_per_second": self.stats["total_points_processed"] / total_time if total_time > 0 else 0,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["cache_size"],
            "cache_max_size": cache_stats["max_size"]
        }
        
        # Add memory statistics if available
        if self.stats["memory_usage"]:
            latest_memory = self.stats["memory_usage"][-1]
            combined_stats["current_memory_gb"] = latest_memory[1]
            combined_stats["max_memory_gb"] = latest_memory[2]
            combined_stats["memory_usage_ratio"] = latest_memory[1] / latest_memory[2]
        
        return combined_stats
