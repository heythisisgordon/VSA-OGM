# Phase 3: Enhanced VSA Mapper Implementation - Core Structure

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

Phase 1: Adaptive Spatial Indexing Implementation
Phase 2: Optimized Vector Caching Implementation
Phase 3: Enhanced VSA Mapper Implementation - Core Structure (Current Phase)
Phase 4: Shannon Entropy Feature Extraction Implementation
Phase 5: Main Interface and CLI Updates
Phase 6: Comprehensive Testing and Documentation

## Phase 3 Focus: Enhanced VSA Mapper Core Structure

In this phase, we will implement the basic structure of an enhanced version of the `VSAMapper` class called `EnhancedVSAMapper` that incorporates the optimized spatial indexing and vector caching from the previous phases, along with memory-aware processing and improved incremental capabilities. This is the core component that will enable efficient processing of large point clouds with limited memory resources.

### Current Implementation Analysis

The current `VSAMapper` class in `src/mapper.py` provides basic functionality for processing point clouds, but has several limitations:
- Limited memory management for large point clouds
- Basic incremental processing without optimized sampling
- No explicit handling of very large environments
- Limited performance monitoring and statistics

### Implementation Plan

1. **Enhanced VSA Mapper Class**

```python
class EnhancedVSAMapper:
    """
    Enhanced VSA Mapper with adaptive spatial indexing, vector caching, and memory monitoring.
    
    This class implements the VSA-OGM algorithm with optimizations for memory usage,
    computational efficiency, and scalability.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the enhanced VSA mapper.
        
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
        
        # Initialize processing statistics
        self.stats = {
            "init_time": 0.0,
            "process_time": 0.0,
            "incremental_time": 0.0,
            "total_points_processed": 0,
            "total_samples_processed": 0,
            "memory_usage": []
        }
        
        # Record initialization time
        self.stats["init_time"] = time.time()
        
        if self.verbose:
            print(f"Initialized EnhancedVSAMapper with grid size: {self.grid_width}x{self.grid_height}")
            print(f"Using device: {self.device}")
```

2. **Memory Monitoring and Management**

```python
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
            
            # Manage cache size
            removed = self.vector_cache.manage_cache_size(current_memory, max_memory)
            
            if removed == 0:
                # If no items were removed from cache, clear it completely
                self.vector_cache.clear()
            
            torch.cuda.empty_cache()
            return True
    
    return False
```

3. **Basic Observation Processing**

```python
def process_observation(
    self, 
    points: torch.Tensor, 
    labels: torch.Tensor
) -> None:
    """
    Process a point cloud observation with memory monitoring.
    
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
    
    # Update class grid based on occupied and empty grids
    self._update_class_grid()
    
    # Check memory usage and clear cache if needed
    self.check_memory_usage()
    
    # Update statistics
    self.stats["process_time"] = time.time() - start_time
    self.stats["total_points_processed"] += points.shape[0]
    
    if self.verbose:
        cache_stats = self.vector_cache.get_cache_stats()
        print(f"Vector cache stats: {cache_stats['hit_rate']*100:.1f}% hit rate, "
              f"{cache_stats['cache_size']}/{cache_stats['max_size']} entries")
        print(f"Processing completed in {self.stats['process_time']:.2f} seconds")
```

4. **Optimized Points Processing**

```python
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
```

5. **Basic Grid Update**

```python
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
```

6. **Basic Incremental Processing**

```python
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
    
    # Update the class grid
    self._update_class_grid()
    
    # Clear vector cache to free memory
    self.vector_cache.clear()
    
    # Update statistics
    self.stats["incremental_time"] = time.time() - start_time
    self.stats["total_points_processed"] += total_points_processed
    self.stats["total_samples_processed"] += positions.shape[0]
    
    if self.verbose:
        print(f"Incremental processing complete. Processed {total_points_processed} points "
              f"from {positions.shape[0]} sample positions")
        print(f"Incremental processing completed in {self.stats['incremental_time']:.2f} seconds")
```

7. **Basic Grid Retrieval Methods**

```python
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
```

8. **Performance Statistics Method**

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get processing statistics.
    
    Returns:
        Dictionary with processing statistics
    """
    # Calculate total time
    total_time = self.stats["init_time"] + self.stats["process_time"] + self.stats["incremental_time"]
    
    # Get cache statistics
    cache_stats = self.vector_cache.get_cache_stats()
    
    # Combine statistics
    combined_stats = {
        "total_time": total_time,
        "init_time": self.stats["init_time"],
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
```

### Implementation Notes

This phase focuses on implementing the core structure of the enhanced VSA mapper with memory-aware processing and improved incremental capabilities. The key enhancements over the original implementation include:

1. **Memory Monitoring**: The enhanced mapper continuously monitors memory usage and takes appropriate actions to prevent out-of-memory errors.

2. **Optimized Incremental Processing**: The incremental processing approach is optimized with configurable parameters and safety margin checking.

3. **Performance Statistics**: Detailed statistics are collected for performance monitoring and analysis.

The base implementation in this phase will be extended in Phase 4 to add the Shannon entropy-based feature extraction capability described in the paper, which will significantly enhance the mapper's ability to extract features from noisy HDC representations.

### Next Steps

After implementing the core structure of the enhanced VSA mapper, we will proceed to Phase 4, which will focus on implementing the Shannon entropy-based feature extraction capability described in the paper.