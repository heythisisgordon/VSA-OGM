# Enhanced VSA-OGM Implementation Plan: Optimized Spatial Indexing and Vector Processing

Building upon the streamlined approach in refactor19.md and incorporating the critiques and suggestions from refactor20.md, this enhanced plan further optimizes the VSA-OGM implementation while maintaining a balance between performance and maintainability.

## Core Approach

We'll maintain the focus on:

1. **Efficient Spatial Indexing**: Implement an adaptive grid-based spatial index with optimizations for varying point densities
2. **Optimized Vector Computation**: Process vectors in parallel with efficient caching and memory management
3. **Direct Spatial Processing**: Process points based on their spatial location with squared distance calculations
4. **Intelligent Memory Management**: Monitor and manage memory usage to prevent out-of-memory errors

The key improvements over the previous plan include:
- Adaptive cell sizing based on point density
- Fully parallelized vector computation
- Squared distance calculations to avoid unnecessary square root operations
- Explicit GPU memory monitoring and management
- Optimized range queries for better performance

## Key Components

### 1. Enhanced Adaptive Spatial Index

The spatial index will now adapt to point density and use squared distances for more efficient calculations:

```python
class AdaptiveSpatialIndex:
    def __init__(self, points, labels, min_resolution, max_resolution, device):
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
    
    def _optimize_cell_size(self, points, min_resolution, max_resolution):
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
        ideal_cell_size = torch.sqrt(target_points_per_cell / point_density)
        
        # Clamp to min/max resolution
        cell_size = torch.clamp(ideal_cell_size, min=min_resolution, max=max_resolution)
        
        return cell_size.item()
    
    def query_range(self, center, radius):
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
```

### 2. Optimized Vector Cache with Parallel Processing

The vector cache now processes all missing vectors in a single parallelized operation:

```python
class VectorCache:
    def __init__(self, xy_axis_vectors, length_scale, device, grid_resolution=0.1, max_size=10000):
        """
        Cache for VSA vectors to avoid redundant computation.
        
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
    
    def get_batch_vectors(self, points):
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
    
    def get_cache_stats(self):
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
    
    def clear(self):
        """Clear the cache to free memory"""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
```

### 3. Enhanced VSAMapper with Memory Monitoring

The VSAMapper now includes GPU memory monitoring and optimized processing:

```python
class EnhancedVSAMapper:
    """
    Enhanced VSA Mapper with adaptive spatial indexing, vector caching, and memory monitoring.
    """
    
    def __init__(self, config, device=None):
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
        
        if self.verbose:
            print(f"Initialized EnhancedVSAMapper with grid size: {self.grid_width}x{self.grid_height}")
            print(f"Using device: {self.device}")
    
    def check_memory_usage(self):
        """
        Monitor GPU memory usage and clear cache if needed.
        
        Returns:
            True if cache was cleared, False otherwise
        """
        if self.device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            max_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            
            if current_memory > self.memory_threshold * max_memory:
                if self.verbose:
                    print(f"Memory usage high ({current_memory:.2f}/{max_memory:.2f} GB), clearing cache")
                
                self.vector_cache.clear()
                torch.cuda.empty_cache()
                return True
        
        return False
    
    def process_observation(self, points, labels):
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
        
        if self.verbose:
            cache_stats = self.vector_cache.get_cache_stats()
            print(f"Vector cache stats: {cache_stats['hit_rate']*100:.1f}% hit rate, "
                  f"{cache_stats['cache_size']}/{cache_stats['max_size']} entries")
    
    def _process_points_spatially(self, points, labels):
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
    
    def _update_class_grid(self):
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
    
    def get_occupancy_grid(self):
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
    
    def get_empty_grid(self):
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
    
    def get_class_grid(self):
        """
        Get the current class grid.
        
        Returns:
            Tensor of shape [H, W] containing class labels (-1=empty, 0=unknown, 1=occupied)
        """
        return self.class_grid
    
    def process_incrementally(self, horizon_distance=10.0, sample_resolution=None, max_samples=None):
        """
        Process the point cloud incrementally from sample positions with optimized memory management.
        
        Args:
            horizon_distance: Maximum distance from sample point to consider points
            sample_resolution: Resolution for sampling grid (default: 10x resolution)
            max_samples: Maximum number of sample positions to process
        """
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
        
        if self.verbose:
            print(f"Incremental processing complete. Processed {total_points_processed} points "
                  f"from {positions.shape[0]} sample positions")
```

### 4. Updated Main Processing Function

```python
def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    vsa_dimensions: int = 16000,
    use_cuda: bool = True,
    verbose: bool = False,
    incremental: bool = False,
    horizon_distance: float = 10.0,
    sample_resolution: Optional[float] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    length_scale: float = 2.0,
    min_cell_resolution: Optional[float] = None,
    max_cell_resolution: Optional[float] = None,
    cache_size: int = 10000,
    memory_threshold: float = 0.8
) -> None:
    """
    Convert a point cloud to an occupancy grid map using the enhanced VSA mapper.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npy file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
        use_cuda: Whether to use CUDA if available
        verbose: Whether to print verbose output
        incremental: Whether to use incremental processing
        horizon_distance: Maximum distance from sample point to consider points
        sample_resolution: Resolution for sampling grid (default: 10x resolution)
        max_samples: Maximum number of sample positions to process
        batch_size: Batch size for processing points
        length_scale: Length scale for power operation
        min_cell_resolution: Minimum resolution for spatial indexing (default: 5x resolution)
        max_cell_resolution: Maximum resolution for spatial indexing (default: 20x resolution)
        cache_size: Maximum size of vector cache
        memory_threshold: Threshold for GPU memory usage (0.0-1.0)
    """
    # Set device (default to CUDA when available)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading point cloud from {input_file}")
    
    # Load point cloud
    points, labels = io.load_pointcloud(input_file, device)
    
    if verbose:
        print(f"Loaded point cloud with {points.shape[0]} points")
    
    # Set default world bounds if not provided
    if world_bounds is None:
        # Calculate from point cloud with padding
        x_min, y_min = points.min(dim=0).values.cpu().numpy() - 5.0
        x_max, y_max = points.max(dim=0).values.cpu().numpy() + 5.0
        world_bounds = [float(x_min), float(x_max), float(y_min), float(y_max)]
        
        if verbose:
            print(f"Automatically determined world bounds: {world_bounds}")
    
    # Set default cell resolutions if not provided
    if min_cell_resolution is None:
        min_cell_resolution = resolution * 5
    
    if max_cell_resolution is None:
        max_cell_resolution = resolution * 20
    
    # Create mapper configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "min_cell_resolution": min_cell_resolution,
        "max_cell_resolution": max_cell_resolution,
        "vsa_dimensions": vsa_dimensions,
        "length_scale": length_scale,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose,
        "batch_size": batch_size,
        "cache_size": cache_size,
        "memory_threshold": memory_threshold
    }
    
    if verbose:
        print("Initializing enhanced VSA mapper...")
    
    # Create enhanced mapper
    mapper = EnhancedVSAMapper(config, device=device)
    
    # Process point cloud
    if verbose:
        print("Processing point cloud...")
    
    # Record processing time
    start_time = time.time()
    
    mapper.process_observation(points, labels)
    
    # Process incrementally if requested
    if incremental:
        if verbose:
            print("Processing incrementally...")
        
        mapper.process_incrementally(
            horizon_distance=horizon_distance,
            sample_resolution=sample_resolution,
            max_samples=max_samples
        )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get and save occupancy grid
    if verbose:
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    io.save_occupancy_grid(grid, output_file, metadata={
        "world_bounds": world_bounds, 
        "resolution": resolution,
        "incremental": incremental,
        "horizon_distance": horizon_distance if incremental else None,
        "sample_resolution": sample_resolution if incremental else None,
        "processing_time": processing_time
    })
    
    if verbose:
        print(f"Occupancy grid saved to {output_file}")
```

## Implementation Strategy

### Phase 1: Adaptive Spatial Indexing

1. Create a new file `src/spatial.py` to implement the `AdaptiveSpatialIndex` class
2. Implement the adaptive cell sizing algorithm based on point density
3. Optimize range queries using squared distances for efficiency
4. Write unit tests to verify spatial indexing functionality
5. Benchmark performance with different point distributions

### Phase 2: Optimized Vector Cache

1. Create a new file `src/cache.py` to implement the enhanced `VectorCache` class
2. Implement fully parallelized vector computation for missing vectors
3. Add cache statistics tracking for performance monitoring
4. Write unit tests to verify caching functionality
5. Benchmark cache hit rates with different resolution settings

### Phase 3: Memory-Aware VSA Mapper

1. Create the `EnhancedVSAMapper` class in `src/enhanced_mapper.py`
2. Implement GPU memory monitoring and management
3. Integrate adaptive spatial indexing and vector caching
4. Optimize batch processing with periodic memory checks
5. Write unit tests to verify mapping functionality

### Phase 4: Incremental Processing

1. Enhance the incremental processing method with memory monitoring
2. Implement optimized sample position generation
3. Add progress tracking and statistics reporting
4. Write unit tests to verify incremental mapping functionality
5. Benchmark with different horizon distances and sample resolutions

### Phase 5: Integration and CLI Updates

1. Update the main processing function in `src/main.py` to use the enhanced mapper
2. Add command-line options for new parameters
3. Update documentation and examples
4. Perform integration testing with different point clouds
5. Create visualization tools for performance analysis

## Performance Considerations

1. **Memory Efficiency**:
   - Adaptive spatial indexing optimizes memory usage based on point distribution
   - Vector caching with statistics tracking balances memory and performance
   - Explicit GPU memory monitoring prevents out-of-memory errors
   - Periodic cache clearing during processing maintains stable memory usage

2. **Computational Efficiency**:
   - Fully parallelized vector computation leverages GPU capabilities
   - Squared distance calculations avoid unnecessary square root operations
   - Batched processing with vectorized operations maximizes throughput
   - Adaptive cell sizing ensures optimal spatial query performance

3. **Scalability**:
   - Memory monitoring ensures stability with very large point clouds
   - Adaptive parameters automatically adjust to different point distributions
   - Configurable batch sizes allow tuning for different hardware capabilities
   - Incremental processing with horizon limits handles arbitrarily large environments

## Expected Outcomes

1. **Improved Memory Usage**: More efficient memory utilization through adaptive spatial indexing and explicit memory management
2. **Faster Processing**: Reduced computation through optimized vector caching and parallel processing
3. **Better Scalability**: Ability to handle larger point clouds through adaptive parameters and memory monitoring
4. **Flexible Configuration**: Adjustable parameters for different environments and hardware capabilities
5. **Maintainable Code**: Modular implementation with clear separation of concerns

## Validation Approach

1. **Unit Testing**: Test each component individually with different input scenarios
2. **Integration Testing**: Test the complete pipeline with different point clouds
3. **Performance Benchmarking**: Compare memory usage and processing time with the original implementation
4. **Visualization**: Generate visualizations of the mapping process and performance metrics
5. **Stress Testing**: Evaluate behavior with extremely large point clouds and limited memory

## Conclusion

This enhanced implementation plan builds upon the streamlined approach from refactor19.md while addressing the critiques and suggestions from refactor20.md. By focusing on adaptive spatial indexing, parallel vector computation, and intelligent memory management, we can achieve significant performance improvements while maintaining a clean, modular implementation.

The key innovations in this plan include:
- Adaptive cell sizing based on point density
- Fully parallelized vector computation
- Explicit GPU memory monitoring and management
- Optimized distance calculations using squared distances
- Enhanced incremental processing with memory awareness

These improvements will make the VSA-OGM implementation more efficient, scalable, and robust while maintaining compatibility with the existing framework. The modular design ensures that each component can be tested and optimized independently, leading to a more maintainable codebase.
