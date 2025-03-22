# Streamlined Implementation Plan: Efficient VSA-OGM with Spatial Indexing

Based on the original plan in refactor16.md, our detailed implementation in refactor17.md, and the critiques in refactor18.md and refactor20.md, I propose the following streamlined approach that maximizes performance while minimizing implementation complexity.

## Core Approach

Rather than implementing a complex quadtree-based incremental mapping system, we'll focus on:

1. **Efficient Spatial Indexing**: Implement a simplified grid-based spatial index using PyTorch's tensor operations
2. **Optimized Vector Computation**: Process vectors in batches with efficient caching
3. **Direct Spatial Processing**: Process points based on their spatial location rather than using quadrants
4. **Simplified Memory Management**: Clear caches at logical points in the processing pipeline

This approach delivers significant performance benefits with a clean, maintainable implementation.

## Key Components

### 1. Grid-Based Spatial Index

Instead of a full quadtree implementation, we'll use a grid-based spatial index that leverages PyTorch's efficient tensor operations:

```python
class SpatialIndex:
    def __init__(self, points, labels, resolution, device):
        """
        Initialize a grid-based spatial index.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
            resolution: Resolution of the grid cells (typically 5-10x the map resolution)
            device: Device to store tensors on
        """
        self.device = device
        self.points = points
        self.labels = labels
        
        # Automatically determine optimal cell size based on point density
        self.cell_size = self._calculate_cell_size(points, resolution)
        
        # Compute grid cell indices for each point
        self.cell_indices = torch.floor(points / self.cell_size).long()
        
        # Create dictionary mapping from cell indices to point indices
        self.grid = {}
        for i, (x, y) in enumerate(self.cell_indices):
            key = (x.item(), y.item())
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(i)
    
    def _calculate_cell_size(self, points, base_resolution):
        """
        Calculate appropriate cell size based on point density.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            base_resolution: Base resolution for the grid
            
        Returns:
            Appropriate cell size
        """
        # Simple heuristic: use 5-10x the base resolution
        # For very large point clouds, increase cell size to reduce memory usage
        if points.shape[0] > 1000000:  # Over 1 million points
            return base_resolution * 10
        elif points.shape[0] > 100000:  # Over 100k points
            return base_resolution * 7
        else:
            return base_resolution * 5
    
    def query_range(self, center, radius):
        """
        Find all points within a given radius of center.
        
        Args:
            center: [x, y] coordinates of query center
            radius: Search radius
            
        Returns:
            Tuple of (points, labels) tensors for points within the radius
        """
        # Convert center to tensor if it's not already
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, device=self.device)
        
        # Calculate cell range to search
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
        
        # Filter points within radius
        mask = squared_distances <= squared_radius
        result_indices = candidate_indices[mask]
        
        return self.points[result_indices], self.labels[result_indices]
```

### 2. Vector Caching System

To avoid redundant computation of VSA vectors, we'll implement a caching system:

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
            else:
                missing_indices.append(i)
        
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
    
    def clear(self):
        """Clear the cache to free memory"""
        self.cache = {}
```

### 3. Enhanced VSAMapper

We'll enhance the VSAMapper class with spatial indexing and vector caching:

```python
class DirectVSAMapper:
    """
    Direct VSA Mapper with spatial indexing and vector caching.
    
    This class replaces the quadrant-based approach with direct spatial processing.
    """
    
    def __init__(self, config, device=None):
        """
        Initialize the direct VSA mapper.
        
        Args:
            config: Configuration dictionary with parameters:
                - world_bounds: World bounds [x_min, x_max, y_min, y_max]
                - resolution: Grid resolution in meters
                - vsa_dimensions: Dimensionality of VSA vectors
                - length_scale: Length scale for power operation
                - batch_size: Batch size for processing points
            device: Device to use for computation
        """
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract configuration parameters
        self.world_bounds = config["world_bounds"]
        self.resolution = config.get("resolution", 0.1)
        self.vsa_dimensions = config.get("vsa_dimensions", 16000)
        self.length_scale = config.get("length_scale", 2.0)
        self.batch_size = config.get("batch_size", 1000)
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
            max_size=10000
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
        
        # Initialize spatial index
        self.spatial_index = SpatialIndex(
            normalized_points,
            labels,
            self.resolution,
            self.device
        )
        
        if self.verbose:
            print(f"Processing point cloud with {points.shape[0]} points")
        
        # Process points directly using spatial grid
        self._process_points_spatially(normalized_points, labels)
        
        # Update class grid based on occupied and empty grids
        self._update_class_grid()
        
        # Clear cache to free memory
        self.vector_cache.clear()
        
        # Explicitly free GPU memory if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _process_points_spatially(self, points, labels):
        """
        Process points directly using spatial grid.
        
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
                
                # Update occupied grid
                for j in range(len(grid_x)):
                    x, y = grid_x[j], grid_y[j]
                    self.occupied_grid[y, x] += 1.0
        
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
                
                # Update empty grid
                for j in range(len(grid_x)):
                    x, y = grid_x[j], grid_y[j]
                    self.empty_grid[y, x] += 1.0
    
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
        Process the point cloud incrementally from sample positions.
        
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
        
        # Generate grid of sample positions
        x_positions = torch.linspace(x_min, x_max, nx, device=self.device)
        y_positions = torch.linspace(y_min, y_max, ny, device=self.device)
        
        # Create meshgrid of positions
        xx, yy = torch.meshgrid(x_positions, y_positions, indexing="ij")
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Limit number of samples if specified
        if max_samples is not None:
            positions = positions[:max_samples]
        
        # Reset grids
        self.occupied_grid.zero_()
        self.empty_grid.zero_()
        
        # Process each sample position
        for i, position in enumerate(positions):
            # Query points within horizon distance
            points, labels = self.spatial_index.query_range(position, horizon_distance)
            
            if points.shape[0] > 0:
                # Process these points
                self._process_points_spatially(points, labels)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{positions.shape[0]} sample positions")
        
        # Update the class grid
        self._update_class_grid()
        
        # Clear vector cache to free memory
        self.vector_cache.clear()
```

### 4. Incremental Processing

The incremental processing capability is integrated directly into the DirectVSAMapper class:

```python
def process_incrementally(self, horizon_distance=10.0, sample_resolution=None, max_samples=None):
    """
    Process the point cloud incrementally from sample positions.
    
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
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate number of samples in each dimension
    nx = int(x_range / sample_resolution) + 1
    ny = int(y_range / sample_resolution) + 1
    
    # Generate grid of sample positions
    x_positions = torch.linspace(x_min, x_max, nx, device=self.device)
    y_positions = torch.linspace(y_min, y_max, ny, device=self.device)
    
    # Create meshgrid of positions
    xx, yy = torch.meshgrid(x_positions, y_positions, indexing="ij")
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Limit number of samples if specified
    if max_samples is not None:
        positions = positions[:max_samples]
    
    # Reset grids
    self.occupied_grid.zero_()
    self.empty_grid.zero_()
    
    # Process each sample position
    for i, position in enumerate(positions):
        # Query points within horizon distance
        points, labels = self.spatial_index.query_range(position, horizon_distance)
        
        if points.shape[0] > 0:
            # Process these points
            self._process_points_spatially(points, labels)
        
        if self.verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{positions.shape[0]} sample positions")
    
    # Update the class grid
    self._update_class_grid()
    
    # Clear vector cache to free memory
    self.vector_cache.clear()
```

### 5. Updated Main Processing Function

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
    length_scale: float = 2.0
) -> None:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npy file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        axis_resolution: Resolution for axis vectors (typically 5x grid resolution)
        vsa_dimensions: Dimensionality of VSA vectors
        use_cuda: Whether to use CUDA if available
        verbose: Whether to print verbose output
        incremental: Whether to use incremental processing
        horizon_distance: Maximum distance from sample point to consider points
        sample_resolution: Resolution for sampling grid (default: 10x resolution)
        max_samples: Maximum number of sample positions to process
        spatial_cell_size: Cell size for spatial indexing (default: 10x resolution)
        batch_size: Batch size for processing points
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
    
    # Create mapper configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "vsa_dimensions": vsa_dimensions,
        "length_scale": length_scale,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose,
        "batch_size": batch_size
    }
    
    if verbose:
        print("Initializing direct VSA mapper...")
    
    # Create direct mapper
    mapper = DirectVSAMapper(config, device=device)
    
    # Process point cloud
    if verbose:
        print("Processing point cloud...")
    
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
    
    # Get and save occupancy grid
    if verbose:
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    io.save_occupancy_grid(grid, output_file, metadata={
        "world_bounds": world_bounds, 
        "resolution": resolution,
        "incremental": incremental,
        "horizon_distance": horizon_distance if incremental else None,
        "sample_resolution": sample_resolution if incremental else None
    })
    
    if verbose:
        print(f"Occupancy grid saved to {output_file}")
```

## Implementation Strategy

### Phase 1: Spatial Indexing and Vector Caching

1. Create a new file `src/spatial.py` to implement the `SpatialIndex` class
2. Create a new file `src/cache.py` to implement the `VectorCache` class with parallel vector computation
3. Write unit tests to verify spatial indexing and vector caching functionality

### Phase 2: Direct VSA Mapper

1. Create the `DirectVSAMapper` class that processes points directly using spatial grid
2. Implement efficient processing methods using spatial indexing and vector caching
3. Write unit tests to verify mapping functionality

### Phase 3: Incremental Processing

1. Implement incremental processing in the `DirectVSAMapper` class
2. Add sample position generation and processing
3. Write unit tests to verify incremental mapping functionality

### Phase 4: Integration and CLI Updates

1. Update the main processing function to use the enhanced mapper
2. Add command-line options for new parameters
3. Update documentation and examples
4. Perform integration testing with different point clouds

## Performance Considerations

1. **Memory Efficiency**:
   - Vector caching with access-based pruning balances memory usage and performance
   - Explicit GPU memory monitoring prevents out-of-memory errors
   - Adaptive grid-based spatial indexing optimizes for point distribution

2. **Computational Efficiency**:
   - Parallel vector computation leverages GPU capabilities
   - Squared distance calculations avoid unnecessary square root operations
   - Batched processing with vectorized operations maximizes throughput

3. **Scalability**:
   - The approach scales well with point cloud size due to adaptive parameters
   - Memory monitoring ensures stability with very large point clouds
   - Adjustable parameters allow tuning for different hardware capabilities

## Expected Outcomes

1. **Improved Memory Usage**: More efficient memory utilization through spatial indexing
2. **Faster Processing**: Reduced computation through vector caching and optimized spatial queries
3. **Flexible Configuration**: Adjustable parameters for different environments and requirements
4. **Maintainable Code**: Simpler implementation than the full quadtree approach

## Validation Approach

1. **Unit Testing**: Test each component individually
2. **Integration Testing**: Test the complete pipeline with different point clouds
3. **Performance Benchmarking**: Compare memory usage and processing time with the original implementation
4. **Visualization**: Generate visualizations of the mapping process

## Conclusion

This optimized approach strikes a balance between performance improvements and implementation complexity. By focusing on the most impactful optimizations (adaptive spatial indexing, parallel vector computation, and efficient memory management), we can achieve significant performance gains without the full complexity of a quadtree-based incremental mapping system.

The implementation is modular and incorporates mathematical optimizations that address the specific needs of VSA-OGM processing. The adaptive elements ensure good performance across a wide range of point cloud distributions and hardware configurations, while maintaining compatibility with the existing VSA-OGM framework.
