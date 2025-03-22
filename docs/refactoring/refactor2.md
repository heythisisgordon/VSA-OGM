# Phase 1: Adaptive Spatial Indexing Implementation

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

1. Efficient spatial indexing with adaptive cell sizing
2. Optimized vector computation with parallel processing and caching
3. Memory-aware processing with GPU memory monitoring
4. Shannon entropy feature extraction
5. Enhanced class grid generation based on entropy values 
6. Incremental processing with horizon-limited visibility
7. Enhanced VSA mapper with direct spatial processing

## Phase 1 Focus: Adaptive Spatial Indexing

In this phase, we will focus on implementing an enhanced version of the `AdaptiveSpatialIndex` class that optimizes spatial queries for point clouds. This component is critical for efficient incremental processing as it enables fast retrieval of points within a given radius of a sample position.

### Current Implementation Analysis

The current `AdaptiveSpatialIndex` class in `src/spatial.py` provides basic spatial indexing functionality but has several limitations:
- Fixed cell size calculation that doesn't fully adapt to point density
- Limited optimization for range queries
- No explicit handling of very large point clouds

### Implementation Plan

1. **Enhanced AdaptiveSpatialIndex Class**

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
```

2. **Optimized Cell Size Calculation**

```python
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
    point_density_tensor = torch.tensor(point_density, device=self.device)
    ideal_cell_size = torch.sqrt(target_points_per_cell / point_density_tensor)
    
    # Clamp to min/max resolution
    cell_size = torch.clamp(ideal_cell_size, min=min_resolution, max=max_resolution)
    
    return cell_size.item()
```

3. **Optimized Range Query with Squared Distances**

```python
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

4. **Region Safety Check for Sample Position Validation**

```python
def is_region_free(self, bounds, safety_margin):
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
```

### Testing Plan

1. **Unit Tests for AdaptiveSpatialIndex**

```python
def test_adaptive_spatial_index():
    """Test the enhanced AdaptiveSpatialIndex class."""
    # Create a test point cloud
    points = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ])
    
    labels = torch.tensor([0, 1, 0, 1, 0])
    
    # Create spatial index
    device = torch.device("cpu")
    min_resolution = 0.1
    max_resolution = 1.0
    
    spatial_index = AdaptiveSpatialIndex(
        points,
        labels,
        min_resolution,
        max_resolution,
        device
    )
    
    # Test cell size calculation
    assert spatial_index.cell_size >= min_resolution
    assert spatial_index.cell_size <= max_resolution
    
    # Test range query
    center = torch.tensor([0.5, 0.5])
    radius = 0.6
    
    query_points, query_labels = spatial_index.query_range(center, radius)
    
    # Should return all points (all within 0.6 of center)
    assert query_points.shape[0] == 5
    
    # Test with smaller radius
    radius = 0.3
    query_points, query_labels = spatial_index.query_range(center, radius)
    
    # Should only return the center point
    assert query_points.shape[0] == 1
    
    # Test region safety check
    bounds = [0.4, 0.6, 0.4, 0.6]  # Small region around center
    
    # With small safety margin, should be free
    assert spatial_index.is_region_free(bounds, 0.1)
    
    # With large safety margin, should not be free (occupied points nearby)
    assert not spatial_index.is_region_free(bounds, 0.5)
```

2. **Performance Tests**

```python
def test_spatial_index_performance():
    """Test the performance of the AdaptiveSpatialIndex class."""
    # Create a larger test point cloud
    n_points = 10000
    points = torch.rand((n_points, 2)) * 100  # Random points in 100x100 area
    labels = torch.randint(0, 2, (n_points,))  # Random labels
    
    # Create spatial index
    device = torch.device("cpu")
    min_resolution = 1.0
    max_resolution = 10.0
    
    start_time = time.time()
    spatial_index = AdaptiveSpatialIndex(
        points,
        labels,
        min_resolution,
        max_resolution,
        device
    )
    init_time = time.time() - start_time
    
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Cell size: {spatial_index.cell_size:.4f}")
    
    # Test range query performance
    center = torch.tensor([50.0, 50.0])
    radius = 10.0
    
    start_time = time.time()
    query_points, query_labels = spatial_index.query_range(center, radius)
    query_time = time.time() - start_time
    
    print(f"Range query time: {query_time:.4f} seconds")
    print(f"Found {query_points.shape[0]} points within radius {radius}")
    
    # Compare with brute force approach
    start_time = time.time()
    diffs = points - center
    distances = torch.sqrt(torch.sum(diffs * diffs, dim=1))
    mask = distances <= radius
    brute_force_points = points[mask]
    brute_force_time = time.time() - start_time
    
    print(f"Brute force time: {brute_force_time:.4f} seconds")
    print(f"Found {brute_force_points.shape[0]} points within radius {radius}")
    
    # Calculate speedup
    speedup = brute_force_time / query_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results match
    assert query_points.shape[0] == brute_force_points.shape[0]
```

### Integration with Existing Code

The enhanced `AdaptiveSpatialIndex` class will be a drop-in replacement for the current implementation in `src/spatial.py`. It maintains the same interface but provides improved performance and additional functionality.

### Expected Outcomes

1. **Improved Performance**: The optimized range queries will be significantly faster than brute force approaches, especially for large point clouds.
2. **Adaptive Cell Sizing**: The cell size will automatically adjust based on point density, providing a good balance between memory usage and query performance.
3. **Region Safety Checking**: The ability to check if a region is free of occupied points will be useful for sample position validation in incremental processing.

### Next Steps

After implementing the enhanced spatial indexing, we will proceed to Phase 2, which will focus on optimizing vector caching for efficient VSA operations.
