# Phase 1: Adaptive Spatial Indexing Implementation

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

Phase 1: Adaptive Spatial Indexing Implementation (Current Phase)
Phase 2: Simplified Vector Caching Implementation 
Phase 3: Enhanced VSA Mapper Implementation - Core Structure
Phase 4: Shannon Entropy Feature Extraction Implementation
Phase 5: Main Interface and CLI Updates
Phase 6: Comprehensive Testing and Documentation

## Phase 1 Focus: Adaptive Spatial Indexing

In this phase, we will focus on enhancing the existing `AdaptiveSpatialIndex` class in `src/spatial.py` to optimize spatial queries for point clouds and add new functionality for region safety checking. This component is critical for efficient incremental processing as it enables fast retrieval of points within a given radius of a sample position.

### Current Implementation Analysis

The current `AdaptiveSpatialIndex` class in `src/spatial.py` provides basic spatial indexing functionality but has several limitations:
- Fixed cell size calculation that doesn't fully adapt to point density
- Limited optimization for range queries
- No explicit handling of very large point clouds
- No support for region safety checking

### Implementation Plan

1. **Enhance Existing AdaptiveSpatialIndex Class**

We will directly modify the existing `AdaptiveSpatialIndex` class in `src/spatial.py` to add the necessary functionality. The existing methods will be optimized and a new method for region safety checking will be added.

2. **Add Region Safety Check Method**

```python
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
```

### Testing Plan

1. **Update Existing Tests**

We will update the existing tests in `tests/test_vsa.py` to test the enhanced functionality of the `AdaptiveSpatialIndex` class, including the new `is_region_free` method.

```python
def test_adaptive_spatial_index():
    """
    Test the AdaptiveSpatialIndex class.
    """
    print("Testing AdaptiveSpatialIndex...")
    
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
    
    # With small safety margin, should be free (center point is not occupied)
    assert spatial_index.is_region_free(bounds, 0.1)
    
    # With large safety margin, should not be free (occupied points nearby)
    assert not spatial_index.is_region_free(bounds, 0.5)
    
    print("AdaptiveSpatialIndex tests passed!")
    return True
```

2. **Add Performance Tests**

We will add performance tests to measure the efficiency of the enhanced `AdaptiveSpatialIndex` class, particularly for large point clouds.

```python
def test_spatial_index_performance():
    """
    Test the performance of the AdaptiveSpatialIndex class.
    """
    print("Testing spatial index performance...")
    
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
    
    # Test region safety check performance
    bounds = [45.0, 55.0, 45.0, 55.0]  # 10x10 region around center
    safety_margin = 5.0
    
    start_time = time.time()
    is_free = spatial_index.is_region_free(bounds, safety_margin)
    region_check_time = time.time() - start_time
    
    print(f"Region safety check time: {region_check_time:.4f} seconds")
    print(f"Region is {'free' if is_free else 'not free'}")
    
    print("Spatial index performance tests passed!")
    return True
```

### Expected Outcomes

1. **Improved Performance**: The optimized range queries will be significantly faster than brute force approaches, especially for large point clouds.
2. **Adaptive Cell Sizing**: The cell size will automatically adjust based on point density, providing a good balance between memory usage and query performance.
3. **Region Safety Checking**: The ability to check if a region is free of occupied points will be useful for sample position validation in incremental processing.

### Next Steps

After enhancing the spatial indexing functionality, we will proceed to Phase 2, which will focus on optimizing vector caching for efficient VSA operations.
