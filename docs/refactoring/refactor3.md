# Phase 2: Simplified Vector Caching Implementation

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

Phase 1: Adaptive Spatial Indexing Implementation
Phase 2: Simplified Vector Caching Implementation (Current Phase)
Phase 3: Enhanced VSA Mapper Implementation - Core Structure
Phase 4: Shannon Entropy Feature Extraction Implementation
Phase 5: Main Interface and CLI Updates
Phase 6: Comprehensive Testing and Documentation

## Phase 2 Focus: Simplified Vector Caching

In this phase, we will focus on implementing an enhanced version of the `VectorCache` class that optimizes vector computation and caching for VSA operations. This component avoids redundant computation of VSA vectors for the same spatial locations.

### Current Implementation Analysis

The current `VectorCache` class in `src/cache.py` provides basic vector caching functionality but lacks optimization for batch vector computation and performance statistics tracking.

### Implementation Plan

1. **Simplified VectorCache Class**

```python
class VectorCache:
    """
    Cache for VSA vectors to avoid redundant computation.
    
    This class implements a cache for VSA vectors to avoid redundant computation
    of the same vectors, improving performance for repeated operations.
    """
    
    def __init__(
        self, 
        xy_axis_vectors: torch.Tensor, 
        length_scale: float, 
        device: torch.device, 
        grid_resolution: float = 0.1, 
        max_size: int = 10000
    ):
        """
        Initialize the vector cache.
        
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
```

2. **Optimized Batch Vector Retrieval**

```python
def get_batch_vectors(self, points: torch.Tensor) -> torch.Tensor:
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
            
            # Simple LRU-like cache management: if cache is full, remove oldest items
            if len(self.cache) >= self.max_size:
                # Remove first item (approximate LRU)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            # Add new item to cache
            self.cache[key_tuple] = missing_vectors[idx]
            
            # Update result
            result[i] = missing_vectors[idx]
    
    return result
```

3. **Simple Cache Statistics Tracking**

```python
def get_cache_stats(self) -> Dict[str, Any]:
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
```

4. **Clear Cache Method**

```python
def clear(self) -> None:
    """Clear the cache to free memory"""
    self.cache = {}
    # Keep statistics for monitoring
```

5. **Precompute Common Vectors**

```python
def precompute_grid_vectors(self, bounds: List[float], resolution: float) -> None:
    """
    Precompute vectors for a grid of points.
    
    Args:
        bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution
    """
    x_min, x_max, y_min, y_max = bounds
    
    # Create grid of points
    x_range = np.arange(x_min, x_max + resolution, resolution)
    y_range = np.arange(y_min, y_max + resolution, resolution)
    
    # Limit grid size to avoid memory issues
    max_points = 10000
    if len(x_range) * len(y_range) > max_points:
        # Sample points instead of full grid
        num_x = min(len(x_range), int(np.sqrt(max_points * len(x_range) / len(y_range))))
        num_y = min(len(y_range), int(np.sqrt(max_points * len(y_range) / len(x_range))))
        
        x_indices = np.linspace(0, len(x_range) - 1, num_x, dtype=int)
        y_indices = np.linspace(0, len(y_range) - 1, num_y, dtype=int)
        
        x_range = x_range[x_indices]
        y_range = y_range[y_indices]
    
    # Create meshgrid
    xx, yy = np.meshgrid(x_range, y_range)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Convert to tensor
    points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
    
    # Compute vectors
    self.get_batch_vectors(points_tensor)
    
    # Print statistics
    stats = self.get_cache_stats()
    print(f"Precomputed {stats['cache_size']} vectors")
```

### Testing Plan

1. **Unit Tests for VectorCache**

```python
def test_vector_cache():
    """Test the enhanced VectorCache class."""
    # Create random axis vectors for testing
    device = torch.device("cpu")
    vsa_dimensions = 1000
    length_scale = 2.0
    
    xy_axis_vectors = torch.randn((2, vsa_dimensions), device=device)
    
    # Create vector cache
    cache = VectorCache(
        xy_axis_vectors,
        length_scale,
        device,
        grid_resolution=0.1,
        max_size=1000
    )
    
    # Create test points
    points = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.1],  # Should be discretized to same as [0.0, 0.0]
        [1.0, 1.0],
        [1.1, 1.1],  # Should be discretized to same as [1.0, 1.0]
        [2.0, 2.0]
    ], device=device)
    
    # First batch - should all be cache misses
    batch1 = cache.get_batch_vectors(points)
    
    stats1 = cache.get_cache_stats()
    assert stats1["hits"] == 0
    assert stats1["misses"] == 5
    assert stats1["cache_size"] <= 5  # May be less due to discretization
    
    # Second batch - should have some cache hits
    batch2 = cache.get_batch_vectors(points)
    
    stats2 = cache.get_cache_stats()
    assert stats2["hits"] > 0
    assert stats2["cache_size"] <= 5
    
    # Test clear
    cache.clear()
    
    stats3 = cache.get_cache_stats()
    assert stats3["cache_size"] == 0
```

2. **Performance Tests**

```python
def test_vector_cache_performance():
    """Test the performance of the VectorCache class."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping performance test.")
        return
    
    # Create random axis vectors for testing
    device = torch.device("cuda")
    vsa_dimensions = 16000
    length_scale = 2.0
    
    xy_axis_vectors = torch.randn((2, vsa_dimensions), device=device)
    
    # Create vector cache
    cache = VectorCache(
        xy_axis_vectors,
        length_scale,
        device,
        grid_resolution=0.1,
        max_size=10000
    )
    
    # Create test points - large batch
    n_points = 10000
    points = torch.rand((n_points, 2), device=device) * 100  # Random points in 100x100 area
    
    # First batch - should all be cache misses
    start_time = time.time()
    batch1 = cache.get_batch_vectors(points)
    batch1_time = time.time() - start_time
    
    stats1 = cache.get_cache_stats()
    print(f"First batch time: {batch1_time:.4f} seconds")
    print(f"Cache stats after first batch: {stats1}")
    
    # Second batch - same points, should all be cache hits
    start_time = time.time()
    batch2 = cache.get_batch_vectors(points)
    batch2_time = time.time() - start_time
    
    stats2 = cache.get_cache_stats()
    print(f"Second batch time: {batch2_time:.4f} seconds")
    print(f"Cache stats after second batch: {stats2}")
    
    # Calculate speedup
    speedup = batch1_time / batch2_time
    print(f"Speedup from caching: {speedup:.2f}x")
```

### Integration with Existing Code

The simplified `VectorCache` class will be a drop-in replacement for the current implementation in `src/cache.py`. It maintains the same core interface but provides improved performance through batch processing and basic statistics tracking.

### Expected Outcomes

1. **Improved Performance**: The optimized batch vector computation will be significantly faster, especially for repeated operations.
2. **Simple Monitoring**: The basic statistics tracking will provide insights into cache performance.
3. **Fixed-Size Cache**: The simple LRU-like cache policy will prevent unbounded growth while maintaining good performance.

### Next Steps

After implementing the simplified vector caching, we will proceed to Phase 3, which will focus on implementing the enhanced VSA mapper with incremental processing capabilities.