# Phase 2: Optimized Vector Caching Implementation

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

Phase 1: Adaptive Spatial Indexing Implementation
Phase 2: Optimized Vector Caching Implementation (Current Phase)
Phase 3: Enhanced VSA Mapper Implementation - Core Structure
Phase 4: Shannon Entropy Feature Extraction Implementation
Phase 5: Main Interface and CLI Updates
Phase 6: Comprehensive Testing and Documentation

## Phase 2 Focus: Optimized Vector Caching

In this phase, we will focus on implementing an enhanced version of the `VectorCache` class that optimizes vector computation and caching for VSA operations. This component is critical for efficient processing as it avoids redundant computation of VSA vectors for the same spatial locations.

### Current Implementation Analysis

The current `VectorCache` class in `src/cache.py` provides basic vector caching functionality but has several limitations:
- Limited statistics tracking for performance monitoring
- No explicit memory management for large point clouds
- Limited optimization for batch vector computation
- No adaptive cache sizing based on available memory

### Implementation Plan

1. **Enhanced VectorCache Class**

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
        self.total_compute_time = 0.0
        self.total_lookup_time = 0.0
```

2. **Optimized Batch Vector Retrieval with Timing Statistics**

```python
def get_batch_vectors(self, points: torch.Tensor) -> torch.Tensor:
    """
    Get or compute VSA vectors for a batch of points in parallel.
    
    Args:
        points: Tensor of shape [N, 2] containing point coordinates
        
    Returns:
        Tensor of shape [N, vsa_dimensions] containing VSA vectors
    """
    import time
    
    # Start timing for lookup
    lookup_start = time.time()
    
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
    
    # Update lookup timing
    self.total_lookup_time += time.time() - lookup_start
    
    if missing_indices:
        # Start timing for computation
        compute_start = time.time()
        
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
        
        # Update computation timing
        self.total_compute_time += time.time() - compute_start
    
    return result
```

3. **Enhanced Cache Statistics Tracking**

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
        "max_size": self.max_size,
        "compute_time": self.total_compute_time,
        "lookup_time": self.total_lookup_time,
        "total_time": self.total_compute_time + self.total_lookup_time,
        "time_saved": self.cache_hits * (self.total_compute_time / self.cache_misses if self.cache_misses > 0 else 0)
    }
```

4. **Adaptive Cache Management**

```python
def manage_cache_size(self, current_memory_usage: float = None, max_memory_usage: float = None):
    """
    Manage cache size based on memory usage.
    
    Args:
        current_memory_usage: Current memory usage in GB (if None, will be calculated for CUDA)
        max_memory_usage: Maximum memory usage in GB (if None, will be calculated for CUDA)
        
    Returns:
        Number of items removed from cache
    """
    # If no memory usage provided and using CUDA, calculate it
    if current_memory_usage is None and self.device.type == 'cuda':
        current_memory_usage = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
    
    if max_memory_usage is None and self.device.type == 'cuda':
        max_memory_usage = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
    
    # If not using CUDA or no memory info provided, use cache size as proxy
    if current_memory_usage is None or max_memory_usage is None:
        # If cache is more than 90% full, remove 10% of items
        if len(self.cache) > 0.9 * self.max_size:
            items_to_remove = int(0.1 * len(self.cache))
            self._remove_least_used_items(items_to_remove)
            return items_to_remove
        return 0
    
    # Calculate memory usage ratio
    memory_ratio = current_memory_usage / max_memory_usage
    
    # If memory usage is high, remove items from cache
    if memory_ratio > 0.8:  # 80% memory usage
        # Calculate how many items to remove based on memory pressure
        items_to_remove = int((memory_ratio - 0.7) * 10 * len(self.cache))
        items_to_remove = max(1, min(items_to_remove, int(0.5 * len(self.cache))))  # Remove between 1 and 50% of cache
        
        self._remove_least_used_items(items_to_remove)
        return items_to_remove
    
    return 0

def _remove_least_used_items(self, num_items: int):
    """
    Remove the least recently used items from the cache.
    
    Args:
        num_items: Number of items to remove
    """
    if num_items <= 0 or not self.cache:
        return
    
    # Simple implementation: just remove random items
    # In a real implementation, we would track access times and remove least recently used
    keys_to_remove = list(self.cache.keys())[:num_items]
    
    for key in keys_to_remove:
        del self.cache[key]
```

5. **Clear Cache Method**

```python
def clear(self) -> None:
    """Clear the cache to free memory"""
    self.cache = {}
    # Keep statistics for monitoring
```

6. **Precompute Common Vectors**

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
    
    # Test with points that should all be cache hits
    batch3 = cache.get_batch_vectors(points[:2])  # First two points
    
    stats3 = cache.get_cache_stats()
    assert stats3["hits"] > stats2["hits"]
    
    # Test cache management
    cache.manage_cache_size()
    
    stats4 = cache.get_cache_stats()
    assert stats4["cache_size"] <= stats3["cache_size"]
    
    # Test clear
    cache.clear()
    
    stats5 = cache.get_cache_stats()
    assert stats5["cache_size"] == 0
    assert stats5["hits"] == stats4["hits"]  # Should preserve hit/miss stats
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
    
    # Test with different points - should have some cache hits due to discretization
    new_points = torch.rand((n_points, 2), device=device) * 100
    
    start_time = time.time()
    batch3 = cache.get_batch_vectors(new_points)
    batch3_time = time.time() - start_time
    
    stats3 = cache.get_cache_stats()
    print(f"New points batch time: {batch3_time:.4f} seconds")
    print(f"Cache stats after new points: {stats3}")
    
    # Test precomputation
    cache.clear()
    
    start_time = time.time()
    cache.precompute_grid_vectors([0, 100, 0, 100], 1.0)
    precompute_time = time.time() - start_time
    
    stats4 = cache.get_cache_stats()
    print(f"Precomputation time: {precompute_time:.4f} seconds")
    print(f"Cache stats after precomputation: {stats4}")
    
    # Test with points after precomputation
    start_time = time.time()
    batch4 = cache.get_batch_vectors(points)
    batch4_time = time.time() - start_time
    
    stats5 = cache.get_cache_stats()
    print(f"Batch time after precomputation: {batch4_time:.4f} seconds")
    print(f"Cache stats after batch with precomputation: {stats5}")
    
    # Test memory management
    if torch.cuda.is_available():
        # Force high memory usage
        dummy_tensor = torch.zeros((1000, 1000, 100), device=device)
        
        # Manage cache size
        removed = cache.manage_cache_size()
        
        print(f"Removed {removed} items from cache due to memory pressure")
        
        # Clean up
        del dummy_tensor
        torch.cuda.empty_cache()
```

### Integration with Existing Code

The enhanced `VectorCache` class will be a drop-in replacement for the current implementation in `src/cache.py`. It maintains the same core interface but provides improved performance, better statistics tracking, and memory management.

### Expected Outcomes

1. **Improved Performance**: The optimized batch vector computation will be significantly faster, especially for repeated operations.
2. **Better Monitoring**: The enhanced statistics tracking will provide insights into cache performance and potential bottlenecks.
3. **Memory Management**: The adaptive cache management will prevent out-of-memory errors when processing large point clouds.
4. **Precomputation**: The ability to precompute vectors for common grid points will improve performance for incremental processing.

### Next Steps

After implementing the enhanced vector caching, we will proceed to Phase 3, which will focus on implementing the enhanced VSA mapper with memory-aware processing and incremental capabilities.
