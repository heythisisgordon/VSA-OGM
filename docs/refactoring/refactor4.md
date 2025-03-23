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

In this phase, we will enhance the existing `VSAMapper` class in `src/mapper.py` to incorporate memory-aware processing and improved incremental capabilities. This will enable efficient processing of large point clouds with limited memory resources.

### Current Implementation Analysis

The current `VSAMapper` class in `src/mapper.py` provides basic functionality for processing point clouds, but has several limitations:
- Limited memory management for large point clouds
- Basic incremental processing without optimized sampling
- No explicit handling of very large environments
- Limited performance monitoring and statistics

### Implementation Plan

1. **Enhance VSAMapper Class with Statistics Tracking**

We will add statistics tracking to the existing `VSAMapper` class by adding a new field in the `__init__` method:

```python
def __init__(
    self, 
    config: Dict[str, Any],
    device: Optional[torch.device] = None
):
    # Existing initialization code...
    
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
```

2. **Enhance Memory Monitoring**

We will enhance the existing `check_memory_usage` method to track memory usage statistics:

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
            
            self.vector_cache.clear()
            torch.cuda.empty_cache()
            return True
    
    return False
```

3. **Enhance Observation Processing with Statistics**

We will enhance the existing `process_observation` method to track processing statistics:

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
    
    # Existing processing code...
    
    # Update statistics
    self.stats["process_time"] += time.time() - start_time
    self.stats["total_points_processed"] += points.shape[0]
    
    if self.verbose:
        cache_stats = self.vector_cache.get_cache_stats()
        print(f"Vector cache stats: {cache_stats['hit_rate']*100:.1f}% hit rate, "
              f"{cache_stats['cache_size']}/{cache_stats['max_size']} entries")
        print(f"Processing completed in {time.time() - start_time:.2f} seconds")
```

4. **Enhance Incremental Processing with Safety Margin**

We will enhance the existing `process_incrementally` method to use the `is_region_free` method from the spatial index:

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
    
    # Existing initialization code...
    
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
    
    # Existing processing code...
    
    # Update statistics
    self.stats["incremental_time"] += time.time() - start_time
    self.stats["total_samples_processed"] += positions.shape[0]
    
    if self.verbose:
        print(f"Incremental processing complete. Processed {total_points_processed} points "
              f"from {positions.shape[0]} sample positions")
        print(f"Incremental processing completed in {time.time() - start_time:.2f} seconds")
```

5. **Add Statistics Method**

We will add a new method to retrieve processing statistics:

```python
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
```

### Implementation Notes

This phase focuses on enhancing the existing VSA mapper with memory-aware processing and improved incremental capabilities. The key enhancements include:

1. **Memory Monitoring**: Enhanced memory usage tracking and management to prevent out-of-memory errors.

2. **Optimized Incremental Processing**: The incremental processing approach is optimized with safety margin checking to avoid sampling in occupied areas.

3. **Performance Statistics**: Detailed statistics are collected for performance monitoring and analysis.

The implementation directly enhances the existing `VSAMapper` class rather than creating a new class, which avoids code duplication and maintains a clean codebase.

### Next Steps

After enhancing the VSA mapper with memory-aware processing and improved incremental capabilities, we will proceed to Phase 4, which will focus on implementing the Shannon entropy-based feature extraction capability described in the paper.
