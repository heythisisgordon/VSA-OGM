# Plan to Address Vector Dimensionality Issues

## Problem Statement
The current implementation uses a fixed dimensionality of 1024 for VSA vectors, which is too small for complex environments. The original implementation adaptively uses dimensions up to 200,000 for complex tasks.

## Implementation Plan

### 1. Update Configuration System
- Modify `src/config.py` to support adaptive dimensionality based on environment complexity
- Add parameters to calculate recommended dimensions based on:
  - Environment size
  - Desired resolution
  - Number of quadrants

```python
# Add to Config class in src/config.py
def calculate_recommended_dimensions(self, 
                                    world_size: float,
                                    resolution: float,
                                    quadrant_size: int) -> int:
    """Calculate recommended VSA dimensions based on environment parameters"""
    # Estimate points per dimension
    points_per_dim = world_size / resolution
    # Estimate total points
    total_points = points_per_dim ** 2
    # Calculate minimum dimensions needed (using heuristic from original)
    min_dims = int(max(1024, min(200000, total_points / (quadrant_size ** 2) * 16)))
    # Round to nearest power of 2 for better FFT performance
    power = int(np.ceil(np.log2(min_dims)))
    return 2 ** power
```

### 2. Update Initialization Logic
- Modify the `VSAMapper.__init__` method to calculate dimensions if not specified

```python
# In src/mapper.py VSAMapper.__init__
if 'dimensions' not in config.get("vsa"):
    # Auto-calculate dimensions
    world_width = world_bounds[1] - world_bounds[0]
    world_height = world_bounds[3] - world_bounds[2]
    world_size = max(world_width, world_height)
    resolution = self.config.get("sequential", "sample_resolution")
    quadrant_size = self.config.get("quadrant", "size")
    
    recommended_dims = self.config.calculate_recommended_dimensions(
        world_size, resolution, quadrant_size)
    
    self.config.set("vsa", "dimensions", recommended_dims)
    
    if self.config.get("system", "show_progress"):
        print(f"Auto-configured VSA dimensions to {recommended_dims}")
```

### 3. Add Dimension Validation
- Add validation to ensure dimensions are sufficient for the environment

```python
# Add to QuadrantMemory.__init__ in src/quadrant_memory.py
# Validate that dimensions are sufficient
min_points_per_quadrant = (self.quadrant_width * self.quadrant_height) / (length_scale ** 2)
min_recommended_dims = min_points_per_quadrant * 4  # 4x oversampling

if vector_dim < min_recommended_dims:
    print(f"WARNING: Vector dimension {vector_dim} may be too small for this environment.")
    print(f"Recommended minimum: {int(min_recommended_dims)}")
```

### 4. Implement Progressive Dimensionality
- Update processing logic to start with lower dimensions and increase if needed

```python
# Add to VSAMapper.process_point_cloud in src/mapper.py
def process_point_cloud(self, points, labels=None):
    # Start with current dimensions
    current_dims = self.config.get("vsa", "dimensions")
    
    # Process with current dimensions
    try:
        # Existing processing code...
        self._init_components()
        self.process_incrementally()
        
        # Evaluate quality
        quality = self._evaluate_mapping_quality()
        
        # If quality is poor and dimensions can be increased
        if quality < 0.7 and current_dims < 200000:
            # Try with higher dimensions
            new_dims = min(200000, current_dims * 2)
            print(f"Increasing dimensions from {current_dims} to {new_dims} to improve quality")
            
            # Reinitialize with new dimensions
            self.config.set("vsa", "dimensions", new_dims)
            self._init_components()
            self.process_incrementally()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and self.device == "cuda":
            # Fall back to smaller dimensions or CPU
            new_dims = max(1024, current_dims // 2)
            print(f"CUDA memory error. Reducing dimensions to {new_dims}")
            self.config.set("vsa", "dimensions", new_dims)
            self.device = "cpu"
            self.config.set("system", "device", "cpu")
            self._init_components()
            self.process_incrementally()
        else:
            raise
```

### 5. Add Dimension Monitoring and Reporting
- Implement metrics tracking for dimensionality vs performance

```python
# Add to src/utils/metrics.py
def dimension_vs_performance(dimensions_list, metrics_list):
    """Plot performance metrics vs dimensions"""
    plt.figure(figsize=(10, 6))
    for metric_name in metrics_list[0].keys():
        values = [metrics[metric_name] for metrics in metrics_list]
        plt.plot(dimensions_list, values, marker='o', label=metric_name)
    
    plt.xscale('log', base=2)
    plt.xlabel('VSA Dimensions')
    plt.ylabel('Metric Value')
    plt.title('Performance vs VSA Dimensions')
    plt.legend()
    plt.grid(True)
    return plt.gcf()
```

### 6. Update Documentation
- Document dimension selection in README and code comments
- Add examples showing dimension impacts on processing time and accuracy
- Provide guidelines for manual dimension selection

This plan addresses the dimensionality issue by implementing adaptive dimensionality selection while maintaining backward compatibility with existing code. The changes focus on the configuration system and initialization logic without requiring extensive modifications to the core processing algorithms.