# VSA-OGM Performance Guidelines

This document provides guidelines for optimizing the performance of VSA-OGM.

## Hardware Considerations

VSA-OGM can run on both CPU and GPU, but it is significantly faster on GPU. Here are some hardware considerations:

- **GPU**: A CUDA-capable GPU is recommended for processing large point clouds. VSA-OGM has been tested with NVIDIA GPUs and CUDA 11.7.
- **Memory**: The memory requirements depend on the size of the point cloud and the VSA dimensions. For large point clouds, at least 8GB of GPU memory is recommended.
- **CPU**: For CPU-only processing, a multi-core CPU is recommended. VSA-OGM uses PyTorch's parallelization capabilities to speed up processing on CPU.

## Performance Parameters

VSA-OGM provides several parameters that can be tuned to optimize performance:

### VSA Dimensions

The `vsa_dimensions` parameter controls the dimensionality of the VSA vectors. Higher dimensions provide better representational capacity but require more memory and computation time.

- **Default**: 16000
- **Recommended Range**: 4000-32000
- **Performance Impact**: Reducing dimensions can significantly improve performance at the cost of some accuracy.

```python
# Example: Reduce dimensions for faster processing
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/optimized_grid.npz",
    vsa_dimensions=8000,  # Reduced from default 16000
    verbose=True
)
```

### Batch Size

The `batch_size` parameter controls the number of points processed in each batch. Larger batch sizes can improve performance by reducing overhead, but they also require more memory.

- **Default**: 1000
- **Recommended Range**: 500-5000
- **Performance Impact**: Increasing batch size can improve performance, especially on GPU, but may cause out-of-memory errors if set too high.

```python
# Example: Increase batch size for faster processing
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/optimized_grid.npz",
    batch_size=2000,  # Increased from default 1000
    verbose=True
)
```

### Cache Size

The `cache_size` parameter controls the maximum number of vectors stored in the cache. Larger cache sizes can improve performance by reducing redundant computation, but they also require more memory.

- **Default**: 10000
- **Recommended Range**: 5000-50000
- **Performance Impact**: Increasing cache size can significantly improve performance for point clouds with many repeated coordinates, but may cause out-of-memory errors if set too high.

```python
# Example: Increase cache size for better hit rate
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/optimized_grid.npz",
    cache_size=20000,  # Increased from default 10000
    verbose=True
)
```

### Memory Threshold

The `memory_threshold` parameter controls the maximum fraction of GPU memory that can be used before the cache is cleared. Higher thresholds allow more memory usage but increase the risk of out-of-memory errors.

- **Default**: 0.8 (80% of available GPU memory)
- **Recommended Range**: 0.5-0.9
- **Performance Impact**: Increasing memory threshold can improve performance by allowing more cache entries, but may cause out-of-memory errors if set too high.

```python
# Example: Increase memory threshold for better performance
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/optimized_grid.npz",
    memory_threshold=0.9,  # Increased from default 0.8
    verbose=True
)
```

### Incremental Processing

For large point clouds, incremental processing can significantly reduce memory usage by processing the point cloud in smaller chunks.

- **Parameters**:
  - `incremental`: Whether to use incremental processing
  - `horizon_distance`: Maximum distance from sample point to consider points
  - `sample_resolution`: Resolution for sampling grid
  - `max_samples`: Maximum number of sample positions to process
  - `safety_margin`: Minimum distance from occupied points for sampling

- **Performance Impact**: Incremental processing can significantly reduce memory usage, but may increase processing time. It is recommended for large point clouds that would otherwise cause out-of-memory errors.

```python
# Example: Use incremental processing for large point clouds
result = pointcloud_to_ogm(
    input_file="inputs/large_point_cloud.npy",
    output_file="outputs/incremental_grid.npz",
    incremental=True,
    horizon_distance=10.0,
    sample_resolution=1.0,
    max_samples=100,
    safety_margin=0.5,
    verbose=True
)
```

### Shannon Entropy Parameters

The Shannon entropy feature extraction can be tuned with the following parameters:

- `occupied_disk_radius`: Radius for occupied disk filter in entropy calculation
- `empty_disk_radius`: Radius for empty disk filter in entropy calculation

- **Default**: occupied_disk_radius=2, empty_disk_radius=4
- **Recommended Range**: occupied_disk_radius=1-4, empty_disk_radius=2-8
- **Performance Impact**: Larger disk radii provide better feature extraction but require more computation time.

```python
# Example: Adjust Shannon entropy parameters
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/entropy_grid.npz",
    occupied_disk_radius=3,  # Increased from default 2
    empty_disk_radius=6,     # Increased from default 4
    verbose=True
)
```

## Performance Benchmarking

VSA-OGM includes benchmarking scripts to compare different processing modes and parameters. These can be used to find the optimal parameters for your specific use case.

```python
from vsa_ogm.tests.benchmark import benchmark_processing_modes, benchmark_cache_sizes, benchmark_entropy_parameters

# Benchmark standard vs. incremental processing
benchmark_processing_modes(
    input_file="inputs/obstacle_map.npy",
    world_bounds=[-50, 50, -50, 50]
)

# Benchmark different cache sizes
benchmark_cache_sizes(
    input_file="inputs/obstacle_map.npy",
    world_bounds=[-50, 50, -50, 50]
)

# Benchmark different Shannon entropy parameters
benchmark_entropy_parameters(
    input_file="inputs/obstacle_map.npy",
    world_bounds=[-50, 50, -50, 50]
)
```

## Performance Tips

Here are some additional tips for optimizing performance:

1. **Use GPU**: If available, use a GPU for processing. VSA-OGM is significantly faster on GPU.

2. **Reduce VSA Dimensions**: If processing time is a concern, try reducing the VSA dimensions. Even 4000-8000 dimensions can provide good results for many applications.

3. **Increase Batch Size**: If you have sufficient memory, try increasing the batch size to reduce overhead.

4. **Optimize Cache Size**: Monitor the cache hit rate (available in the stats) and adjust the cache size accordingly. A higher hit rate indicates better cache utilization.

5. **Use Incremental Processing**: For large point clouds, use incremental processing to reduce memory usage.

6. **Adjust Resolution**: If the point cloud is very dense, you can increase the grid resolution to reduce the number of grid cells and improve performance.

7. **Precompute Grid Vectors**: If you're using the VSAMapper class directly, you can precompute grid vectors for frequently used regions:

```python
# Precompute grid vectors for a specific region
mapper.vector_cache.precompute_grid_vectors(
    world_bounds=[-10, 10, -10, 10],
    grid_resolution=0.2
)
```

8. **Monitor Memory Usage**: Use the `verbose` option to monitor memory usage and adjust parameters accordingly.

9. **Benchmark Different Parameters**: Use the benchmarking scripts to find the optimal parameters for your specific use case.

## Example Performance Configurations

Here are some example configurations for different scenarios:

### Fast Processing (Lower Accuracy)

```python
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/fast_grid.npz",
    vsa_dimensions=4000,      # Reduced dimensions
    batch_size=2000,          # Increased batch size
    resolution=0.2,           # Increased resolution (fewer grid cells)
    occupied_disk_radius=1,   # Reduced disk radius
    empty_disk_radius=2,      # Reduced disk radius
    verbose=True
)
```

### High Accuracy (Slower Processing)

```python
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/accurate_grid.npz",
    vsa_dimensions=32000,     # Increased dimensions
    batch_size=1000,          # Default batch size
    resolution=0.05,          # Reduced resolution (more grid cells)
    occupied_disk_radius=4,   # Increased disk radius
    empty_disk_radius=8,      # Increased disk radius
    verbose=True
)
```

### Large Point Cloud (Memory Efficient)

```python
result = pointcloud_to_ogm(
    input_file="inputs/large_point_cloud.npy",
    output_file="outputs/large_grid.npz",
    vsa_dimensions=8000,      # Reduced dimensions
    incremental=True,         # Enable incremental processing
    horizon_distance=10.0,    # Maximum distance from sample point
    sample_resolution=1.0,    # Resolution for sampling grid
    max_samples=200,          # Maximum number of sample positions
    batch_size=1000,          # Default batch size
    cache_size=5000,          # Reduced cache size
    memory_threshold=0.7,     # Reduced memory threshold
    verbose=True
)
```

### Balanced Performance and Accuracy

```python
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/balanced_grid.npz",
    vsa_dimensions=16000,     # Default dimensions
    batch_size=2000,          # Increased batch size
    resolution=0.1,           # Default resolution
    cache_size=20000,         # Increased cache size
    occupied_disk_radius=2,   # Default disk radius
    empty_disk_radius=4,      # Default disk radius
    verbose=True
)
```

## Conclusion

By tuning the parameters described in this document, you can optimize VSA-OGM for your specific use case, balancing performance and accuracy according to your requirements.
