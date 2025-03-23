# VSA-OGM Usage Examples

This document provides examples of how to use VSA-OGM for various tasks.

## Basic Usage

The simplest way to use VSA-OGM is to convert a point cloud to an occupancy grid map:

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/obstacle_grid.npz",
    world_bounds=[-50, 50, -50, 50],  # Optional
    resolution=0.1,                   # Optional
    verbose=True                      # Optional
)

# Access the occupancy grid
occupancy_grid = result["grid"]
```

## Incremental Processing

For large point clouds, you can use incremental processing to reduce memory usage:

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map with incremental processing
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/incremental_grid.npz",
    world_bounds=[-50, 50, -50, 50],
    resolution=0.1,
    incremental=True,                # Enable incremental processing
    horizon_distance=10.0,           # Maximum distance from sample points
    sample_resolution=1.0,           # Resolution for sampling grid
    max_samples=100,                 # Maximum number of sample positions
    safety_margin=0.5,               # Minimum distance from occupied points for sampling
    verbose=True
)

# Access the occupancy grid
occupancy_grid = result["grid"]
```

## Shannon Entropy Feature Extraction

VSA-OGM includes Shannon entropy feature extraction for improved feature detection:

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map with Shannon entropy
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/entropy_grid.npz",
    world_bounds=[-50, 50, -50, 50],
    resolution=0.1,
    occupied_disk_radius=2,          # Radius for occupied disk filter in entropy calculation
    empty_disk_radius=4,             # Radius for empty disk filter in entropy calculation
    save_entropy_grids=True,         # Save entropy grids in output file
    verbose=True
)

# Access the entropy grids
occupied_entropy = result["occupied_entropy"]
empty_entropy = result["empty_entropy"]
global_entropy = result["global_entropy"]
```

## Visualization

VSA-OGM can generate visualizations of the occupancy grid and entropy grids:

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map with visualization
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/visualized_grid.npz",
    world_bounds=[-50, 50, -50, 50],
    resolution=0.1,
    visualize=True,                  # Generate visualizations
    verbose=True
)
```

This will generate the following visualizations:
- `outputs/visualized_grid_visualization.png`: Occupancy grid visualization
- `outputs/visualized_grid_class.png`: Class grid visualization

If Shannon entropy is enabled, it will also generate:
- `outputs/visualized_grid_occupied_entropy.png`: Occupied entropy grid visualization
- `outputs/visualized_grid_empty_entropy.png`: Empty entropy grid visualization
- `outputs/visualized_grid_global_entropy.png`: Global entropy grid visualization
- `outputs/visualized_grid_entropy_comparison.png`: Comparison of all entropy grids

## Performance Optimization

VSA-OGM includes several parameters for performance optimization:

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map with performance optimization
result = pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/optimized_grid.npz",
    world_bounds=[-50, 50, -50, 50],
    resolution=0.1,
    vsa_dimensions=8000,             # Reduce dimensions for faster processing
    batch_size=2000,                 # Increase batch size for faster processing
    cache_size=20000,                # Increase cache size for better hit rate
    memory_threshold=0.9,            # Increase memory threshold for better performance
    verbose=True
)
```

## Advanced Usage: Direct Access to VSAMapper

For more control over the mapping process, you can use the VSAMapper class directly:

```python
import torch
from vsa_ogm.mapper import VSAMapper
from vsa_ogm.io import load_pointcloud, save_occupancy_grid

# Load point cloud
points, labels = load_pointcloud("inputs/obstacle_map.npy")

# Create mapper configuration
config = {
    "world_bounds": [-50, 50, -50, 50],
    "resolution": 0.1,
    "min_cell_resolution": 0.5,
    "max_cell_resolution": 2.0,
    "vsa_dimensions": 16000,
    "length_scale": 2.0,
    "decision_thresholds": [-0.99, 0.99],
    "verbose": True,
    "batch_size": 1000,
    "cache_size": 10000,
    "memory_threshold": 0.8,
    "occupied_disk_radius": 2,
    "empty_disk_radius": 4
}

# Create mapper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mapper = VSAMapper(config, device=device)

# Process observation
mapper.process_observation(points, labels)

# Process incrementally
mapper.process_incrementally(
    horizon_distance=10.0,
    sample_resolution=1.0,
    max_samples=100,
    safety_margin=0.5
)

# Get occupancy grid
occupancy_grid = mapper.get_occupancy_grid()

# Get class grid
class_grid = mapper.get_class_grid()

# Get entropy grids
occupied_entropy = mapper.get_occupied_entropy_grid()
empty_entropy = mapper.get_empty_entropy_grid()
global_entropy = mapper.get_global_entropy_grid()

# Get statistics
stats = mapper.get_stats()
print(f"Total time: {stats['total_time']:.2f} seconds")
print(f"Points per second: {stats['points_per_second']:.2f}")
print(f"Cache hit rate: {stats['cache_hit_rate']*100:.2f}%")

# Save occupancy grid
save_occupancy_grid(occupancy_grid, "outputs/advanced_grid.npz")
```

## Command Line Interface

VSA-OGM provides a command-line interface for easy use:

```bash
# Basic usage
vsa-ogm inputs/obstacle_map.npy outputs/obstacle_grid.npz --verbose

# With incremental processing
vsa-ogm inputs/obstacle_map.npy outputs/incremental_grid.npz --incremental --horizon 10.0 --verbose

# With Shannon entropy and visualization
vsa-ogm inputs/obstacle_map.npy outputs/entropy_grid.npz --incremental --occupied-disk-radius 2 --empty-disk-radius 4 --save-entropy-grids --visualize --verbose

# With performance optimization
vsa-ogm inputs/obstacle_map.npy outputs/optimized_grid.npz --dimensions 8000 --batch-size 2000 --cache-size 20000 --memory-threshold 0.9 --verbose
```

## Benchmarking

VSA-OGM includes benchmarking scripts to compare different processing modes and parameters:

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

This will generate benchmark visualizations in the `outputs/benchmark` directory.
