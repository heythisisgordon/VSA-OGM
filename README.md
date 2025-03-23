# VSA-OGM - Optimized Occupancy Grid Mapping in 2D

## Program Summary

VSA-OGM is a high-performance library for creating occupancy grid maps from 2D point clouds using Vector Symbolic Architecture (VSA) and hyperdimensional computing. It transforms raw point cloud data into probabilistic occupancy grid maps that represent the environment as a grid of cells, each with a probability of being occupied or empty. This is an adaptation of the original VSA-OGM work to operate on a single, final 2D point cloud by ingesting sequential segments of the point cloud and processing them in a manner that emulates a robot with lidar traversing the same environment.

Key features include:
- **Adaptive Spatial Indexing**: Efficiently query points within spatial regions
- **Optimized Vector Caching**: Reduce redundant computation with LRU-like caching
- **Incremental Processing**: Process point clouds in smaller chunks to reduce memory usage
- **Shannon Entropy Feature Extraction**: Enhance feature detection in noisy environments
- **Memory-Aware Processing**: Monitor and manage memory usage to avoid out-of-memory errors
- **Comprehensive Visualization**: Generate visualizations of occupancy grids and entropy maps
- **GPU Acceleration**: Leverage CUDA for faster processing when available

VSA-OGM is designed for robotics, autonomous navigation, and mapping applications where efficient and accurate environment representation is crucial.

## Background

In this application of bio-inspired vector symbolic architectures, we employ a novel hyperdimensional occupancy grid mapping system with Shannon entropy. For the most in-depth exploration of our experiments and results, please take a look at [our paper](https://arxiv.org/pdf/2408.09066).

*This work was supported under grant 5.21 from the University of Michigan's Automotive Research Center (ARC) and the U.S. Army's Ground Vehicle Systems Center (GVSC).* 

<img src="./docs/assets/toy-sim.gif" width="300" height="300"/> <img src="./docs/assets/vsa-toysim-crop.gif" width="370" height="300" />

## Installation

VSA-OGM is packaged as a Python package. You can install it locally with:

```bash
python -m pip install .
```

It has currently been tested on MacOS and Ubuntu with CPU and CUDA 11.7. Other operating systems and CUDA versions should be supported but it has not been formally tested.

## Usage

### Basic Usage

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map
pointcloud_to_ogm(
    input_file="inputs/obstacle_map.npy",
    output_file="outputs/obstacle_grid.npz",
    world_bounds=[-50, 50, -50, 50],  # Optional
    resolution=0.1,                   # Optional
    verbose=True                      # Optional
)
```

### Advanced Features

VSA-OGM now includes optimized processing with adaptive spatial indexing, vector caching, incremental processing, and Shannon entropy feature extraction:

```python
from vsa_ogm import pointcloud_to_ogm

# Convert a point cloud to an occupancy grid map with incremental processing and Shannon entropy
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
    batch_size=1000,                 # Batch size for processing points
    cache_size=10000,                # Maximum size of vector cache
    memory_threshold=0.8,            # Threshold for GPU memory usage
    occupied_disk_radius=2,          # Radius for occupied disk filter in entropy calculation
    empty_disk_radius=4,             # Radius for empty disk filter in entropy calculation
    save_entropy_grids=True,         # Save entropy grids in output file
    save_stats=True,                 # Save processing statistics
    visualize=True,                  # Generate visualizations
    verbose=True
)

# Access results
occupancy_grid = result["grid"]
class_grid = result["class_grid"]
occupied_entropy = result["occupied_entropy"]
empty_entropy = result["empty_entropy"]
global_entropy = result["global_entropy"]
stats = result["stats"]
```

### Command Line Interface

VSA-OGM also provides a command-line interface:

```bash
# After installing the package
vsa-ogm inputs/obstacle_map.npy outputs/obstacle_grid.npz --verbose

# With incremental processing
vsa-ogm inputs/obstacle_map.npy outputs/incremental_grid.npz --incremental --horizon 10.0 --verbose

# With Shannon entropy and visualization
vsa-ogm inputs/obstacle_map.npy outputs/entropy_grid.npz --incremental --occupied-disk-radius 2 --empty-disk-radius 4 --save-entropy-grids --visualize --verbose
```

Or you can run the module directly:

```bash
python -m vsa_ogm.main inputs/obstacle_map.npy outputs/obstacle_grid.npz --verbose
```

### Examples

Check out the `examples` directory for more usage examples:

```bash
python examples/basic_usage.py
```

## Comprehensive Documentation

VSA-OGM now includes comprehensive documentation to help you get the most out of the library:

- [**API Reference**](docs/api.md): Detailed documentation of all classes and methods
- [**Usage Examples**](docs/examples.md): Examples of how to use VSA-OGM for various tasks
- [**Performance Guidelines**](docs/performance.md): Guidelines for optimizing performance

## Testing and Benchmarking

VSA-OGM includes comprehensive testing and benchmarking capabilities:

### Unit Tests

Run the unit tests to verify the correctness of the implementation:

```bash
python -m tests.test_vsa
```

### Integration Tests

Run the integration tests to verify the end-to-end functionality:

```bash
python -m tests.test_integration
```

### Performance Benchmarking

Run the benchmarking scripts to evaluate performance with different parameters:

```bash
python -m tests.benchmark
```

This will generate benchmark visualizations in the `outputs/benchmark` directory.

## Performance Optimization

VSA-OGM provides several parameters that can be tuned to optimize performance. For detailed guidelines, see the [Performance Guidelines](docs/performance.md).

Key parameters for performance optimization include:

- **VSA Dimensions**: Reduce for faster processing, increase for better accuracy
- **Batch Size**: Increase for better GPU utilization
- **Cache Size**: Increase for better hit rate
- **Incremental Processing**: Use for large point clouds to reduce memory usage
- **Shannon Entropy Parameters**: Adjust disk radii for different feature extraction quality

## Directory Structure

- **src**: Core package with the VSA-OGM implementation
  - **main.py**: Main entry point with function-based API
  - **mapper.py**: Core VSA-OGM algorithm implementation with Shannon entropy
  - **functional.py**: Vector operations for VSA-OGM
  - **spatial.py**: Adaptive spatial indexing for efficient point queries
  - **cache.py**: Vector caching for optimized computation
  - **io.py**: Input/output functions
  - **utils.py**: Utility functions including visualization
- **examples**: Example scripts demonstrating usage
- **tests**: 
  - **test_vsa.py**: Unit tests for VSA-OGM components
  - **test_integration.py**: Integration tests for end-to-end functionality
  - **benchmark.py**: Performance benchmarking scripts
- **docs**: 
  - **api.md**: API reference documentation
  - **examples.md**: Usage examples
  - **performance.md**: Performance guidelines
  - **refactoring/**: Refactoring plans and implementation details

## Datasets

For information about the datasets used in the original experiments, please refer to [our paper](https://arxiv.org/pdf/2408.09066).

## Authors and Contact Information

- **Shay Snyder****: [ssnyde9@gmu.edu](ssnyde9@gmu.edu)
- **Andrew Capodieci**: [acapodieci@neyarobotics.com](acapodieci@neyarobotics.com)
- **David Gorsich**: [david.j.gorsich.civ@army.mil](david.j.gorsich.civ@army.mil)
- **Maryam Parsa**: [mparsa@gmu.edu](mparsa@gmu.edu)

If you have any issues, questions, comments, or concerns about VSA-OGM, please reach out to the corresponding author (**). We will respond as soon as possible.

## Reference and Citation

If you find our work useful in your research endeavors, we would appreciate if you would consider citing [our paper](https://arxiv.org/pdf/2408.09066):

```text
@misc{snyder2024braininspiredprobabilisticoccupancy,
      title={Brain Inspired Probabilistic Occupancy Grid Mapping with Hyperdimensional Computing}, 
      author={Shay Snyder and Andrew Capodieci and David Gorsich and Maryam Parsa},
      year={2024},
      eprint={2408.09066},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.09066}, 
}
```
