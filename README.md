# Sequential VSA-OGM

Vector Symbolic Architecture for Occupancy Grid Mapping with Sequential Processing

## Overview

Sequential VSA-OGM is a pythonic implementation for converting static 2D point clouds into probabilistic occupancy grid maps. It utilizes hyperdimensional computing via Vector Symbolic Architectures (VSA) and employs Shannon entropy for feature extraction.

This system processes point clouds by sampling observation points sequentially across a grid, simulating how a robot with sensors would build a map while exploring an environment. At each sample location, only points within sensor range are processed, updating quadrant-based memory vectors.

## Features

- **Efficient Memory Usage**: Constant memory complexity regardless of point cloud size
- **Sequential Processing**: Processes point clouds incrementally from sample positions
- **Quadrant-Based Memory**: Divides the world into quadrants for efficient memory usage
- **Adaptive Dimensionality**: Automatically selects optimal VSA vector dimensions based on environment complexity
- **Shannon Entropy Extraction**: Uses entropy for feature extraction and classification
- **Visualization Tools**: Comprehensive visualization capabilities for results analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- PyTorch
- Matplotlib
- tqdm
- scikit-learn

### Install from Source

```bash
git clone https://github.com/example/sequential-vsa-ogm.git
cd sequential-vsa-ogm
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import numpy as np
from src import VSAMapper, Config

# Load point cloud
points = np.load("data/sample_point_cloud.npy")
labels = np.ones(points.shape[0], dtype=np.int32)  # All points are occupied

# Define world bounds
world_bounds = (-50, 50, -50, 50)  # (xmin, xmax, ymin, ymax)

# Create mapper with default configuration
mapper = VSAMapper(world_bounds)

# Process point cloud
mapper.process_point_cloud(points, labels)

# Get occupancy grid
occupancy_grid = mapper.get_occupancy_grid()

# Get entropy grid
entropy_grid = mapper.get_entropy_grid()

# Get classification
classification = mapper.get_classification()
```

### Custom Configuration

```python
from src import Config

# Create custom configuration
config_dict = {
    "vsa": {
        "dimensions": 1024,           # Dimensionality of VSA vectors
        "length_scale": 1.0,          # Length scale for fractional binding
    },
    "quadrant": {
        "size": 8,                    # Number of quadrants along each axis
    },
    "sequential": {
        "sample_resolution": 0.5,     # Distance between sample points
        "sensor_range": 10.0,         # Maximum range of the simulated sensor
    },
    "entropy": {
        "disk_radius": 3,             # Radius of the disk filter
        "occupied_threshold": 0.6,    # Threshold for occupied classification
        "empty_threshold": 0.3,       # Threshold for empty classification
    },
    "system": {
        "device": "cpu",              # Device to perform calculations on
        "show_progress": True,        # Whether to show progress bars
    }
}

config = Config(config_dict)
mapper = VSAMapper(world_bounds, config)
```

### Visualization

```python
from src.utils import plot_occupancy_grid, plot_entropy_map, plot_classification

# Plot occupancy grid
fig = plot_occupancy_grid(
    occupancy_grid['grid'],
    occupancy_grid['x_coords'],
    occupancy_grid['y_coords'],
    title="Occupancy Grid"
)

# Plot entropy map
fig = plot_entropy_map(
    entropy_grid['grid'],
    entropy_grid['x_coords'],
    entropy_grid['y_coords'],
    title="Entropy Map"
)

# Plot classification
fig = plot_classification(
    classification['grid'],
    classification['x_coords'],
    classification['y_coords'],
    title="Classification"
)
```

## Examples

See the `examples` directory for more detailed examples:

- `basic_usage.py`: Simple example showing core functionality
- `visualization.py`: Examples of different visualization options

## Adaptive Dimensionality

The system now features adaptive dimensionality selection for VSA vectors, which automatically determines the optimal number of dimensions based on environment complexity:

- **Auto-configuration**: When no dimensions are specified, the system calculates the recommended dimensions based on world size, resolution, and quadrant configuration
- **Progressive Dimensionality**: Starts with a lower dimension and increases if mapping quality is insufficient
- **Dimension Validation**: Warns if the selected dimensions are too small for the environment
- **CUDA Memory Management**: Automatically reduces dimensions if CUDA memory is exceeded
- **Performance Monitoring**: Includes tools to analyze the relationship between dimensions and performance metrics

The dimensionality is calculated using the following heuristic:
```python
# Estimate points per dimension
points_per_dim = world_size / resolution
# Estimate total points
total_points = points_per_dim ** 2
# Calculate minimum dimensions needed
min_dims = max(1024, min(200000, total_points / (quadrant_size ** 2) * 16))
# Round to nearest power of 2 for better FFT performance
power = int(np.ceil(np.log2(min_dims)))
dimensions = 2 ** power
```

This approach ensures that the system uses sufficient dimensions for complex environments while maintaining computational efficiency for simpler ones.

## Architecture

The system consists of the following core components:

1. **VSAMapper**: Main entry point for the system
2. **QuadrantMemory**: Memory storage system based on vector symbolic architectures
3. **SequentialProcessor**: Processes point clouds sequentially
4. **SpatialIndex**: Indexes spatial data for efficient querying
5. **EntropyExtractor**: Implements Shannon entropy for feature extraction
6. **VectorOperations**: Performs VSA operations (binding, power, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
