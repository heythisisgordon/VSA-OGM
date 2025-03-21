# Simplified Plan for Implementing OccupancyGridDataLoader

## Overview
This document outlines a simplified plan for implementing a new dataloader class that will allow VSA-OGM to process pre-generated 2D occupancy grids as input. We'll focus on creating a minimal viable implementation to test the basic pipeline first.

## Implementation Steps

### 1. Create Basic OccupancyGridDataLoader Class
Create a new file `vsa_ogm/dataloaders/dl_occupancy_grid.py` with a class that:
- Loads a 2D occupancy grid from a .npy file
- Converts the grid to a labeled point cloud format
- Provides an interface compatible with existing dataloaders

```python
class OccupancyGridDataLoader:
    """
    A data loader class for loading 2D occupancy grid data from a .npy file
    and converting it to a labeled point cloud format.
    
    Args:
        config (DictConfig): Configuration dictionary containing:
            - data_dir (str): Path to the .npy file containing the occupancy grid
            - world_bounds (list): Physical bounds of the world [x_min, x_max, y_min, y_max] in meters
            - resolution (float, optional): Resolution of the grid in meters per cell
    """
```

### 2. Implement Grid to Point Cloud Conversion
Develop methods to:
- Convert grid coordinates to world coordinates based on resolution and world bounds, using a corner of the grid as the origin (0,0)
- Perform a 1-for-1 conversion: every cell in the grid becomes a point in the point cloud
- Create a labeled point cloud where each point has (x, y) coordinates and a binary label (0 for free, 1 for occupied)

### 3. Update Dataloader Interface
Implement the standard dataloader interface methods:
- `__init__`: Initialize with configuration
- `reset`: Load the grid and convert to point cloud format
- `step`: Return the same data (since we're working with a static grid)
- `max_steps`: Return 1 since we have a single grid

### 4. Update Dataloader Registry
Update `vsa_ogm/dataloaders/__init__.py` to include the new dataloader:

```python
from .dl_csv import CSVDataLoader
from .dl_pickle import PickleDataLoader
from .dl_toysim import ToySimDataLoader
from .dl_occupancy_grid import OccupancyGridDataLoader
```

### 5. Update Functional Module
Modify `vsa_ogm/dataloaders/functional.py` to support the new dataloader:

```python
def load_single_data(config: DictConfig) -> tuple:
    """
    Load a single data set based on the provided configuration.
    """
    # Existing code...
    
    elif config.data.dataset_name == "occupancy_grid":
        dataloader = OccupancyGridDataLoader(config.data.occupancy_grid)
        world_size = config.data.occupancy_grid.world_bounds
    
    # Rest of the function...
```

### 6. Minimal Configuration
Add a simple configuration section for the occupancy grid dataset:

```python
BASE_CONFIG: dict = {
    # Existing config...
    "data": {
        "dataset_name": "occupancy_grid",
        "test_split": 0.1,
        "occupancy_grid": {
            "data_dir": "datasets/obstacle_map.npy",
            "world_bounds": [-50, 50, -50, 50],  # To be adjusted based on the grid
            "resolution": 0.1   # Meters per cell, to be adjusted
        },
    },
    # Rest of config...
}
```

### 7. Basic Testing
- Create a simple test script to verify the dataloader works correctly
- Visualize the converted point cloud to ensure it represents the original grid
- Test integration with the VSA-OGM system

## Key Considerations

### Resolution and Scale
- The resolution parameter will map grid cells to physical world coordinates
- This needs to be aligned with the VSA-OGM length_scale parameter
- Use a corner of the grid as the origin (0,0) for the coordinate system

### World Bounds
- Set world bounds to match the physical scale of the environment
- This is critical for proper integration with the VSA-OGM system

## Next Steps
1. Test with obstacle_map.npy
2. Adjust parameters (resolution, length_scale) as needed
3. Evaluate the results
4. Add more sophisticated features only after the basic pipeline is working
