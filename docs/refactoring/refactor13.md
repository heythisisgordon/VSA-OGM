# Pythonic Refactoring for VSA-OGM

## Overview

This document outlines a plan to refactor the VSA-OGM project into a clean, Pythonic structure that supports our core use case: converting a 2D point cloud (.npy file) to a probabilistic occupancy grid map. The original code was developed using Jupyter Notebooks, and we want to transition to a proper Python package structure.

## Current Issues

1. The core VSA-OGM algorithm is in the `spl` package instead of `vsa_ogm`
2. Code is organized around Jupyter Notebooks rather than proper Python modules
3. Data loading is scattered across multiple files
4. No clear entry point for the main use case
5. Lack of proper Python package structure

## Simplified Structure

```
vsa-ogm/
├── README.md
├── setup.py                    # Proper Python package setup
├── examples/
│   └── basic_usage.py          # Simple example of point cloud to grid conversion
├── tests/
│   └── test_core.py            # Basic tests
├── src/                        # Source code directory
│   ├── __init__.py             # Package exports and version info
│   ├── main.py                 # Main entry point
│   ├── mapper.py               # Core VSA-OGM algorithm (from spl.mapping)
│   ├── functional.py           # Vector operations (from spl.functional)
│   ├── io.py                   # Input/output functions
│   └── utils.py                # Utility functions
└── docs/
    └── refactoring/
        ├── refactor1.md
        ├── refactor2.md
        ├── refactor11.md
        ├── refactor12.md
        └── refactor13.md
```

## Pythonic Implementation Steps

### 1. Create a Proper Python Package Structure

Update `setup.py` to define a proper installable package:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="vsa_ogm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=[""],
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "vsa-ogm=main:cli_main",
        ],
    },
    python_requires=">=3.8",
    description="Vector Symbolic Architecture for Occupancy Grid Mapping",
    author="Your Name",
)
```

Create a proper `__init__.py` that exports the main functionality:

```python
# src/__init__.py
"""Vector Symbolic Architecture for Occupancy Grid Mapping."""

__version__ = "0.1.0"

from .main import pointcloud_to_ogm
from .mapper import VSAMapper

__all__ = ["pointcloud_to_ogm", "VSAMapper"]
```

### 2. Create a Clean Main Module with Function-Based API

```python
# src/main.py
"""Main entry point for VSA-OGM."""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List, Union
import argparse
import sys

from mapper import VSAMapper
from io import load_pointcloud, save_occupancy_grid

def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    use_cuda: bool = True
) -> None:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npy file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        use_cuda: Whether to use CUDA if available
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    
    # Load point cloud
    points, labels = load_pointcloud(input_file, device)
    
    # Set default world bounds if not provided
    if world_bounds is None:
        # Calculate from point cloud with padding
        x_min, y_min = points.min(dim=0).values.cpu().numpy() - 5.0
        x_max, y_max = points.max(dim=0).values.cpu().numpy() + 5.0
        world_bounds = [float(x_min), float(x_max), float(y_min), float(y_max)]
    
    # Create mapper
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "axis_resolution": resolution * 5,  # Reasonable default
        "vsa_dimensions": 16000,
        "quadrant_hierarchy": [4],
        "length_scale": 2.0,
        "use_query_normalization": True,
        "decision_thresholds": [-0.99, 0.99]
    }
    
    mapper = VSAMapper(config, device=device)
    
    # Process point cloud
    mapper.process_observation(points, labels)
    
    # Get and save occupancy grid
    grid = mapper.get_occupancy_grid()
    save_occupancy_grid(grid, output_file, metadata={"world_bounds": world_bounds, "resolution": resolution})
    
    print(f"Occupancy grid saved to {output_file}")

def cli_main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid map")
    parser.add_argument("input", help="Input point cloud file (.npy)")
    parser.add_argument("output", help="Output occupancy grid file (.npy)")
    parser.add_argument("--bounds", "-b", nargs=4, type=float, help="World bounds [x_min x_max y_min y_max]")
    parser.add_argument("--resolution", "-r", type=float, default=0.1, help="Grid resolution in meters")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    
    args = parser.parse_args()
    
    pointcloud_to_ogm(
        args.input,
        args.output,
        world_bounds=args.bounds,
        resolution=args.resolution,
        use_cuda=not args.cpu
    )

# Allow running as a script
if __name__ == "__main__":
    cli_main()
```

### 3. Create a Clean I/O Module with Type Hints

```python
# src/io.py
"""Input/output functions for VSA-OGM."""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union
import os

def load_pointcloud(
    filepath: str,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a point cloud from a .npy file.
    
    Args:
        filepath: Path to the .npy file
        device: Device to load the point cloud to
        
    Returns:
        Tuple of (points, labels) where:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
    """
    data = np.load(filepath)
    
    # Handle different formats
    if data.ndim == 2:
        if data.shape[1] == 3:  # [x, y, label]
            points = data[:, :2]
            labels = data[:, 2].astype(int)
        elif data.shape[1] == 2:  # [x, y] (assume all occupied)
            points = data
            labels = np.ones(data.shape[0], dtype=int)
        else:
            raise ValueError(f"Unexpected point cloud shape: {data.shape}")
    else:
        raise ValueError(f"Unexpected point cloud dimensions: {data.ndim}")
    
    # Convert to torch tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int)
    
    # Move to device if specified
    if device is not None:
        points_tensor = points_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
    
    return points_tensor, labels_tensor

def save_occupancy_grid(
    grid: torch.Tensor,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save an occupancy grid to a .npy file.
    
    Args:
        grid: Tensor of shape [H, W] containing occupancy probabilities
        filepath: Path to save the .npy file
        metadata: Optional dictionary of metadata to save with the grid
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert to numpy array
    grid_np = grid.detach().cpu().numpy()
    
    # Save with metadata if provided
    if metadata is not None:
        np.savez(filepath, grid=grid_np, **metadata)
    else:
        np.save(filepath, grid_np)
```

### 4. Convert the Mapper Class from Notebook-Style to Pythonic Class

```python
# src/mapper.py
"""Core VSA-OGM mapper implementation."""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from functional import bind, power

class VSAMapper:
    """
    Vector Symbolic Architecture Mapper for Occupancy Grid Mapping.
    
    This class implements the core VSA-OGM algorithm, converting point clouds
    to occupancy grid maps using vector symbolic operations.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the VSA mapper.
        
        Args:
            config: Configuration dictionary
            device: Device to use for computation
        """
        self.config = config
        self.device = device if device is not None else torch.device("cpu")
        
        # Initialize core components
        self._initialize_vectors()
        self._initialize_quadrants()
        
    def _initialize_vectors(self) -> None:
        """Initialize the VSA vectors."""
        # Implementation details here
        pass
        
    def _initialize_quadrants(self) -> None:
        """Initialize the quadrant structure."""
        # Implementation details here
        pass
        
    def process_observation(
        self,
        points: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """
        Process a point cloud observation.
        
        Args:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
        """
        # Implementation details here
        pass
        
    def get_occupancy_grid(self) -> torch.Tensor:
        """
        Get the current occupancy grid.
        
        Returns:
            Tensor of shape [H, W] containing occupancy probabilities
        """
        # Implementation details here
        return torch.zeros((100, 100), device=self.device)  # Placeholder
```

### 5. Create a Simple Example Script

```python
# examples/basic_usage.py
"""Basic usage example for VSA-OGM."""

import sys
sys.path.append("src")
from main import pointcloud_to_ogm

def main():
    """Run a basic example of VSA-OGM."""
    # Convert a point cloud to an occupancy grid map
    pointcloud_to_ogm(
        input_file="data/sample_pointcloud.npy",
        output_file="data/sample_occupancy_grid.npy",
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1
    )

if __name__ == "__main__":
    main()
```

## Pythonic Code Improvements

1. **Use Type Hints**: Add proper type hints to all functions and methods for better IDE support and documentation.

2. **Docstrings**: Add proper docstrings to all modules, classes, and functions following Google or NumPy style.

3. **Module Structure**: Organize code into logical modules with clear responsibilities.

4. **Function-Based API**: Create a clean, function-based API for the main use case.

5. **Command-Line Interface**: Implement a proper CLI using argparse with entry points in setup.py.

6. **Proper Package Structure**: Use `__init__.py` to export the public API and hide implementation details.

7. **Avoid Jupyter-Style Code**: Remove cell-based organization, inline plotting, and other notebook-specific patterns.

8. **Consistent Naming**: Use consistent naming conventions (snake_case for functions/variables, PascalCase for classes).

9. **Avoid Global State**: Encapsulate state in classes rather than using global variables.

10. **Separate Concerns**: Clearly separate I/O, computation, and visualization.

## Migration Approach

1. Create the new `src` directory with the simplified structure
2. Move and adapt the core functionality from the existing code
3. Remove notebook-specific patterns (inline plotting, cell-based organization)
4. Add proper type hints and docstrings
5. Create a clean, function-based API for the main use case
6. Update imports to use relative imports within the src directory
7. Keep the original code intact until the new structure is tested and working

## Expected Benefits

- Clean, Pythonic code structure
- Proper package that can be installed with pip
- Clear, function-based API for the main use case
- Better IDE support with type hints
- Command-line interface for easy usage
- Easier to understand, maintain, and extend
