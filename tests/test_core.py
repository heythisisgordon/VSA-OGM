"""Basic tests for the core VSA-OGM functionality."""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src directory
from src.mapper import VSAMapper
from src.io import load_pointcloud, save_occupancy_grid, convert_occupancy_grid_to_pointcloud
from src.utils import visualize_occupancy_grid, visualize_class_grid

def test_io_functions():
    """
    Test the I/O functions for loading and converting point clouds and occupancy grids.
    """
    print("Testing I/O functions...")
    
    # Test loading a point cloud
    input_file = "inputs/obstacle_map.npy"
    points, labels = load_pointcloud(input_file)
    
    print(f"Loaded point cloud with {points.shape[0]} points")
    print(f"Point cloud shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test converting an occupancy grid to a point cloud
    grid = np.load(input_file)
    world_bounds = [-50, 50, -50, 50]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    print(f"Converted occupancy grid to point cloud with {points.shape[0]} points")
    print(f"Point cloud shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test saving an occupancy grid
    test_grid = torch.rand((10, 10))
    output_file = "outputs/test_grid.npz"
    os.makedirs("outputs", exist_ok=True)
    
    save_occupancy_grid(test_grid, output_file, metadata={"world_bounds": world_bounds, "resolution": resolution})
    print(f"Saved test grid to {output_file}")
    
    return True

def test_vsa_mapper():
    """
    Test the VSAMapper class with a small example.
    """
    print("Testing VSAMapper...")
    
    # Create a small test grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 1, 0, 1]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Create mapper
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "axis_resolution": 0.2,
        "vsa_dimensions": 1000,  # Use smaller dimensions for testing
        "quadrant_hierarchy": [2],  # Use smaller hierarchy for testing
        "length_scale": 2.0,
        "use_query_normalization": True,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": True
    }
    
    # Use CPU for testing to ensure compatibility
    device = torch.device("cpu")
    mapper = VSAMapper(config, device=device)
    
    # Process observation
    mapper.process_observation(points, labels)
    
    # Get occupancy grid
    occupancy_grid = mapper.get_occupancy_grid()
    
    print(f"Generated occupancy grid with shape: {occupancy_grid.shape}")
    
    # Get class grid
    class_grid = mapper.get_class_grid()
    
    print(f"Generated class grid with shape: {class_grid.shape}")
    
    # Visualize
    os.makedirs("outputs", exist_ok=True)
    
    # Visualize occupancy grid
    visualize_occupancy_grid(
        grid=occupancy_grid,
        output_file="outputs/test_occupancy_grid.png",
        world_bounds=world_bounds,
        colormap="viridis",
        show=False
    )
    
    # Visualize class grid
    visualize_class_grid(
        grid=class_grid,
        output_file="outputs/test_class_grid.png",
        world_bounds=world_bounds,
        show=False
    )
    
    print("Visualizations saved to 'outputs/test_occupancy_grid.png' and 'outputs/test_class_grid.png'")
    
    return True

def test_utils():
    """
    Test the utility functions.
    """
    print("Testing utility functions...")
    
    # Create a test grid
    grid = np.random.rand(20, 20)
    world_bounds = [-10, 10, -10, 10]
    
    # Test visualization functions
    os.makedirs("outputs", exist_ok=True)
    
    # Test occupancy grid visualization
    visualize_occupancy_grid(
        grid=grid,
        output_file="outputs/test_utils_occupancy.png",
        world_bounds=world_bounds,
        colormap="plasma",
        show=False
    )
    
    # Test class grid visualization
    class_grid = np.zeros((20, 20))
    class_grid[5:10, 5:10] = 1  # Occupied
    class_grid[10:15, 10:15] = -1  # Empty
    
    visualize_class_grid(
        grid=class_grid,
        output_file="outputs/test_utils_class.png",
        world_bounds=world_bounds,
        show=False
    )
    
    print("Utility visualizations saved to 'outputs/test_utils_occupancy.png' and 'outputs/test_utils_class.png'")
    
    return True

def run_all_tests():
    """
    Run all tests.
    """
    tests = [
        test_io_functions,
        test_vsa_mapper,
        test_utils
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"{test.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test.__name__}: FAILED with exception: {e}")
            results.append(False)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
