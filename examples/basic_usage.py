"""Basic usage example for VSA-OGM."""

import numpy as np
import os
import sys
import time

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the src directory
from src.main import pointcloud_to_ogm
from src.utils import visualize_occupancy_grid, visualize_class_grid

def main():
    """Run a basic example of VSA-OGM."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    output_file = "outputs/obstacle_grid.npz"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Example 1: Standard Processing
    print("\n=== Example 1: Standard Processing ===\n")
    
    start_time = time.time()
    
    # Convert a point cloud to an occupancy grid map
    pointcloud_to_ogm(
        input_file=input_file,
        output_file="outputs/standard_grid.npz",
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=False,
        batch_size=1000,
        cache_size=10000,
        memory_threshold=0.8,
        verbose=True
    )
    
    standard_time = time.time() - start_time
    print(f"Standard processing time: {standard_time:.2f} seconds")
    
    # Example 2: Incremental Processing
    print("\n=== Example 2: Incremental Processing ===\n")
    
    start_time = time.time()
    
    # Convert a point cloud to an occupancy grid map with incremental processing
    pointcloud_to_ogm(
        input_file=input_file,
        output_file="outputs/incremental_grid.npz",
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        batch_size=1000,
        cache_size=10000,
        memory_threshold=0.8,
        verbose=True
    )
    
    incremental_time = time.time() - start_time
    print(f"Incremental processing time: {incremental_time:.2f} seconds")
    
    # Load and visualize the results
    print("\n=== Visualizing Results ===\n")
    
    # Standard grid
    standard_data = np.load("outputs/standard_grid.npz")
    standard_grid = standard_data['grid']
    world_bounds = standard_data['world_bounds']
    
    # Incremental grid
    incremental_data = np.load("outputs/incremental_grid.npz")
    incremental_grid = incremental_data['grid']
    
    # Visualize standard grid
    visualize_occupancy_grid(
        grid=standard_grid,
        output_file='outputs/standard_visualization.png',
        world_bounds=world_bounds,
        colormap='viridis',
        show=False
    )
    
    # Visualize incremental grid
    visualize_occupancy_grid(
        grid=incremental_grid,
        output_file='outputs/incremental_visualization.png',
        world_bounds=world_bounds,
        colormap='viridis',
        show=False
    )
    
    print("Visualizations saved to:")
    print("- 'outputs/standard_visualization.png'")
    print("- 'outputs/incremental_visualization.png'")
    
    # Print performance comparison
    print("\n=== Performance Comparison ===\n")
    print(f"Standard processing time: {standard_time:.2f} seconds")
    print(f"Incremental processing time: {incremental_time:.2f} seconds")
    
    # Calculate speedup/slowdown
    standard_vs_incremental = standard_time / incremental_time if incremental_time > 0 else float('inf')
    
    if standard_vs_incremental > 1:
        print(f"Incremental processing is {standard_vs_incremental:.2f}x faster than standard processing")
    else:
        print(f"Incremental processing is {1/standard_vs_incremental:.2f}x slower than standard processing")

if __name__ == "__main__":
    main()
