"""Basic usage example for VSA-OGM."""

import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the src directory
from src.main import pointcloud_to_ogm
from src.utils import visualize_occupancy_grid

def main():
    """Run a basic example of VSA-OGM."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    output_file = "outputs/obstacle_grid.npz"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Convert a point cloud to an occupancy grid map
    pointcloud_to_ogm(
        input_file=input_file,
        output_file=output_file,
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        verbose=True
    )
    
    # Load and visualize the result
    data = np.load(output_file)
    grid = data['grid']
    world_bounds = data['world_bounds']
    
    # Use the utility function to visualize the grid
    visualize_occupancy_grid(
        grid=grid,
        output_file='outputs/occupancy_grid_visualization.png',
        world_bounds=world_bounds,
        colormap='viridis',
        show=True
    )
    print("Visualization saved to 'outputs/occupancy_grid_visualization.png'")

if __name__ == "__main__":
    main()
