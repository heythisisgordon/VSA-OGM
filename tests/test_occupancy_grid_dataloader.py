import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

# Add the parent directory to the path so we can import the vsa_ogm package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_ogm.dataloaders import OccupancyGridDataLoader
from vsa_ogm.io import load_pointcloud, convert_occupancy_grid_to_pointcloud
from vsa_ogm.mapper import VSAMapper

def test_occupancy_grid_dataloader():
    """
    Test the OccupancyGridDataLoader class by loading the obstacle_map.npy file
    and visualizing the converted point cloud.
    """
    # Create a configuration for the dataloader
    config = OmegaConf.create({
        "data_dir": "inputs/obstacle_map.npy",
        "world_bounds": [-50, 50, -50, 50],  # Adjust based on the grid
        "resolution": 0.1  # Meters per cell
    })
    
    # Create the dataloader
    dataloader = OccupancyGridDataLoader(config)
    
    # Get the point cloud data
    point_cloud = dataloader.reset()
    
    # Extract the coordinates and occupancy values
    coords = point_cloud["lidar_data"]
    occupancy = point_cloud["occupancy"]
    
    # Print some statistics
    print(f"Point cloud shape: {coords.shape}")
    print(f"Occupancy shape: {occupancy.shape}")
    print(f"Number of occupied cells: {np.sum(occupancy)}")
    print(f"Number of free cells: {len(occupancy) - np.sum(occupancy)}")
    
    # Visualize the point cloud
    plt.figure(figsize=(10, 10))
    
    # Plot free cells in blue
    free_mask = occupancy == 0
    plt.scatter(coords[free_mask, 0], coords[free_mask, 1], 
                c='blue', s=1, alpha=0.5, label='Free')
    
    # Plot occupied cells in red
    occupied_mask = occupancy == 1
    plt.scatter(coords[occupied_mask, 0], coords[occupied_mask, 1], 
                c='red', s=1, alpha=0.5, label='Occupied')
    
    plt.title('Occupancy Grid as Point Cloud')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Save the figure
    plt.savefig('occupancy_grid_point_cloud.png', dpi=300)
    print("Visualization saved to 'occupancy_grid_point_cloud.png'")

def test_io_functions():
    """
    Test the I/O functions for loading and converting point clouds and occupancy grids.
    """
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

def test_vsa_mapper():
    """
    Test the VSAMapper class with a small example.
    """
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
    
    mapper = VSAMapper(config)
    
    # Process observation
    mapper.process_observation(points, labels)
    
    # Get occupancy grid
    occupancy_grid = mapper.get_occupancy_grid()
    
    print(f"Generated occupancy grid with shape: {occupancy_grid.shape}")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(grid, origin='lower', cmap='viridis')
    plt.title('Original Grid')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(occupancy_grid.detach().cpu().numpy(), origin='lower', cmap='viridis')
    plt.title('VSA Occupancy Grid')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('vsa_mapper_test.png', dpi=300)
    print("Visualization saved to 'vsa_mapper_test.png'")

if __name__ == "__main__":
    print("Testing OccupancyGridDataLoader...")
    test_occupancy_grid_dataloader()
    
    print("\nTesting I/O functions...")
    test_io_functions()
    
    print("\nTesting VSAMapper...")
    test_vsa_mapper()
