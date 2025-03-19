import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add the parent directory to the path so we can import the vsa_ogm package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_ogm.dataloaders import OccupancyGridDataLoader

def test_occupancy_grid_dataloader():
    """
    Test the OccupancyGridDataLoader class by loading the obstacle_map.npy file
    and visualizing the converted point cloud.
    """
    # Create a configuration for the dataloader
    config = OmegaConf.create({
        "data_dir": "datasets/obstacle_map.npy",
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
    
    # Show the figure
    plt.show()

if __name__ == "__main__":
    test_occupancy_grid_dataloader()
