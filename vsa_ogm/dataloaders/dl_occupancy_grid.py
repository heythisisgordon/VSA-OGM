import copy
import numpy as np
from omegaconf import DictConfig

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
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize an OccupancyGridDataLoader object with the given configuration.

        Args:
            config (DictConfig): A configuration object containing the following keys:
                - data_dir (str): The path to the .npy file containing the occupancy grid
                - world_bounds (list): Physical bounds of the world [x_min, x_max, y_min, y_max] in meters
                - resolution (float, optional): Resolution of the grid in meters per cell

        Returns:
            None
        """
        self.data_dir = config.data_dir
        self.world_bounds = config.world_bounds
        self.resolution = getattr(config, 'resolution', 0.1)  # Default resolution of 0.1 meters per cell
        
        # Initialize point cloud data
        self.point_cloud = None
        
        # Load the grid and convert to point cloud format
        self.reset()
        
    def reset(self) -> dict:
        """
        Loads the occupancy grid from the .npy file and converts it to a labeled point cloud format.
        
        Returns:
            dict: A dictionary containing the point cloud data with keys:
                - lidar_data: numpy array of shape (N, 2) containing the (x, y) coordinates
                - occupancy: numpy array of shape (N,) containing the binary labels (0 for free, 1 for occupied)
        """
        # Load the occupancy grid from the .npy file
        occupancy_grid = np.load(self.data_dir)
        
        # Get grid dimensions
        grid_height, grid_width = occupancy_grid.shape
        
        # Create arrays to store point cloud data
        grid_coords = np.indices((grid_height, grid_width)).transpose(1, 2, 0)
        num_points = grid_height * grid_width
        
        # Reshape to get a list of [row, col] coordinates
        grid_coords = grid_coords.reshape(num_points, 2)
        
        # Convert grid coordinates to world coordinates using the bottom-left corner as origin (0,0)
        # Note: In grid coordinates, [0,0] is typically the top-left corner
        # We need to flip the y-axis to match the world coordinate system
        x_min, x_max, y_min, y_max = self.world_bounds
        world_width = x_max - x_min
        world_height = y_max - y_min
        
        # Scale grid coordinates to world coordinates
        world_coords = np.zeros_like(grid_coords, dtype=float)
        world_coords[:, 0] = x_min + grid_coords[:, 1] * self.resolution  # x = x_min + col * resolution
        world_coords[:, 1] = y_max - grid_coords[:, 0] * self.resolution  # y = y_max - row * resolution (flip y-axis)
        
        # Get occupancy values (True/False) and convert to binary labels (1/0)
        occupancy_values = occupancy_grid.flatten().astype(int)
        
        # Store the point cloud data
        self.point_cloud = {
            "lidar_data": world_coords,
            "occupancy": occupancy_values
        }
        
        return copy.deepcopy(self.point_cloud)
    
    def step(self) -> dict:
        """
        Returns the point cloud data.
        Since we're working with a static grid, this just returns the same data each time.
        
        Returns:
            dict: A dictionary containing the point cloud data.
        """
        return copy.deepcopy(self.point_cloud)
    
    def max_steps(self) -> int:
        """
        Returns the maximum number of time steps in the dataset.
        Since we're working with a static grid, this is always 1.
        
        Returns:
            int: Always returns 1.
        """
        return 1
