"""Utility functions for VSA-OGM."""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union, List
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_directory(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)

def visualize_occupancy_grid(
    grid: Union[np.ndarray, torch.Tensor],
    output_file: Optional[str] = None,
    world_bounds: Optional[List[float]] = None,
    colormap: str = 'viridis',
    show: bool = False
) -> None:
    """
    Visualize an occupancy grid.
    
    Args:
        grid: Occupancy grid as a numpy array or torch tensor
        output_file: Path to save the visualization (optional)
        world_bounds: World bounds [x_min, x_max, y_min, y_max] (optional)
        colormap: Matplotlib colormap to use
        show: Whether to show the plot
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(grid, torch.Tensor):
        grid_np = grid.detach().cpu().numpy()
    else:
        grid_np = grid
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot the grid
    plt.imshow(grid_np, cmap=colormap, origin='lower')
    plt.colorbar(label='Occupancy Probability')
    
    # Set axis labels and title
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.title('Occupancy Grid Map')
    
    # Set axis limits if world bounds are provided
    if world_bounds is not None:
        x_min, x_max, y_min, y_max = world_bounds
        plt.xlim(0, grid_np.shape[1])
        plt.ylim(0, grid_np.shape[0])
        
        # Add world coordinates as secondary axis
        ax = plt.gca()
        secax_x = ax.secondary_xaxis('top', functions=(
            lambda x: x_min + (x_max - x_min) * x / grid_np.shape[1],
            lambda x: (x - x_min) * grid_np.shape[1] / (x_max - x_min)
        ))
        secax_y = ax.secondary_yaxis('right', functions=(
            lambda y: y_min + (y_max - y_min) * y / grid_np.shape[0],
            lambda y: (y - y_min) * grid_np.shape[0] / (y_max - y_min)
        ))
        secax_x.set_xlabel('X (meters)')
        secax_y.set_ylabel('Y (meters)')
    
    # Save the figure if output_file is provided
    if output_file is not None:
        create_directory(os.path.dirname(os.path.abspath(output_file)))
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_class_grid(
    grid: Union[np.ndarray, torch.Tensor],
    output_file: Optional[str] = None,
    world_bounds: Optional[List[float]] = None,
    show: bool = False
) -> None:
    """
    Visualize a class grid with three classes: empty (-1), unknown (0), occupied (1).
    
    Args:
        grid: Class grid as a numpy array or torch tensor
        output_file: Path to save the visualization (optional)
        world_bounds: World bounds [x_min, x_max, y_min, y_max] (optional)
        show: Whether to show the plot
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(grid, torch.Tensor):
        grid_np = grid.detach().cpu().numpy()
    else:
        grid_np = grid
    
    # Create a custom colormap for the three classes
    colors = [(0.2, 0.2, 0.8), (0.8, 0.8, 0.8), (0.8, 0.2, 0.2)]  # blue, gray, red
    cmap = LinearSegmentedColormap.from_list('class_cmap', colors, N=3)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot the grid
    plt.imshow(grid_np, cmap=cmap, origin='lower', vmin=-1, vmax=1)
    
    # Add colorbar with custom ticks
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Empty', 'Unknown', 'Occupied'])
    
    # Set axis labels and title
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.title('Class Grid Map')
    
    # Set axis limits if world bounds are provided
    if world_bounds is not None:
        x_min, x_max, y_min, y_max = world_bounds
        plt.xlim(0, grid_np.shape[1])
        plt.ylim(0, grid_np.shape[0])
        
        # Add world coordinates as secondary axis
        ax = plt.gca()
        secax_x = ax.secondary_xaxis('top', functions=(
            lambda x: x_min + (x_max - x_min) * x / grid_np.shape[1],
            lambda x: (x - x_min) * grid_np.shape[1] / (x_max - x_min)
        ))
        secax_y = ax.secondary_yaxis('right', functions=(
            lambda y: y_min + (y_max - y_min) * y / grid_np.shape[0],
            lambda y: (y - y_min) * grid_np.shape[0] / (y_max - y_min)
        ))
        secax_x.set_xlabel('X (meters)')
        secax_y.set_ylabel('Y (meters)')
    
    # Save the figure if output_file is provided
    if output_file is not None:
        create_directory(os.path.dirname(os.path.abspath(output_file)))
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()
