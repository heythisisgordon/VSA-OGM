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
        elif len(data.shape) == 2 and data.dtype == bool:  # 2D occupancy grid
            # Convert 2D occupancy grid to point cloud
            world_bounds = [-50, 50, -50, 50]  # Default world bounds
            resolution = 0.1  # Default resolution
            return convert_occupancy_grid_to_pointcloud(data, world_bounds, resolution, device)
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

def load_occupancy_grid(
    filepath: str,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """
    Load an occupancy grid from a .npy or .npz file.
    
    Args:
        filepath: Path to the .npy or .npz file
        device: Device to load the grid to
        
    Returns:
        Tuple of (grid, metadata) where:
            grid: Tensor of shape [H, W] containing occupancy probabilities
            metadata: Dictionary of metadata if available, None otherwise
    """
    # Check file extension
    if filepath.endswith('.npz'):
        # Load with metadata
        data = np.load(filepath)
        grid = data['grid']
        
        # Extract metadata
        metadata = {}
        for key in data.files:
            if key != 'grid':
                metadata[key] = data[key]
    else:
        # Load without metadata
        grid = np.load(filepath)
        metadata = None
    
    # Convert to torch tensor
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Move to device if specified
    if device is not None:
        grid_tensor = grid_tensor.to(device)
    
    return grid_tensor, metadata

def convert_occupancy_grid_to_pointcloud(
    grid: torch.Tensor,
    world_bounds: list,
    resolution: float = 0.1,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an occupancy grid to a point cloud.
    
    Args:
        grid: Tensor of shape [H, W] containing occupancy values
        world_bounds: Physical bounds of the world [x_min, x_max, y_min, y_max] in meters
        resolution: Resolution of the grid in meters per cell
        device: Device to load the point cloud to
        
    Returns:
        Tuple of (points, labels) where:
            points: Tensor of shape [N, 2] containing point coordinates
            labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
    """
    # Convert grid to numpy if it's a torch tensor
    if isinstance(grid, torch.Tensor):
        grid_np = grid.detach().cpu().numpy()
    else:
        grid_np = grid
    
    # Get grid dimensions
    grid_height, grid_width = grid_np.shape
    
    # Create arrays to store point cloud data
    grid_coords = np.indices((grid_height, grid_width)).transpose(1, 2, 0)
    num_points = grid_height * grid_width
    
    # Reshape to get a list of [row, col] coordinates
    grid_coords = grid_coords.reshape(num_points, 2)
    
    # Convert grid coordinates to world coordinates
    x_min, x_max, y_min, y_max = world_bounds
    
    # Scale grid coordinates to world coordinates
    world_coords = np.zeros_like(grid_coords, dtype=float)
    world_coords[:, 0] = x_min + grid_coords[:, 1] * resolution  # x = x_min + col * resolution
    world_coords[:, 1] = y_max - grid_coords[:, 0] * resolution  # y = y_max - row * resolution (flip y-axis)
    
    # Get occupancy values and convert to binary labels
    occupancy_values = grid_np.flatten().astype(int)
    
    # Convert to torch tensors
    points_tensor = torch.tensor(world_coords, dtype=torch.float32)
    labels_tensor = torch.tensor(occupancy_values, dtype=torch.int)
    
    # Move to device if specified
    if device is not None:
        points_tensor = points_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
    
    return points_tensor, labels_tensor
