"""
Input/output operations for Sequential VSA-OGM.

This module provides functions for loading and saving data for the Sequential
VSA-OGM system, including point clouds, occupancy grids, and configurations.
"""

import numpy as np
import torch
import os
import json
from typing import Dict, Tuple, List, Union, Optional, Any


def load_point_cloud(filepath: str, device: str = "cpu") -> torch.Tensor:
    """
    Load a point cloud from a file.
    
    Args:
        filepath: Path to the point cloud file (numpy .npy format)
        device: Device to load the point cloud on
        
    Returns:
        Point cloud as tensor of shape (N, D)
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Point cloud file '{filepath}' not found")
        
    # Load point cloud
    points = np.load(filepath)
    
    # Convert to tensor
    points_tensor = torch.from_numpy(points).float().to(device)
    
    return points_tensor


def save_point_cloud(points: Union[np.ndarray, torch.Tensor], filepath: str) -> None:
    """
    Save a point cloud to a file.
    
    Args:
        points: Point cloud as array/tensor of shape (N, D)
        filepath: Path to save the point cloud file (numpy .npy format)
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
    # Save point cloud
    np.save(filepath, points)


def load_occupancy_grid(filepath: str, device: str = "cpu") -> torch.Tensor:
    """
    Load an occupancy grid from a file.
    
    Args:
        filepath: Path to the occupancy grid file (numpy .npy format)
        device: Device to load the occupancy grid on
        
    Returns:
        Occupancy grid as tensor
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Occupancy grid file '{filepath}' not found")
        
    # Load occupancy grid
    grid = np.load(filepath)
    
    # Convert to tensor
    grid_tensor = torch.from_numpy(grid).float().to(device)
    
    return grid_tensor


def save_occupancy_grid(grid: Union[np.ndarray, torch.Tensor], filepath: str) -> None:
    """
    Save an occupancy grid to a file.
    
    Args:
        grid: Occupancy grid as array/tensor
        filepath: Path to save the occupancy grid file (numpy .npy format)
    """
    # Convert to numpy if needed
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
    # Save occupancy grid
    np.save(filepath, grid)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load a configuration from a file.
    
    Args:
        filepath: Path to the configuration file (JSON format)
        
    Returns:
        Configuration dictionary
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file '{filepath}' not found")
        
    # Load configuration
    with open(filepath, 'r') as f:
        config = json.load(f)
        
    return config


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration file (JSON format)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
    # Save configuration
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_labels(filepath: str, device: str = "cpu") -> torch.Tensor:
    """
    Load labels from a file.
    
    Args:
        filepath: Path to the labels file (numpy .npy format)
        device: Device to load the labels on
        
    Returns:
        Labels as tensor
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Labels file '{filepath}' not found")
        
    # Load labels
    labels = np.load(filepath)
    
    # Convert to tensor
    labels_tensor = torch.from_numpy(labels).to(device)
    
    return labels_tensor


def save_labels(labels: Union[np.ndarray, torch.Tensor], filepath: str) -> None:
    """
    Save labels to a file.
    
    Args:
        labels: Labels as array/tensor
        filepath: Path to save the labels file (numpy .npy format)
    """
    # Convert to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
    # Save labels
    np.save(filepath, labels)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from a file.
    
    Args:
        filepath: Path to the results file (JSON format)
        
    Returns:
        Results dictionary
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file '{filepath}' not found")
        
    # Load results
    with open(filepath, 'r') as f:
        results = json.load(f)
        
    return results


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to a file.
    
    Args:
        results: Results dictionary
        filepath: Path to save the results file (JSON format)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
    # Convert numpy arrays and tensors to lists
    results_json = {}
    for key, value in results.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            results_json[key] = value.tolist()
        elif isinstance(value, dict):
            results_json[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    results_json[key][k] = v.tolist()
                else:
                    results_json[key][k] = v
        else:
            results_json[key] = value
        
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)


def create_directory(dirpath: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        dirpath: Path to the directory
    """
    os.makedirs(dirpath, exist_ok=True)


def list_files(dirpath: str, extension: Optional[str] = None) -> List[str]:
    """
    List files in a directory.
    
    Args:
        dirpath: Path to the directory
        extension: Optional file extension to filter by
        
    Returns:
        List of file paths
    """
    # Check if directory exists
    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"Directory '{dirpath}' not found")
        
    # List files
    files = []
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        if os.path.isfile(filepath):
            if extension is None or filename.endswith(extension):
                files.append(filepath)
                
    return files


def load_memory_vectors(filepath: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load memory vectors from a file.
    
    Args:
        filepath: Path to the memory vectors file (numpy .npz format)
        device: Device to load the memory vectors on
        
    Returns:
        Dictionary with memory vectors
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Memory vectors file '{filepath}' not found")
        
    # Load memory vectors
    data = np.load(filepath)
    
    # Convert to tensors
    memory_vectors = {}
    for key in data.files:
        memory_vectors[key] = torch.from_numpy(data[key]).to(device)
        
    return memory_vectors


def save_memory_vectors(memory_vectors: Dict[str, Union[np.ndarray, torch.Tensor]], filepath: str) -> None:
    """
    Save memory vectors to a file.
    
    Args:
        memory_vectors: Dictionary with memory vectors
        filepath: Path to save the memory vectors file (numpy .npz format)
    """
    # Convert to numpy if needed
    memory_vectors_np = {}
    for key, value in memory_vectors.items():
        if isinstance(value, torch.Tensor):
            memory_vectors_np[key] = value.detach().cpu().numpy()
        else:
            memory_vectors_np[key] = value
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
    # Save memory vectors
    np.savez(filepath, **memory_vectors_np)
