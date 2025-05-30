"""Test CUDA-only acceleration for VSA-OGM."""

import torch
import numpy as np
import time
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mapper import VSAMapper
from src.io import load_pointcloud

def test_cuda_only():
    """Test CUDA-only acceleration for VSA-OGM."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return
    
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    
    # Load a sample point cloud
    input_file = "inputs/obstacle_map.npy"
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping test.")
        return
    
    # Set world bounds
    world_bounds = [-50, 50, -50, 50]
    
    # Create configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": 0.1,
        "min_cell_resolution": 0.5,
        "max_cell_resolution": 2.0,
        "vsa_dimensions": 8000,  # Reduced dimensions to avoid OOM errors
        "length_scale": 2.0,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": True,
        "batch_size": 1000,
        "cache_size": 10000,
        "memory_threshold": 0.8
    }
    
    # Test with CUDA
    print("\n=== Testing with CUDA ===")
    device_cuda = torch.device("cuda")
    
    # Load point cloud
    points_cuda, labels_cuda = load_pointcloud(input_file, device_cuda)
    print(f"Loaded point cloud with {points_cuda.shape[0]} points")
    
    # Time CUDA initialization
    print("\nInitializing VSAMapper...")
    start_time = time.time()
    mapper_cuda = VSAMapper(config, device=device_cuda)
    cuda_init_time = time.time() - start_time
    print(f"CUDA initialization time: {cuda_init_time:.4f} seconds")
    
    # Time CUDA processing
    print("\nProcessing observation...")
    start_time = time.time()
    mapper_cuda.process_observation(points_cuda, labels_cuda)
    cuda_process_time = time.time() - start_time
    print(f"CUDA processing time: {cuda_process_time:.4f} seconds")
    
    # Print GPU memory usage
    print("\n=== GPU Memory Usage ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print("\nCUDA acceleration test completed successfully!")

if __name__ == "__main__":
    test_cuda_only()
