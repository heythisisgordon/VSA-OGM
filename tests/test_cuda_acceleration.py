"""Test CUDA acceleration for VSA-OGM."""

import torch
import numpy as np
import time
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_ogm.mapper import VSAMapper
from vsa_ogm.io import load_pointcloud

def test_cuda_acceleration():
    """Test CUDA acceleration for VSA-OGM."""
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
        "axis_resolution": 0.5,
        "vsa_dimensions": 16000,
        "quadrant_hierarchy": [4],
        "length_scale": 2.0,
        "use_query_normalization": True,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": True
    }
    
    # Test with CPU
    print("\n=== Testing with CPU ===")
    device_cpu = torch.device("cpu")
    
    # Load point cloud
    points_cpu, labels_cpu = load_pointcloud(input_file, device_cpu)
    
    # Time CPU initialization
    start_time = time.time()
    mapper_cpu = VSAMapper(config, device=device_cpu)
    cpu_init_time = time.time() - start_time
    print(f"CPU initialization time: {cpu_init_time:.4f} seconds")
    
    # Time CPU processing
    start_time = time.time()
    mapper_cpu.process_observation(points_cpu, labels_cpu)
    cpu_process_time = time.time() - start_time
    print(f"CPU processing time: {cpu_process_time:.4f} seconds")
    
    # Test with CUDA
    print("\n=== Testing with CUDA ===")
    device_cuda = torch.device("cuda")
    
    # Load point cloud
    points_cuda, labels_cuda = load_pointcloud(input_file, device_cuda)
    
    # Time CUDA initialization
    start_time = time.time()
    mapper_cuda = VSAMapper(config, device=device_cuda)
    cuda_init_time = time.time() - start_time
    print(f"CUDA initialization time: {cuda_init_time:.4f} seconds")
    
    # Time CUDA processing
    start_time = time.time()
    mapper_cuda.process_observation(points_cuda, labels_cuda)
    cuda_process_time = time.time() - start_time
    print(f"CUDA processing time: {cuda_process_time:.4f} seconds")
    
    # Calculate speedup
    init_speedup = cpu_init_time / cuda_init_time
    process_speedup = cpu_process_time / cuda_process_time
    
    print("\n=== Results ===")
    print(f"Initialization speedup: {init_speedup:.2f}x")
    print(f"Processing speedup: {process_speedup:.2f}x")
    
    # Check if results are similar
    grid_cpu = mapper_cpu.get_occupancy_grid().cpu()
    grid_cuda = mapper_cuda.get_occupancy_grid().cpu()
    
    max_diff = torch.max(torch.abs(grid_cpu - grid_cuda)).item()
    print(f"Maximum difference between CPU and CUDA results: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("CUDA acceleration is working correctly!")
    else:
        print("Warning: CUDA results differ from CPU results.")
    
    # Print GPU memory usage
    if torch.cuda.is_available():
        print("\n=== GPU Memory Usage ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_cuda_acceleration()
