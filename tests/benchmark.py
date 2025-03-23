"""Performance benchmarking for the VSA-OGM implementation."""

import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src directory
from src.main import pointcloud_to_ogm

def benchmark_processing_modes(
    input_file: str,
    world_bounds: list,
    resolution: float = 0.1,
    vsa_dimensions: int = 8000
):
    """
    Benchmark standard vs. incremental processing.
    
    Args:
        input_file: Path to input point cloud file
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
    """
    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping benchmark.")
        return
    
    # Create output directory
    os.makedirs("outputs/benchmark", exist_ok=True)
    
    # Benchmark standard processing
    print("\n=== Benchmarking Standard Processing ===\n")
    
    standard_output = "outputs/benchmark/standard.npz"
    
    standard_result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=standard_output,
        world_bounds=world_bounds,
        resolution=resolution,
        vsa_dimensions=vsa_dimensions,
        use_cuda=True,
        verbose=True,
        incremental=False,
        save_stats=True
    )
    
    standard_time = standard_result["processing_time"]
    standard_stats = standard_result["stats"]
    
    print(f"Standard processing time: {standard_time:.4f} seconds")
    
    # Benchmark incremental processing
    print("\n=== Benchmarking Incremental Processing ===\n")
    
    incremental_output = "outputs/benchmark/incremental.npz"
    
    incremental_result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=incremental_output,
        world_bounds=world_bounds,
        resolution=resolution,
        vsa_dimensions=vsa_dimensions,
        use_cuda=True,
        verbose=True,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        safety_margin=0.5,
        save_stats=True
    )
    
    incremental_time = incremental_result["processing_time"]
    incremental_stats = incremental_result["stats"]
    
    print(f"Incremental processing time: {incremental_time:.4f} seconds")
    
    # Calculate speedup
    if incremental_time > 0:
        speedup = standard_time / incremental_time
        if speedup > 1:
            print(f"Incremental processing is {speedup:.2f}x faster than standard processing")
        else:
            print(f"Incremental processing is {1/speedup:.2f}x slower than standard processing")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot timing comparison
    plt.subplot(1, 2, 1)
    
    times = [standard_time, incremental_time]
    labels = ["Standard", "Incremental"]
    
    plt.bar(labels, times)
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time Comparison")
    
    # Plot memory usage
    plt.subplot(1, 2, 2)
    
    if "current_memory_gb" in incremental_stats:
        memory = [
            standard_stats.get("current_memory_gb", 0),
            incremental_stats.get("current_memory_gb", 0)
        ]
        
        plt.bar(labels, memory)
        plt.ylabel("Memory Usage (GB)")
        plt.title("Memory Usage Comparison")
    
    plt.tight_layout()
    plt.savefig("outputs/benchmark/processing_comparison.png")
    plt.close()
    
    print("Benchmark visualization saved to 'outputs/benchmark/processing_comparison.png'")
    
    return {
        "standard_time": standard_time,
        "incremental_time": incremental_time,
        "standard_stats": standard_stats,
        "incremental_stats": incremental_stats,
        "speedup": speedup if incremental_time > 0 else 0
    }

def benchmark_cache_sizes(
    input_file: str,
    world_bounds: list,
    resolution: float = 0.1,
    vsa_dimensions: int = 8000
):
    """
    Benchmark different cache sizes.
    
    Args:
        input_file: Path to input point cloud file
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
    """
    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping benchmark.")
        return
    
    # Create output directory
    os.makedirs("outputs/benchmark", exist_ok=True)
    
    # Test different cache sizes
    cache_sizes = [1000, 5000, 10000, 20000]
    
    times = []
    hit_rates = []
    
    for cache_size in cache_sizes:
        print(f"\n=== Benchmarking Cache Size: {cache_size} ===\n")
        
        output_file = f"outputs/benchmark/cache_size_{cache_size}.npz"
        
        result = pointcloud_to_ogm(
            input_file=input_file,
            output_file=output_file,
            world_bounds=world_bounds,
            resolution=resolution,
            vsa_dimensions=vsa_dimensions,
            use_cuda=True,
            verbose=True,
            incremental=True,
            horizon_distance=10.0,
            sample_resolution=1.0,
            max_samples=100,
            cache_size=cache_size,
            save_stats=True
        )
        
        processing_time = result["processing_time"]
        stats = result["stats"]
        
        times.append(processing_time)
        hit_rates.append(stats["cache_hit_rate"] * 100)
        
        print(f"Processing time: {processing_time:.4f} seconds")
        print(f"Cache hit rate: {stats['cache_hit_rate']*100:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot processing time
    plt.subplot(1, 2, 1)
    plt.plot(cache_sizes, times, 'o-')
    plt.xlabel("Cache Size")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time vs. Cache Size")
    
    # Plot cache hit rate
    plt.subplot(1, 2, 2)
    plt.plot(cache_sizes, hit_rates, 'o-')
    plt.xlabel("Cache Size")
    plt.ylabel("Cache Hit Rate (%)")
    plt.title("Cache Hit Rate vs. Cache Size")
    
    plt.tight_layout()
    plt.savefig("outputs/benchmark/cache_size_comparison.png")
    plt.close()
    
    print("Benchmark visualization saved to 'outputs/benchmark/cache_size_comparison.png'")
    
    return {
        "cache_sizes": cache_sizes,
        "times": times,
        "hit_rates": hit_rates
    }

def benchmark_entropy_parameters(
    input_file: str,
    world_bounds: list,
    resolution: float = 0.1,
    vsa_dimensions: int = 8000
):
    """
    Benchmark different Shannon entropy parameters.
    
    Args:
        input_file: Path to input point cloud file
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
    """
    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping benchmark.")
        return
    
    # Create output directory
    os.makedirs("outputs/benchmark", exist_ok=True)
    
    # Test different disk radii combinations
    disk_radii_combinations = [
        (1, 2),  # (occupied, empty)
        (2, 4),
        (3, 6),
        (4, 8)
    ]
    
    results = {}
    
    for occupied_radius, empty_radius in disk_radii_combinations:
        param_str = f"occ{occupied_radius}_emp{empty_radius}"
        output_file = f"outputs/benchmark/entropy_{param_str}.npz"
        
        print(f"\n=== Benchmarking Entropy Parameters: occupied_radius={occupied_radius}, empty_radius={empty_radius} ===\n")
        
        result = pointcloud_to_ogm(
            input_file=input_file,
            output_file=output_file,
            world_bounds=world_bounds,
            resolution=resolution,
            vsa_dimensions=vsa_dimensions,
            use_cuda=True,
            verbose=True,
            incremental=True,
            horizon_distance=10.0,
            sample_resolution=1.0,
            max_samples=100,
            occupied_disk_radius=occupied_radius,
            empty_disk_radius=empty_radius,
            save_entropy_grids=True,
            save_stats=True
        )
        
        processing_time = result["processing_time"]
        stats = result["stats"]
        
        # Get class grid
        class_grid = result["class_grid"]
        
        # Count class distribution
        occupied_count = torch.sum(class_grid == 1).item()
        empty_count = torch.sum(class_grid == -1).item()
        unknown_count = torch.sum(class_grid == 0).item()
        total_count = class_grid.numel()
        
        occupied_percent = 100 * occupied_count / total_count
        empty_percent = 100 * empty_count / total_count
        unknown_percent = 100 * unknown_count / total_count
        
        print(f"Processing time: {processing_time:.4f} seconds")
        print(f"Class distribution:")
        print(f"  - Occupied: {occupied_count} ({occupied_percent:.2f}%)")
        print(f"  - Empty: {empty_count} ({empty_percent:.2f}%)")
        print(f"  - Unknown: {unknown_count} ({unknown_percent:.2f}%)")
        
        # Store results
        results[param_str] = {
            "processing_time": processing_time,
            "occupied_count": occupied_count,
            "empty_count": empty_count,
            "unknown_count": unknown_count,
            "occupied_percent": occupied_percent,
            "empty_percent": empty_percent,
            "unknown_percent": unknown_percent
        }
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot class distribution
    plt.subplot(1, 2, 1)
    
    x = np.arange(len(disk_radii_combinations))
    width = 0.25
    
    occupied_percentages = [results[f"occ{occ}_emp{emp}"]["occupied_percent"] for occ, emp in disk_radii_combinations]
    empty_percentages = [results[f"occ{occ}_emp{emp}"]["empty_percent"] for occ, emp in disk_radii_combinations]
    unknown_percentages = [results[f"occ{occ}_emp{emp}"]["unknown_percent"] for occ, emp in disk_radii_combinations]
    
    plt.bar(x - width, occupied_percentages, width, label='Occupied')
    plt.bar(x, empty_percentages, width, label='Empty')
    plt.bar(x + width, unknown_percentages, width, label='Unknown')
    
    plt.xlabel("Disk Radii Combination")
    plt.ylabel("Percentage (%)")
    plt.title("Class Distribution vs. Entropy Parameters")
    plt.xticks(x, [f"({occ},{emp})" for occ, emp in disk_radii_combinations])
    plt.legend()
    
    # Plot processing time
    plt.subplot(1, 2, 2)
    
    times = [results[f"occ{occ}_emp{emp}"]["processing_time"] for occ, emp in disk_radii_combinations]
    
    plt.bar(x, times)
    plt.xlabel("Disk Radii Combination")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time vs. Entropy Parameters")
    plt.xticks(x, [f"({occ},{emp})" for occ, emp in disk_radii_combinations])
    
    plt.tight_layout()
    plt.savefig("outputs/benchmark/entropy_parameters_comparison.png")
    plt.close()
    
    print("Benchmark visualization saved to 'outputs/benchmark/entropy_parameters_comparison.png'")
    
    # Print comparison summary
    print("\n=== Shannon Entropy Parameter Comparison Summary ===\n")
    print("| Occupied Radius | Empty Radius | Processing Time | Occupied % | Empty % | Unknown % |")
    print("|----------------|--------------|-----------------|------------|---------|-----------|")
    
    for occupied_radius, empty_radius in disk_radii_combinations:
        param_str = f"occ{occupied_radius}_emp{empty_radius}"
        data = results[param_str]
        
        print(f"| {occupied_radius:14d} | {empty_radius:12d} | {data['processing_time']:15.2f}s | {data['occupied_percent']:10.2f}% | {data['empty_percent']:7.2f}% | {data['unknown_percent']:9.2f}% |")
    
    return results

def run_all_benchmarks():
    """Run all benchmarks."""
    # Input file
    input_file = "inputs/obstacle_map.npy"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping benchmarks.")
        return
    
    # World bounds
    world_bounds = [-50, 50, -50, 50]
    
    # Run benchmarks
    print("\n=== Running Processing Modes Benchmark ===\n")
    benchmark_processing_modes(input_file, world_bounds)
    
    print("\n=== Running Cache Sizes Benchmark ===\n")
    benchmark_cache_sizes(input_file, world_bounds)
    
    print("\n=== Running Entropy Parameters Benchmark ===\n")
    benchmark_entropy_parameters(input_file, world_bounds)
    
    print("\nAll benchmarks completed!")

if __name__ == "__main__":
    run_all_benchmarks()
