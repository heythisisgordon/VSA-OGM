"""Main entry point for VSA-OGM."""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List, Union
import argparse
import sys
import os
import time

from .mapper import VSAMapper
from . import io
from . import utils

def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    vsa_dimensions: int = 16000,
    use_cuda: bool = True,
    verbose: bool = False,
    incremental: bool = False,
    horizon_distance: float = 10.0,
    sample_resolution: Optional[float] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    length_scale: float = 2.0,
    min_cell_resolution: Optional[float] = None,
    max_cell_resolution: Optional[float] = None,
    cache_size: int = 10000,
    memory_threshold: float = 0.8,
    safety_margin: float = 0.5,
    occupied_disk_radius: int = 2,
    empty_disk_radius: int = 4,
    save_entropy_grids: bool = False,
    save_stats: bool = False,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npz file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
        use_cuda: Whether to use CUDA if available
        verbose: Whether to print verbose output
        incremental: Whether to use incremental processing
        horizon_distance: Maximum distance from sample point to consider points
        sample_resolution: Resolution for sampling grid (default: 10x resolution)
        max_samples: Maximum number of sample positions to process
        batch_size: Batch size for processing points
        length_scale: Length scale for power operation
        min_cell_resolution: Minimum resolution for spatial indexing (default: 5x resolution)
        max_cell_resolution: Maximum resolution for spatial indexing (default: 20x resolution)
        cache_size: Maximum size of vector cache
        memory_threshold: Threshold for GPU memory usage (0.0-1.0)
        safety_margin: Minimum distance from occupied points for sampling
        occupied_disk_radius: Radius for occupied disk filter in entropy calculation
        empty_disk_radius: Radius for empty disk filter in entropy calculation
        save_entropy_grids: Whether to save entropy grids
        save_stats: Whether to save processing statistics
        visualize: Whether to visualize results
        
    Returns:
        Dictionary with processing results and statistics
    """
    # Record start time
    start_time = time.time()
    
    # Set device (default to CUDA when available)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading point cloud from {input_file}")
    
    # Load point cloud
    points, labels = io.load_pointcloud(input_file, device)
    
    if verbose:
        print(f"Loaded point cloud with {points.shape[0]} points")
    
    # Set default world bounds if not provided
    if world_bounds is None:
        # Calculate from point cloud with padding
        x_min, y_min = points.min(dim=0).values.cpu().numpy() - 5.0
        x_max, y_max = points.max(dim=0).values.cpu().numpy() + 5.0
        world_bounds = [float(x_min), float(x_max), float(y_min), float(y_max)]
        
        if verbose:
            print(f"Automatically determined world bounds: {world_bounds}")
    
    # Set default cell resolutions if not provided
    if min_cell_resolution is None:
        min_cell_resolution = resolution * 5
    
    if max_cell_resolution is None:
        max_cell_resolution = resolution * 20
    
    # Create mapper configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "min_cell_resolution": min_cell_resolution,
        "max_cell_resolution": max_cell_resolution,
        "vsa_dimensions": vsa_dimensions,
        "length_scale": length_scale,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose,
        "batch_size": batch_size,
        "cache_size": cache_size,
        "memory_threshold": memory_threshold,
        "occupied_disk_radius": occupied_disk_radius,
        "empty_disk_radius": empty_disk_radius
    }
    
    # Initialize VSA mapper
    if verbose:
        print("Initializing VSAMapper...")
    
    mapper = VSAMapper(config, device=device)
    
    # Process point cloud
    if verbose:
        print("Processing point cloud...")
    
    mapper.process_observation(points, labels)
    
    # Process incrementally if requested
    if incremental:
        if verbose:
            print("Processing incrementally...")
        
        mapper.process_incrementally(
            horizon_distance=horizon_distance,
            sample_resolution=sample_resolution,
            max_samples=max_samples,
            safety_margin=safety_margin
        )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get occupancy grid
    if verbose:
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    
    # Create metadata
    metadata = {
        "world_bounds": world_bounds, 
        "resolution": resolution,
        "incremental": incremental,
        "horizon_distance": horizon_distance if incremental else None,
        "sample_resolution": sample_resolution if incremental else None,
        "processing_time": processing_time,
        "occupied_disk_radius": occupied_disk_radius,
        "empty_disk_radius": empty_disk_radius
    }
    
    # Get entropy grids if requested
    if save_entropy_grids:
        occupied_entropy = mapper.get_occupied_entropy_grid()
        empty_entropy = mapper.get_empty_entropy_grid()
        global_entropy = mapper.get_global_entropy_grid()
        
        metadata.update({
            "occupied_entropy": occupied_entropy.cpu().numpy(),
            "empty_entropy": empty_entropy.cpu().numpy(),
            "global_entropy": global_entropy.cpu().numpy()
        })
    
    # Add mapper statistics if requested
    if save_stats:
        stats = mapper.get_stats()
        metadata["stats"] = stats
    
    # Save occupancy grid
    io.save_occupancy_grid(grid, output_file, metadata=metadata)
    
    # Visualize results if requested
    if visualize:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Visualize occupancy grid
        visualize_file = os.path.splitext(output_file)[0] + "_visualization.png"
        utils.visualize_occupancy_grid(
            grid=grid,
            output_file=visualize_file,
            world_bounds=world_bounds,
            colormap="viridis",
            show=False
        )
        
        if verbose:
            print(f"Visualization saved to {visualize_file}")
        
        # Visualize class grid
        class_grid = mapper.get_class_grid()
        class_file = os.path.splitext(output_file)[0] + "_class.png"
        utils.visualize_class_grid(
            grid=class_grid,
            output_file=class_file,
            world_bounds=world_bounds,
            show=False
        )
        
        if verbose:
            print(f"Class grid visualization saved to {class_file}")
        
        # Visualize entropy grids
        occupied_entropy = mapper.get_occupied_entropy_grid()
        empty_entropy = mapper.get_empty_entropy_grid()
        global_entropy = mapper.get_global_entropy_grid()
        
        # Visualize occupied entropy
        occupied_entropy_file = os.path.splitext(output_file)[0] + "_occupied_entropy.png"
        utils.visualize_entropy_grid(
            grid=occupied_entropy,
            output_file=occupied_entropy_file,
            world_bounds=world_bounds,
            colormap="plasma",
            show=False
        )
        
        # Visualize empty entropy
        empty_entropy_file = os.path.splitext(output_file)[0] + "_empty_entropy.png"
        utils.visualize_entropy_grid(
            grid=empty_entropy,
            output_file=empty_entropy_file,
            world_bounds=world_bounds,
            colormap="plasma",
            show=False
        )
        
        # Visualize global entropy
        global_entropy_file = os.path.splitext(output_file)[0] + "_global_entropy.png"
        utils.visualize_entropy_grid(
            grid=global_entropy,
            output_file=global_entropy_file,
            world_bounds=world_bounds,
            colormap="viridis",
            show=False
        )
        
        # Create entropy comparison visualization
        entropy_comparison_file = os.path.splitext(output_file)[0] + "_entropy_comparison.png"
        utils.visualize_entropy_comparison(
            occupied_entropy=occupied_entropy,
            empty_entropy=empty_entropy,
            global_entropy=global_entropy,
            output_file=entropy_comparison_file,
            world_bounds=world_bounds,
            show=False
        )
        
        if verbose:
            print(f"Entropy visualizations saved to:")
            print(f"  - {occupied_entropy_file}")
            print(f"  - {empty_entropy_file}")
            print(f"  - {global_entropy_file}")
            print(f"  - {entropy_comparison_file}")
    
    if verbose:
        print(f"Occupancy grid saved to {output_file}")
    
    # Prepare result dictionary
    result = {
        "grid": grid,
        "class_grid": mapper.get_class_grid(),
        "occupied_entropy": mapper.get_occupied_entropy_grid(),
        "empty_entropy": mapper.get_empty_entropy_grid(),
        "global_entropy": mapper.get_global_entropy_grid(),
        "metadata": metadata,
        "processing_time": processing_time,
        "stats": mapper.get_stats()
    }
    
    return result

def cli_main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid map")
    parser.add_argument("input", help="Input point cloud file (.npy)")
    parser.add_argument("output", help="Output occupancy grid file (.npz)")
    
    # Basic options
    parser.add_argument("--bounds", "-b", nargs=4, type=float, help="World bounds [x_min x_max y_min y_max]")
    parser.add_argument("--resolution", "-r", type=float, default=0.1, help="Grid resolution in meters")
    parser.add_argument("--dimensions", "-d", type=int, default=16000, help="VSA dimensions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Processing options
    parser.add_argument("--incremental", "-i", action="store_true", help="Use incremental processing")
    parser.add_argument("--horizon", type=float, default=10.0, help="Horizon distance for incremental processing")
    parser.add_argument("--sample-resolution", type=float, help="Resolution for sampling grid in incremental processing")
    parser.add_argument("--max-samples", type=int, help="Maximum number of sample positions for incremental processing")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing points")
    parser.add_argument("--length-scale", type=float, default=2.0, help="Length scale for power operation")
    parser.add_argument("--min-cell-resolution", type=float, help="Minimum resolution for spatial indexing")
    parser.add_argument("--max-cell-resolution", type=float, help="Maximum resolution for spatial indexing")
    parser.add_argument("--cache-size", type=int, default=10000, help="Maximum size of vector cache")
    parser.add_argument("--memory-threshold", type=float, default=0.8, help="Threshold for GPU memory usage (0.0-1.0)")
    parser.add_argument("--safety-margin", type=float, default=0.5, help="Minimum distance from occupied points for sampling")
    
    # Shannon entropy options
    parser.add_argument("--occupied-disk-radius", type=int, default=2, help="Radius for occupied disk filter in entropy calculation")
    parser.add_argument("--empty-disk-radius", type=int, default=4, help="Radius for empty disk filter in entropy calculation")
    parser.add_argument("--save-entropy-grids", action="store_true", help="Save entropy grids in output file")
    parser.add_argument("--save-stats", "-s", action="store_true", help="Save processing statistics")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    
    args = parser.parse_args()
    
    # Call the main function with CLI arguments
    pointcloud_to_ogm(
        args.input,
        args.output,
        world_bounds=args.bounds,
        resolution=args.resolution,
        vsa_dimensions=args.dimensions,
        use_cuda=not args.cpu,
        verbose=args.verbose,
        incremental=args.incremental,
        horizon_distance=args.horizon,
        sample_resolution=args.sample_resolution,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        length_scale=args.length_scale,
        min_cell_resolution=args.min_cell_resolution,
        max_cell_resolution=args.max_cell_resolution,
        cache_size=args.cache_size,
        memory_threshold=args.memory_threshold,
        safety_margin=args.safety_margin,
        occupied_disk_radius=args.occupied_disk_radius,
        empty_disk_radius=args.empty_disk_radius,
        save_entropy_grids=args.save_entropy_grids,
        save_stats=args.save_stats,
        visualize=args.visualize
    )

def basic_example():
    """Run a basic example using the VSA mapper."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    output_file = "outputs/basic_grid.npz"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    print("\n=== Basic VSA Mapper Example ===\n")
    
    # Convert a point cloud to an occupancy grid map
    result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=output_file,
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=False,
        visualize=True,
        verbose=True
    )
    
    # Print statistics
    stats = result["stats"]
    print("\n=== VSA Mapper Statistics ===")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Process time: {stats['process_time']:.2f} seconds")
    print(f"Total points processed: {stats['total_points_processed']}")
    print(f"Points per second: {stats['points_per_second']:.2f}")
    print(f"Cache hit rate: {stats['cache_hit_rate']*100:.2f}%")
    
    return result

def incremental_example():
    """Run an example using incremental processing."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    output_file = "outputs/incremental_grid.npz"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    print("\n=== Incremental Processing Example ===\n")
    
    # Convert a point cloud to an occupancy grid map using incremental processing
    result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=output_file,
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        safety_margin=0.5,
        visualize=True,
        verbose=True
    )
    
    # Print statistics
    stats = result["stats"]
    print("\n=== VSA Mapper Statistics ===")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Process time: {stats['process_time']:.2f} seconds")
    print(f"Incremental time: {stats['incremental_time']:.2f} seconds")
    print(f"Total points processed: {stats['total_points_processed']}")
    print(f"Total samples processed: {stats['total_samples_processed']}")
    print(f"Points per second: {stats['points_per_second']:.2f}")
    print(f"Cache hit rate: {stats['cache_hit_rate']*100:.2f}%")
    
    return result

def compare_entropy_parameters():
    """Compare the performance of different Shannon entropy parameters."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    output_dir = "outputs/entropy_comparison"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Shannon Entropy Parameter Comparison ===\n")
    
    # Define parameter combinations to test
    disk_radii_combinations = [
        (1, 2),  # (occupied, empty)
        (2, 4),
        (3, 6),
        (4, 8)
    ]
    
    results = {}
    
    # Process with each parameter combination
    for occupied_radius, empty_radius in disk_radii_combinations:
        param_str = f"occ{occupied_radius}_emp{empty_radius}"
        output_file = f"{output_dir}/grid_{param_str}.npz"
        
        print(f"\n--- Testing occupied_radius={occupied_radius}, empty_radius={empty_radius} ---\n")
        
        # Process with current parameters
        result = pointcloud_to_ogm(
            input_file=input_file,
            output_file=output_file,
            world_bounds=[-50, 50, -50, 50],
            resolution=0.1,
            incremental=True,
            horizon_distance=10.0,
            sample_resolution=1.0,
            max_samples=100,
            occupied_disk_radius=occupied_radius,
            empty_disk_radius=empty_radius,
            save_entropy_grids=True,
            save_stats=True,
            visualize=True,
            verbose=True
        )
        
        # Store results
        results[param_str] = {
            "processing_time": result["processing_time"],
            "stats": result["stats"]
        }
        
        # Create comparison visualization
        global_entropy = result["global_entropy"]
        class_grid = result["class_grid"]
        
        # Count class distribution
        occupied_count = torch.sum(class_grid == 1).item()
        empty_count = torch.sum(class_grid == -1).item()
        unknown_count = torch.sum(class_grid == 0).item()
        total_count = class_grid.numel()
        
        occupied_percent = 100 * occupied_count / total_count
        empty_percent = 100 * empty_count / total_count
        unknown_percent = 100 * unknown_count / total_count
        
        # Print statistics
        print(f"Class distribution:")
        print(f"  - Occupied: {occupied_count} ({occupied_percent:.2f}%)")
        print(f"  - Empty: {empty_count} ({empty_percent:.2f}%)")
        print(f"  - Unknown: {unknown_count} ({unknown_percent:.2f}%)")
        
        results[param_str].update({
            "occupied_count": occupied_count,
            "empty_count": empty_count,
            "unknown_count": unknown_count,
            "occupied_percent": occupied_percent,
            "empty_percent": empty_percent,
            "unknown_percent": unknown_percent
        })
    
    # Print comparison summary
    print("\n=== Shannon Entropy Parameter Comparison Summary ===\n")
    print("| Occupied Radius | Empty Radius | Processing Time | Occupied % | Empty % | Unknown % |")
    print("|----------------|--------------|-----------------|------------|---------|-----------|")
    
    for occupied_radius, empty_radius in disk_radii_combinations:
        param_str = f"occ{occupied_radius}_emp{empty_radius}"
        data = results[param_str]
        
        print(f"| {occupied_radius:14d} | {empty_radius:12d} | {data['processing_time']:15.2f}s | {data['occupied_percent']:10.2f}% | {data['empty_percent']:7.2f}% | {data['unknown_percent']:9.2f}% |")
    
    return results

def main():
    """Main entry point when run as a module."""
    if len(sys.argv) > 1:
        # If arguments are provided, run CLI
        cli_main()
    else:
        # If no arguments, run example
        print("No arguments provided. Running examples...")
        print("For CLI usage, run with --help flag.")
        
        # Ask user which example to run
        print("\nAvailable examples:")
        print("1. Basic processing")
        print("2. Incremental processing")
        print("3. Compare different Shannon entropy parameters")
        
        choice = input("\nEnter your choice (1-3, or 0 to exit): ")
        
        if choice == "1":
            # Run basic example
            basic_example()
        elif choice == "2":
            # Run incremental example
            incremental_example()
        elif choice == "3":
            # Run entropy parameter comparison
            compare_entropy_parameters()
        elif choice == "0":
            print("Exiting...")
        else:
            print("Invalid choice. Exiting...")

# Allow running as a script
if __name__ == "__main__":
    main()
