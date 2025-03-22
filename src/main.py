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
    memory_threshold: float = 0.8
) -> None:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npy file)
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
    """
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
        "memory_threshold": memory_threshold
    }
    
    if verbose:
        print("Initializing VSA mapper...")
    
    # Record processing time
    start_time = time.time()
    
    # Create mapper
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
            max_samples=max_samples
        )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get and save occupancy grid
    if verbose:
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    io.save_occupancy_grid(grid, output_file, metadata={
        "world_bounds": world_bounds, 
        "resolution": resolution,
        "incremental": incremental,
        "horizon_distance": horizon_distance if incremental else None,
        "sample_resolution": sample_resolution if incremental else None,
        "processing_time": processing_time
    })
    
    if verbose:
        print(f"Occupancy grid saved to {output_file}")

def cli_main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid map")
    parser.add_argument("input", help="Input point cloud file (.npy)")
    parser.add_argument("output", help="Output occupancy grid file (.npy)")
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
    
    args = parser.parse_args()
    
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
        memory_threshold=args.memory_threshold
    )

# Allow running as a script
if __name__ == "__main__":
    cli_main()
