"""Main entry point for VSA-OGM."""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List, Union
import argparse
import sys
import os

from .mapper import VSAMapper
from . import io

def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    axis_resolution: float = 0.5,
    vsa_dimensions: int = 16000,
    use_cuda: bool = True,
    verbose: bool = False
) -> None:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npy file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        axis_resolution: Resolution for axis vectors (typically 5x grid resolution)
        vsa_dimensions: Dimensionality of VSA vectors
        use_cuda: Whether to use CUDA if available
        verbose: Whether to print verbose output
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
    
    # Create mapper
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "axis_resolution": axis_resolution,
        "vsa_dimensions": vsa_dimensions,
        "quadrant_hierarchy": [4],
        "length_scale": 2.0,
        "use_query_normalization": True,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose
    }
    
    if verbose:
        print("Initializing VSA mapper...")
    
    mapper = VSAMapper(config, device=device)
    
    # Process point cloud
    if verbose:
        print("Processing point cloud...")
    
    mapper.process_observation(points, labels)
    
    # Get and save occupancy grid
    if verbose:
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    io.save_occupancy_grid(grid, output_file, metadata={
        "world_bounds": world_bounds, 
        "resolution": resolution
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
    parser.add_argument("--axis-resolution", "-a", type=float, default=0.5, help="Axis resolution in meters")
    parser.add_argument("--dimensions", "-d", type=int, default=16000, help="VSA dimensions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    pointcloud_to_ogm(
        args.input,
        args.output,
        world_bounds=args.bounds,
        resolution=args.resolution,
        axis_resolution=args.axis_resolution,
        vsa_dimensions=args.dimensions,
        use_cuda=not args.cpu,
        verbose=args.verbose
    )

# Allow running as a script
if __name__ == "__main__":
    cli_main()
