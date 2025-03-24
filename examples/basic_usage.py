"""
Basic usage example for Sequential VSA-OGM.

This example demonstrates how to use the Sequential VSA-OGM system to process
a point cloud and generate an occupancy grid map.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import VSAMapper, Config
from src.utils import (
    plot_occupancy_grid, 
    plot_entropy_map, 
    plot_classification,
    plot_point_cloud,
    plot_combined_results,
    save_figure
)


def main():
    """Run the basic usage example."""
    print("Sequential VSA-OGM Basic Usage Example")
    
    # Load sample point cloud
    print("Loading sample point cloud...")
    points = np.load("data/sample_point_cloud.npy")
    
    # Create labels (for this example, all points are occupied)
    labels = np.ones(points.shape[0], dtype=np.int32)
    
    # Plot point cloud
    print("Plotting point cloud...")
    fig = plot_point_cloud(points, labels, title="Sample Point Cloud")
    save_figure(fig, "examples/output/point_cloud.png")
    plt.close(fig)
    
    # Define world bounds based on point cloud
    margin = 5.0  # Add margin around points
    min_x, min_y = np.min(points, axis=0) - margin
    max_x, max_y = np.max(points, axis=0) + margin
    world_bounds = (min_x, max_x, min_y, max_y)
    
    # Create custom configuration
    config_dict = {
        "vsa": {
            "dimensions": 1024,
            "length_scale": 1.0,
        },
        "quadrant": {
            "size": 8,
        },
        "sequential": {
            "sample_resolution": 0.5,
            "sensor_range": 10.0,
        },
        "entropy": {
            "disk_radius": 3,
            "occupied_threshold": 0.6,
            "empty_threshold": 0.3,
        },
        "system": {
            "device": "cpu",
            "show_progress": True,
        }
    }
    config = Config(config_dict)
    
    # Create VSA mapper
    print("Creating VSA mapper...")
    mapper = VSAMapper(world_bounds, config)
    
    # Process point cloud
    print("Processing point cloud...")
    mapper.process_point_cloud(points, labels)
    
    # Get results
    print("Getting results...")
    occupancy_grid = mapper.get_occupancy_grid()
    entropy_grid = mapper.get_entropy_grid()
    classification = mapper.get_classification()
    
    # Create output directory if it doesn't exist
    os.makedirs("examples/output", exist_ok=True)
    
    # Plot results
    print("Plotting results...")
    
    # Plot occupancy grid
    fig = plot_occupancy_grid(
        occupancy_grid['grid'],
        occupancy_grid['x_coords'],
        occupancy_grid['y_coords'],
        title="Occupancy Grid"
    )
    save_figure(fig, "examples/output/occupancy_grid.png")
    plt.close(fig)
    
    # Plot entropy map
    fig = plot_entropy_map(
        entropy_grid['grid'],
        entropy_grid['x_coords'],
        entropy_grid['y_coords'],
        title="Entropy Map"
    )
    save_figure(fig, "examples/output/entropy_map.png")
    plt.close(fig)
    
    # Plot classification
    fig = plot_classification(
        classification['grid'],
        classification['x_coords'],
        classification['y_coords'],
        title="Classification"
    )
    save_figure(fig, "examples/output/classification.png")
    plt.close(fig)
    
    # Plot combined results
    fig = plot_combined_results(
        occupancy_grid['grid'],
        entropy_grid['grid'],
        classification['grid'],
        occupancy_grid['x_coords'],
        occupancy_grid['y_coords']
    )
    save_figure(fig, "examples/output/combined_results.png")
    plt.close(fig)
    
    print("Done! Results saved to examples/output/")


if __name__ == "__main__":
    main()
