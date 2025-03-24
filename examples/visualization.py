"""
Visualization example for Sequential VSA-OGM.

This example demonstrates the visualization capabilities of the Sequential VSA-OGM system.
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
    plot_quadrants,
    plot_sample_positions,
    save_figure
)


def main():
    """Run the visualization example."""
    print("Sequential VSA-OGM Visualization Example")
    
    # Load sample point cloud
    print("Loading sample point cloud...")
    points = np.load("data/sample_point_cloud.npy")
    
    # Create labels (for this example, all points are occupied)
    labels = np.ones(points.shape[0], dtype=np.int32)
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs("examples/output", exist_ok=True)
    
    # Plot various visualizations
    print("Creating visualizations...")
    
    # 1. Point Cloud Visualization
    fig = plot_point_cloud(
        points, 
        labels, 
        title="Point Cloud",
        figsize=(10, 8)
    )
    save_figure(fig, "examples/output/viz_point_cloud.png")
    plt.close(fig)
    
    # 2. Quadrant Visualization
    quadrant_bounds = mapper.get_quadrant_bounds()
    quadrant_centers = mapper.get_quadrant_centers()
    
    fig = plot_quadrants(
        quadrant_bounds,
        quadrant_centers,
        title="Quadrant Structure",
        figsize=(10, 8)
    )
    save_figure(fig, "examples/output/viz_quadrants.png")
    plt.close(fig)
    
    # 3. Sample Positions Visualization
    sample_positions = mapper.get_sample_positions()
    
    fig = plot_sample_positions(
        sample_positions,
        title="Sample Positions",
        figsize=(10, 8)
    )
    save_figure(fig, "examples/output/viz_sample_positions.png")
    plt.close(fig)
    
    # 4. Combined Point Cloud and Quadrants
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot quadrants
    for x in quadrant_bounds[0]:
        ax.axvline(x, color='gray', linestyle='-', alpha=0.5)
    for y in quadrant_bounds[1]:
        ax.axhline(y, color='gray', linestyle='-', alpha=0.5)
    
    # Plot quadrant centers
    ax.scatter(
        quadrant_centers[:, 0], 
        quadrant_centers[:, 1],
        color='red',
        marker='x',
        s=50,
        alpha=0.8,
        label='Quadrant Centers'
    )
    
    # Plot sample positions
    ax.scatter(
        sample_positions[:, 0], 
        sample_positions[:, 1],
        color='blue',
        s=5,
        alpha=0.3,
        label='Sample Positions'
    )
    
    # Plot points
    cmap = plt.cm.get_cmap('viridis', 2)
    scatter = ax.scatter(
        points[:, 0], 
        points[:, 1],
        c=labels,
        cmap=cmap,
        s=10,
        alpha=0.8,
        label='Points'
    )
    
    # Set title and labels
    ax.set_title("Combined Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    
    # Add legend
    ax.legend()
    
    # Set limits
    ax.set_xlim(world_bounds[0], world_bounds[1])
    ax.set_ylim(world_bounds[2], world_bounds[3])
    
    # Save figure
    save_figure(fig, "examples/output/viz_combined.png")
    plt.close(fig)
    
    # 5. Results Visualization
    occupancy_grid = mapper.get_occupancy_grid()
    entropy_grid = mapper.get_entropy_grid()
    classification = mapper.get_classification()
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot point cloud
    plot_point_cloud(
        points, 
        labels, 
        ax=axes[0, 0],
        title="Point Cloud"
    )
    
    # Plot occupancy grid
    plot_occupancy_grid(
        occupancy_grid['grid'],
        occupancy_grid['x_coords'],
        occupancy_grid['y_coords'],
        ax=axes[0, 1],
        title="Occupancy Grid"
    )
    
    # Plot entropy map
    plot_entropy_map(
        entropy_grid['grid'],
        entropy_grid['x_coords'],
        entropy_grid['y_coords'],
        ax=axes[1, 0],
        title="Entropy Map"
    )
    
    # Plot classification
    plot_classification(
        classification['grid'],
        classification['x_coords'],
        classification['y_coords'],
        ax=axes[1, 1],
        title="Classification"
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, "examples/output/viz_results_grid.png")
    plt.close(fig)
    
    print("Done! Visualizations saved to examples/output/")


if __name__ == "__main__":
    main()
