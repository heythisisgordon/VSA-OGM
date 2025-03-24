"""
Visualization tools for Sequential VSA-OGM.

This module provides visualization tools for the Sequential VSA-OGM system,
including functions for visualizing occupancy grids, entropy maps, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Dict, Union, Optional, List
import matplotlib.colors as mcolors


def plot_occupancy_grid(grid: Union[np.ndarray, torch.Tensor],
                        x_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                        y_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                        ax: Optional[plt.Axes] = None,
                        cmap: str = 'binary',
                        title: str = 'Occupancy Grid',
                        show_colorbar: bool = True,
                        show_axes: bool = True,
                        figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot an occupancy grid.
    
    Args:
        grid: Occupancy grid as 2D array/tensor
        x_coords: Optional x coordinates for the grid
        y_coords: Optional y coordinates for the grid
        ax: Optional matplotlib axes to plot on
        cmap: Colormap to use
        title: Title for the plot
        show_colorbar: Whether to show a colorbar
        show_axes: Whether to show axes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    if isinstance(x_coords, torch.Tensor):
        x_coords = x_coords.detach().cpu().numpy()
    if isinstance(y_coords, torch.Tensor):
        y_coords = y_coords.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Create extent if coordinates are provided
    extent = None
    if x_coords is not None and y_coords is not None:
        extent = [
            x_coords[0], x_coords[-1],
            y_coords[0], y_coords[-1]
        ]
        
    # Plot grid
    im = ax.imshow(
        grid.T,  # Transpose for correct orientation
        cmap=cmap,
        origin='lower',
        extent=extent,
        interpolation='nearest'
    )
    
    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax)
        
    # Set title and labels
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax.axis('off')
        
    return fig


def plot_entropy_map(entropy: Union[np.ndarray, torch.Tensor],
                    x_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    y_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    ax: Optional[plt.Axes] = None,
                    cmap: str = 'coolwarm',
                    title: str = 'Entropy Map',
                    show_colorbar: bool = True,
                    show_axes: bool = True,
                    figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot an entropy map.
    
    Args:
        entropy: Entropy map as 2D array/tensor
        x_coords: Optional x coordinates for the grid
        y_coords: Optional y coordinates for the grid
        ax: Optional matplotlib axes to plot on
        cmap: Colormap to use
        title: Title for the plot
        show_colorbar: Whether to show a colorbar
        show_axes: Whether to show axes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(entropy, torch.Tensor):
        entropy = entropy.detach().cpu().numpy()
    if isinstance(x_coords, torch.Tensor):
        x_coords = x_coords.detach().cpu().numpy()
    if isinstance(y_coords, torch.Tensor):
        y_coords = y_coords.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Create extent if coordinates are provided
    extent = None
    if x_coords is not None and y_coords is not None:
        extent = [
            x_coords[0], x_coords[-1],
            y_coords[0], y_coords[-1]
        ]
        
    # Plot entropy map
    im = ax.imshow(
        entropy.T,  # Transpose for correct orientation
        cmap=cmap,
        origin='lower',
        extent=extent,
        interpolation='nearest'
    )
    
    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax)
        
    # Set title and labels
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax.axis('off')
        
    return fig


def plot_classification(classification: Union[np.ndarray, torch.Tensor],
                       x_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                       y_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                       ax: Optional[plt.Axes] = None,
                       title: str = 'Classification',
                       show_colorbar: bool = True,
                       show_axes: bool = True,
                       figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot a classification grid.
    
    Args:
        classification: Classification grid as 2D array/tensor (-1=empty, 0=unknown, 1=occupied)
        x_coords: Optional x coordinates for the grid
        y_coords: Optional y coordinates for the grid
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        show_colorbar: Whether to show a colorbar
        show_axes: Whether to show axes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(classification, torch.Tensor):
        classification = classification.detach().cpu().numpy()
    if isinstance(x_coords, torch.Tensor):
        x_coords = x_coords.detach().cpu().numpy()
    if isinstance(y_coords, torch.Tensor):
        y_coords = y_coords.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Create extent if coordinates are provided
    extent = None
    if x_coords is not None and y_coords is not None:
        extent = [
            x_coords[0], x_coords[-1],
            y_coords[0], y_coords[-1]
        ]
        
    # Create custom colormap for classification
    cmap = mcolors.ListedColormap(['red', 'gray', 'green'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
    # Plot classification
    im = ax.imshow(
        classification.T,  # Transpose for correct orientation
        cmap=cmap,
        norm=norm,
        origin='lower',
        extent=extent,
        interpolation='nearest'
    )
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['Empty', 'Unknown', 'Occupied'])
        
    # Set title and labels
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax.axis('off')
        
    return fig


def plot_point_cloud(points: Union[np.ndarray, torch.Tensor],
                    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    ax: Optional[plt.Axes] = None,
                    title: str = 'Point Cloud',
                    show_axes: bool = True,
                    figsize: Tuple[int, int] = (8, 6),
                    s: float = 5.0) -> plt.Figure:
    """
    Plot a point cloud.
    
    Args:
        points: Point cloud as array/tensor of shape (N, 2)
        labels: Optional labels for each point (1=occupied, 0=empty)
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        show_axes: Whether to show axes
        figsize: Figure size
        s: Point size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Plot points
    if labels is not None:
        # Create a colormap for labels
        cmap = mcolors.ListedColormap(['red', 'green'])
        scatter = ax.scatter(
            points[:, 0], points[:, 1],
            c=labels,
            cmap=cmap,
            s=s,
            alpha=0.8
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Occupied'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Empty')
        ]
        ax.legend(handles=legend_elements)
    else:
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=0.8)
        
    # Set title and labels
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
    else:
        ax.axis('off')
        
    return fig


def plot_quadrants(quadrant_bounds: Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]],
                  quadrant_centers: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  ax: Optional[plt.Axes] = None,
                  title: str = 'Quadrants',
                  show_axes: bool = True,
                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot quadrant boundaries and centers.
    
    Args:
        quadrant_bounds: Tuple of (x_bounds, y_bounds) defining quadrant boundaries
        quadrant_centers: Optional tensor of shape (num_quadrants, 2) with quadrant centers
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        show_axes: Whether to show axes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    x_bounds, y_bounds = quadrant_bounds
    if isinstance(x_bounds, torch.Tensor):
        x_bounds = x_bounds.detach().cpu().numpy()
    if isinstance(y_bounds, torch.Tensor):
        y_bounds = y_bounds.detach().cpu().numpy()
    if isinstance(quadrant_centers, torch.Tensor):
        quadrant_centers = quadrant_centers.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Plot quadrant boundaries
    for x in x_bounds:
        ax.axvline(x, color='gray', linestyle='-', alpha=0.5)
    for y in y_bounds:
        ax.axhline(y, color='gray', linestyle='-', alpha=0.5)
        
    # Plot quadrant centers if provided
    if quadrant_centers is not None:
        ax.scatter(
            quadrant_centers[:, 0], quadrant_centers[:, 1],
            color='red',
            marker='x',
            s=50,
            alpha=0.8
        )
        
    # Set title and labels
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
    else:
        ax.axis('off')
        
    # Set limits
    ax.set_xlim(x_bounds[0], x_bounds[-1])
    ax.set_ylim(y_bounds[0], y_bounds[-1])
        
    return fig


def plot_sample_positions(sample_positions: Union[np.ndarray, torch.Tensor],
                         ax: Optional[plt.Axes] = None,
                         title: str = 'Sample Positions',
                         show_axes: bool = True,
                         figsize: Tuple[int, int] = (8, 6),
                         s: float = 5.0) -> plt.Figure:
    """
    Plot sample positions.
    
    Args:
        sample_positions: Sample positions as array/tensor of shape (N, 2)
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        show_axes: Whether to show axes
        figsize: Figure size
        s: Point size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(sample_positions, torch.Tensor):
        sample_positions = sample_positions.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Plot sample positions
    ax.scatter(
        sample_positions[:, 0], sample_positions[:, 1],
        color='blue',
        s=s,
        alpha=0.8
    )
        
    # Set title and labels
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
    else:
        ax.axis('off')
        
    return fig


def plot_combined_results(occupancy_grid: Union[np.ndarray, torch.Tensor],
                         entropy_grid: Union[np.ndarray, torch.Tensor],
                         classification: Union[np.ndarray, torch.Tensor],
                         x_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                         y_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot combined results (occupancy, entropy, classification).
    
    Args:
        occupancy_grid: Occupancy grid as 2D array/tensor
        entropy_grid: Entropy grid as 2D array/tensor
        classification: Classification grid as 2D array/tensor
        x_coords: Optional x coordinates for the grid
        y_coords: Optional y coordinates for the grid
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot occupancy grid
    plot_occupancy_grid(
        occupancy_grid, x_coords, y_coords,
        ax=axes[0],
        title='Occupancy Grid'
    )
    
    # Plot entropy map
    plot_entropy_map(
        entropy_grid, x_coords, y_coords,
        ax=axes[1],
        title='Entropy Map'
    )
    
    # Plot classification
    plot_classification(
        classification, x_coords, y_coords,
        ax=axes[2],
        title='Classification'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_sensor_readings(angles: Union[np.ndarray, torch.Tensor],
                        distances: Union[np.ndarray, torch.Tensor],
                        ax: Optional[plt.Axes] = None,
                        title: str = 'Sensor Readings',
                        show_axes: bool = True,
                        figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot sensor readings in polar coordinates.
    
    Args:
        angles: Ray angles in radians
        distances: Ray distances
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        show_axes: Whether to show axes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy()
    if isinstance(distances, torch.Tensor):
        distances = distances.detach().cpu().numpy()
        
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure
        
    # Plot sensor readings
    ax.plot(angles, distances)
    
    # Fill the area
    ax.fill(angles, distances, alpha=0.3)
        
    # Set title
    ax.set_title(title)
    
    # Hide axes if requested
    if not show_axes:
        ax.axis('off')
        
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save a figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save the figure
        dpi: Resolution in dots per inch
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
