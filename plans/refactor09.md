# Plan to Address Excessive Debug Logging/Visualization

## Problem Statement
Your implementation may be generating visualizations or logging data during processing rather than only at the end, causing significant performance overhead.

## Implementation Plan

### 1. Implement Dedicated Logging System with Runtime Control

```python
class LoggingManager:
    """
    Centralized logging system with performance optimization.
    Controls when logging and visualization occurs to minimize runtime impact.
    """
    
    def __init__(self, enabled: bool = False, log_level: str = "INFO"):
        """
        Initialize the logging manager with runtime controls.
        
        Args:
            enabled: Whether logging is enabled
            log_level: Level of logging (DEBUG, INFO, WARN, ERROR)
        """
        self.enabled = enabled
        self.log_level = log_level
        self.log_data = {}
        self.visualization_data = {}
        
        # Prioritize performance - no logging during processing
        self.defer_logging = True
```

### 2. Create Deferred Visualization System

```python
def store_visualization_data(self, 
                           name: str, 
                           data: torch.Tensor,
                           metadata: dict = None) -> None:
    """
    Store visualization data for deferred processing.
    
    Args:
        name: Name of the visualization
        data: Tensor data to visualize
        metadata: Optional metadata for visualization
    """
    if not self.enabled:
        return
        
    # Store only references to avoid copying large tensors
    self.visualization_data[name] = {
        'data': data,
        'metadata': metadata or {}
    }
```

### 3. Implement Performance-Focused Metrics Collection

```python
def collect_performance_metrics(self, stage_name: str, start_time: float) -> None:
    """
    Collect performance metrics without impacting runtime.
    
    Args:
        stage_name: Name of the processing stage
        start_time: Start time of the stage
    """
    if not self.enabled:
        return
        
    # Only store timing data - no processing during runtime
    elapsed = time.time() - start_time
    
    if 'timing' not in self.log_data:
        self.log_data['timing'] = {}
        
    if stage_name not in self.log_data['timing']:
        self.log_data['timing'][stage_name] = []
        
    self.log_data['timing'][stage_name].append(elapsed)
```

### 4. Create Post-Processing Visualization Generator

```python
def generate_visualizations(self, log_dir: str) -> None:
    """
    Generate all visualizations after processing is complete.
    
    Args:
        log_dir: Directory to save visualizations
    """
    if not self.enabled:
        return
        
    # Only create visualizations after all processing is done
    for name, data in self.visualization_data.items():
        # Skip generation if data is None
        if data['data'] is None:
            continue
            
        # Create visualization path
        vis_path = os.path.join(log_dir, f"{name}.png")
        
        # Generate appropriate visualization based on data type
        if len(data['data'].shape) == 2:
            self._generate_heatmap(data['data'], vis_path, name, data['metadata'])
        elif len(data['data'].shape) == 1:
            self._generate_line_plot(data['data'], vis_path, name, data['metadata'])
            
    # Clear visualization data after generation
    self.visualization_data = {}
```

### 5. Implement Direct Integration with VSAMapper

```python
# In VSAMapper class
def __init__(self, config, world_bounds):
    # Other initialization...
    
    # Initialize logging manager with config settings
    self.logger = LoggingManager(
        enabled=config.get("logging", "enabled", False),
        log_level=config.get("logging", "level", "INFO")
    )
    
def process_point_cloud(self, points, labels=None):
    """Process point cloud with optimized logging"""
    # Start timing
    start_time = time.time()
    
    # Process point cloud without logging
    # ...
    
    # Only collect timing at the end
    self.logger.collect_performance_metrics("process_point_cloud", start_time)
    
    # Generate maps without logging
    start_time = time.time()
    self._generate_maps()
    self.logger.collect_performance_metrics("generate_maps", start_time)
    
    # Store final results for visualization
    if self.logger.enabled:
        self.logger.store_visualization_data(
            "occupancy_grid", 
            self.occupancy_grid.detach().clone(),
            {'coords': self.grid_coords}
        )
        
        self.logger.store_visualization_data(
            "entropy_grid", 
            self.entropy_grid.detach().clone(),
            {'coords': self.grid_coords}
        )
```

### 6. Implement Direct Memory-Efficient Visualization

```python
def _generate_heatmap(self, 
                     data: torch.Tensor, 
                     save_path: str, 
                     title: str,
                     metadata: dict) -> None:
    """
    Generate a heatmap visualization with memory efficiency.
    
    Args:
        data: 2D tensor to visualize
        save_path: Path to save the visualization
        title: Title for the visualization
        metadata: Metadata for the visualization
    """
    # Convert to numpy without copying if possible
    if data.requires_grad:
        data = data.detach()
        
    if data.device.type != 'cpu':
        data_np = data.cpu().numpy()
    else:
        data_np = data.numpy()
        
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create extent if coordinates are provided
    extent = None
    if 'coords' in metadata and metadata['coords'] is not None:
        coords = metadata['coords']
        if 'x' in coords and 'y' in coords:
            x_coords = coords['x']
            y_coords = coords['y']
            
            if isinstance(x_coords, torch.Tensor):
                x_coords = x_coords.cpu().numpy()
            if isinstance(y_coords, torch.Tensor):
                y_coords = y_coords.cpu().numpy()
                
            extent = [
                x_coords[0], x_coords[-1],
                y_coords[0], y_coords[-1]
            ]
    
    # Plot heatmap
    plt.imshow(
        data_np,
        cmap=metadata.get('cmap', 'viridis'),
        origin='lower',
        extent=extent,
        interpolation='nearest'
    )
    
    # Add colorbar
    plt.colorbar()
    
    # Add title
    plt.title(title)
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    plt.close()
```

### 7. Implement Post-Processing Performance Analysis

```python
def generate_performance_report(self, log_dir: str) -> None:
    """
    Generate performance analysis after processing is complete.
    
    Args:
        log_dir: Directory to save reports
    """
    if not self.enabled or 'timing' not in self.log_data:
        return
        
    # Create report path
    report_path = os.path.join(log_dir, "performance_report.txt")
    plot_path = os.path.join(log_dir, "performance_plot.png")
    
    # Generate text report
    with open(report_path, 'w') as f:
        f.write("Performance Report\n")
        f.write("=================\n\n")
        
        for stage, times in self.log_data['timing'].items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            max_time = max(times)
            min_time = min(times)
            
            f.write(f"Stage: {stage}\n")
            f.write(f"  Calls: {len(times)}\n")
            f.write(f"  Total Time: {total_time:.4f} seconds\n")
            f.write(f"  Average Time: {avg_time:.4f} seconds\n")
            f.write(f"  Max Time: {max_time:.4f} seconds\n")
            f.write(f"  Min Time: {min_time:.4f} seconds\n\n")
    
    # Generate performance plot
    plt.figure(figsize=(12, 8))
    
    stages = list(self.log_data['timing'].keys())
    avg_times = [sum(self.log_data['timing'][s]) / len(self.log_data['timing'][s]) for s in stages]
    
    plt.barh(stages, avg_times)
    plt.xlabel('Average Time (seconds)')
    plt.title('Processing Stage Performance')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
```

### 8. Implement Selective Data Collection

```python
def set_collection_focus(self, focus_areas: List[str]) -> None:
    """
    Set focus areas for data collection to minimize overhead.
    
    Args:
        focus_areas: List of areas to focus data collection on
    """
    self.focus_areas = focus_areas
    
def should_collect(self, area: str) -> bool:
    """
    Determine if data should be collected for a specific area.
    
    Args:
        area: Area name
        
    Returns:
        Whether data should be collected
    """
    if not self.enabled:
        return False
        
    if not hasattr(self, 'focus_areas') or self.focus_areas is None:
        return True
        
    return area in self.focus_areas
```

### 9. Implement Class-Based Integration

```python
# In QuadrantMemory class
def __init__(self, world_bounds, quadrant_size, vector_dim, length_scale, device, logger=None):
    # Other initialization...
    
    # Use provided logger or create a dummy logger
    self.logger = logger or LoggingManager(enabled=False)
    
def update_with_points(self, points, labels):
    """Update memory with optimized logging"""
    # Only time the operation
    start_time = time.time()
    
    # Process without logging
    # ...
    
    # Record timing at the end
    self.logger.collect_performance_metrics("update_with_points", start_time)
```

### 10. Implement Final Visualization Generation

```python
# In VSAMapper class
def finalize_processing(self, log_dir: str) -> None:
    """
    Finalize processing and generate visualizations and reports.
    
    Args:
        log_dir: Directory to save logs and visualizations
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate visualizations
    self.logger.generate_visualizations(log_dir)
    
    # Generate performance report
    self.logger.generate_performance_report(log_dir)
    
    # Save final results
    if self.occupancy_grid is not None:
        occupancy_path = os.path.join(log_dir, "occupancy_grid.npy")
        entropy_path = os.path.join(log_dir, "entropy_grid.npy")
        
        # Save as numpy arrays
        np.save(occupancy_path, self.occupancy_grid.cpu().numpy())
        np.save(entropy_path, self.entropy_grid.cpu().numpy())
        
    # Clear logging data
    self.logger.clear()
```

This implementation completely rewrites the logging and visualization system to prioritize runtime performance. Key optimizations include:

1. Deferring all visualization generation until after processing is complete
2. Storing only references to data to avoid unnecessary copying
3. Implementing selective data collection based on focus areas
4. Creating a centralized logging system with runtime controls
5. Separating logging from processing code
6. Implementing memory-efficient visualization generation
7. Providing performance analysis tools for post-processing

These changes eliminate overhead during processing while still providing valuable debugging and visualization capabilities when needed.# Plan to Address Excessive Debug Logging/Visualization

## Problem Statement
Your implementation may be generating visualizations or logging data during processing rather than only at the end, causing significant performance overhead.

## Implementation Plan

### 1. Implement Dedicated Logging System with Runtime Control

```python
class LoggingManager:
    """
    Centralized logging system with performance optimization.
    Controls when logging and visualization occurs to minimize runtime impact.
    """
    
    def __init__(self, enabled: bool = False, log_level: str = "INFO"):
        """
        Initialize the logging manager with runtime controls.
        
        Args:
            enabled: Whether logging is enabled
            log_level: Level of logging (DEBUG, INFO, WARN, ERROR)
        """
        self.enabled = enabled
        self.log_level = log_level
        self.log_data = {}
        self.visualization_data = {}
        
        # Prioritize performance - no logging during processing
        self.defer_logging = True
```

### 2. Create Deferred Visualization System

```python
def store_visualization_data(self, 
                           name: str, 
                           data: torch.Tensor,
                           metadata: dict = None) -> None:
    """
    Store visualization data for deferred processing.
    
    Args:
        name: Name of the visualization
        data: Tensor data to visualize
        metadata: Optional metadata for visualization
    """
    if not self.enabled:
        return
        
    # Store only references to avoid copying large tensors
    self.visualization_data[name] = {
        'data': data,
        'metadata': metadata or {}
    }
```

### 3. Implement Performance-Focused Metrics Collection

```python
def collect_performance_metrics(self, stage_name: str, start_time: float) -> None:
    """
    Collect performance metrics without impacting runtime.
    
    Args:
        stage_name: Name of the processing stage
        start_time: Start time of the stage
    """
    if not self.enabled:
        return
        
    # Only store timing data - no processing during runtime
    elapsed = time.time() - start_time
    
    if 'timing' not in self.log_data:
        self.log_data['timing'] = {}
        
    if stage_name not in self.log_data['timing']:
        self.log_data['timing'][stage_name] = []
        
    self.log_data['timing'][stage_name].append(elapsed)
```

### 4. Create Post-Processing Visualization Generator

```python
def generate_visualizations(self, log_dir: str) -> None:
    """
    Generate all visualizations after processing is complete.
    
    Args:
        log_dir: Directory to save visualizations
    """
    if not self.enabled:
        return
        
    # Only create visualizations after all processing is done
    for name, data in self.visualization_data.items():
        # Skip generation if data is None
        if data['data'] is None:
            continue
            
        # Create visualization path
        vis_path = os.path.join(log_dir, f"{name}.png")
        
        # Generate appropriate visualization based on data type
        if len(data['data'].shape) == 2:
            self._generate_heatmap(data['data'], vis_path, name, data['metadata'])
        elif len(data['data'].shape) == 1:
            self._generate_line_plot(data['data'], vis_path, name, data['metadata'])
            
    # Clear visualization data after generation
    self.visualization_data = {}
```

### 5. Implement Direct Integration with VSAMapper

```python
# In VSAMapper class
def __init__(self, config, world_bounds):
    # Other initialization...
    
    # Initialize logging manager with config settings
    self.logger = LoggingManager(
        enabled=config.get("logging", "enabled", False),
        log_level=config.get("logging", "level", "INFO")
    )
    
def process_point_cloud(self, points, labels=None):
    """Process point cloud with optimized logging"""
    # Start timing
    start_time = time.time()
    
    # Process point cloud without logging
    # ...
    
    # Only collect timing at the end
    self.logger.collect_performance_metrics("process_point_cloud", start_time)
    
    # Generate maps without logging
    start_time = time.time()
    self._generate_maps()
    self.logger.collect_performance_metrics("generate_maps", start_time)
    
    # Store final results for visualization
    if self.logger.enabled:
        self.logger.store_visualization_data(
            "occupancy_grid", 
            self.occupancy_grid.detach().clone(),
            {'coords': self.grid_coords}
        )
        
        self.logger.store_visualization_data(
            "entropy_grid", 
            self.entropy_grid.detach().clone(),
            {'coords': self.grid_coords}
        )
```

### 6. Implement Direct Memory-Efficient Visualization

```python
def _generate_heatmap(self, 
                     data: torch.Tensor, 
                     save_path: str, 
                     title: str,
                     metadata: dict) -> None:
    """
    Generate a heatmap visualization with memory efficiency.
    
    Args:
        data: 2D tensor to visualize
        save_path: Path to save the visualization
        title: Title for the visualization
        metadata: Metadata for the visualization
    """
    # Convert to numpy without copying if possible
    if data.requires_grad:
        data = data.detach()
        
    if data.device.type != 'cpu':
        data_np = data.cpu().numpy()
    else:
        data_np = data.numpy()
        
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create extent if coordinates are provided
    extent = None
    if 'coords' in metadata and metadata['coords'] is not None:
        coords = metadata['coords']
        if 'x' in coords and 'y' in coords:
            x_coords = coords['x']
            y_coords = coords['y']
            
            if isinstance(x_coords, torch.Tensor):
                x_coords = x_coords.cpu().numpy()
            if isinstance(y_coords, torch.Tensor):
                y_coords = y_coords.cpu().numpy()
                
            extent = [
                x_coords[0], x_coords[-1],
                y_coords[0], y_coords[-1]
            ]
    
    # Plot heatmap
    plt.imshow(
        data_np,
        cmap=metadata.get('cmap', 'viridis'),
        origin='lower',
        extent=extent,
        interpolation='nearest'
    )
    
    # Add colorbar
    plt.colorbar()
    
    # Add title
    plt.title(title)
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    plt.close()
```

### 7. Implement Post-Processing Performance Analysis

```python
def generate_performance_report(self, log_dir: str) -> None:
    """
    Generate performance analysis after processing is complete.
    
    Args:
        log_dir: Directory to save reports
    """
    if not self.enabled or 'timing' not in self.log_data:
        return
        
    # Create report path
    report_path = os.path.join(log_dir, "performance_report.txt")
    plot_path = os.path.join(log_dir, "performance_plot.png")
    
    # Generate text report
    with open(report_path, 'w') as f:
        f.write("Performance Report\n")
        f.write("=================\n\n")
        
        for stage, times in self.log_data['timing'].items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            max_time = max(times)
            min_time = min(times)
            
            f.write(f"Stage: {stage}\n")
            f.write(f"  Calls: {len(times)}\n")
            f.write(f"  Total Time: {total_time:.4f} seconds\n")
            f.write(f"  Average Time: {avg_time:.4f} seconds\n")
            f.write(f"  Max Time: {max_time:.4f} seconds\n")
            f.write(f"  Min Time: {min_time:.4f} seconds\n\n")
    
    # Generate performance plot
    plt.figure(figsize=(12, 8))
    
    stages = list(self.log_data['timing'].keys())
    avg_times = [sum(self.log_data['timing'][s]) / len(self.log_data['timing'][s]) for s in stages]
    
    plt.barh(stages, avg_times)
    plt.xlabel('Average Time (seconds)')
    plt.title('Processing Stage Performance')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
```

### 8. Implement Selective Data Collection

```python
def set_collection_focus(self, focus_areas: List[str]) -> None:
    """
    Set focus areas for data collection to minimize overhead.
    
    Args:
        focus_areas: List of areas to focus data collection on
    """
    self.focus_areas = focus_areas
    
def should_collect(self, area: str) -> bool:
    """
    Determine if data should be collected for a specific area.
    
    Args:
        area: Area name
        
    Returns:
        Whether data should be collected
    """
    if not self.enabled:
        return False
        
    if not hasattr(self, 'focus_areas') or self.focus_areas is None:
        return True
        
    return area in self.focus_areas
```

### 9. Implement Class-Based Integration

```python
# In QuadrantMemory class
def __init__(self, world_bounds, quadrant_size, vector_dim, length_scale, device, logger=None):
    # Other initialization...
    
    # Use provided logger or create a dummy logger
    self.logger = logger or LoggingManager(enabled=False)
    
def update_with_points(self, points, labels):
    """Update memory with optimized logging"""
    # Only time the operation
    start_time = time.time()
    
    # Process without logging
    # ...
    
    # Record timing at the end
    self.logger.collect_performance_metrics("update_with_points", start_time)
```

### 10. Implement Final Visualization Generation

```python
# In VSAMapper class
def finalize_processing(self, log_dir: str) -> None:
    """
    Finalize processing and generate visualizations and reports.
    
    Args:
        log_dir: Directory to save logs and visualizations
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate visualizations
    self.logger.generate_visualizations(log_dir)
    
    # Generate performance report
    self.logger.generate_performance_report(log_dir)
    
    # Save final results
    if self.occupancy_grid is not None:
        occupancy_path = os.path.join(log_dir, "occupancy_grid.npy")
        entropy_path = os.path.join(log_dir, "entropy_grid.npy")
        
        # Save as numpy arrays
        np.save(occupancy_path, self.occupancy_grid.cpu().numpy())
        np.save(entropy_path, self.entropy_grid.cpu().numpy())
        
    # Clear logging data
    self.logger.clear()
```

This implementation completely rewrites the logging and visualization system to prioritize runtime performance. Key optimizations include:

1. Deferring all visualization generation until after processing is complete
2. Storing only references to data to avoid unnecessary copying
3. Implementing selective data collection based on focus areas
4. Creating a centralized logging system with runtime controls
5. Separating logging from processing code
6. Implementing memory-efficient visualization generation
7. Providing performance analysis tools for post-processing

These changes eliminate overhead during processing while still providing valuable debugging and visualization capabilities when needed.