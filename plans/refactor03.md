# Plan to Address Sequential vs. Batch Processing Issues

## Problem Statement
The original VSA-OGM uses batch operations with PyTorch's tensor operations for efficient processing, while the current implementation processes points one by one in loops, leading to significant performance bottlenecks.

## Implementation Plan

### 1. Complete Restructuring of Sequential Processing Architecture

Replace the entire `SequentialProcessor` class with a vectorized batch-oriented design that processes point clouds in a single pass where possible:

```python
class BatchProcessor:
    """
    Processes point clouds in a batch-oriented manner using tensor operations.
    """
    
    def __init__(self, 
                 world_bounds: Tuple[float, float, float, float],
                 sample_resolution: float = 0.2,
                 device: str = "cpu") -> None:
        self.world_bounds = world_bounds
        self.sample_resolution = sample_resolution
        self.device = device
        
        # Pre-compute the grid structure once
        self._build_processing_grid()
    
    def _build_processing_grid(self) -> None:
        """Build the processing grid structure using tensor operations"""
        # Calculate grid dimensions
        x_min, x_max, y_min, y_max = self.world_bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Create coordinate grids
        self.x_coords = torch.arange(
            x_min, x_max, self.sample_resolution, device=self.device
        )
        self.y_coords = torch.arange(
            y_min, y_max, self.sample_resolution, device=self.device
        )
        
        # Create meshgrid
        self.xx, self.yy = torch.meshgrid(self.x_coords, self.y_coords, indexing='ij')
        
        # Pre-allocate grid for batch processing
        self.grid_points = torch.stack([self.xx.flatten(), self.yy.flatten()], dim=1)
        self.grid_shape = self.xx.shape
```

### 2. Implement Direct Point Cloud Processing

```python
def process_point_cloud(self, 
                       points: torch.Tensor,
                       labels: torch.Tensor,
                       memory_updater: Callable) -> None:
    """
    Process a full point cloud in a single batch operation.
    
    Args:
        points: Point cloud tensor of shape (N, 2)
        labels: Binary labels tensor of shape (N)
        memory_updater: Callback function to update memory with processed points
    """
    # Ensure inputs are on the correct device
    points = points.to(self.device)
    labels = labels.to(self.device)
    
    # No sequential processing - directly update memory with the entire batch
    memory_updater(points, labels)
```

### 3. Implement Efficient Spatial Processing

```python
def get_spatial_masks(self, 
                     query_points: torch.Tensor,
                     data_points: torch.Tensor,
                     radius: float) -> torch.Tensor:
    """
    Get masks for points within radius of each query point.
    
    Args:
        query_points: Query points tensor of shape (Q, 2)
        data_points: Data points tensor of shape (N, 2)
        radius: Search radius
        
    Returns:
        Mask tensor of shape (Q, N) where mask[i, j] is True if 
        data_points[j] is within radius of query_points[i]
    """
    # Calculate pairwise distances efficiently
    Q = query_points.shape[0]
    N = data_points.shape[0]
    
    # Use broadcasting to calculate distances
    query_expanded = query_points.unsqueeze(1)  # Shape: (Q, 1, 2)
    data_expanded = data_points.unsqueeze(0)    # Shape: (1, N, 2)
    
    # Calculate squared distances
    distances_squared = torch.sum((query_expanded - data_expanded) ** 2, dim=2)
    
    # Create mask of points within radius
    return distances_squared <= radius ** 2
```

### 4. Implement Efficient Grid Processing

```python
def process_grid(self,
                points: torch.Tensor,
                labels: torch.Tensor,
                radius: float,
                process_fn: Callable) -> None:
    """
    Process all grid positions with respect to a point cloud.
    
    Args:
        points: Point cloud tensor of shape (N, 2)
        labels: Binary labels tensor of shape (N)
        radius: Processing radius
        process_fn: Function to process each grid position
    """
    # Get masks for points visible from each grid position
    masks = self.get_spatial_masks(self.grid_points, points, radius)
    
    # Process all grid positions in parallel
    for i in range(0, len(self.grid_points), 1000):  # Process in chunks to avoid OOM
        end = min(i + 1000, len(self.grid_points))
        chunk_masks = masks[i:end]
        
        # For each grid position, get visible points and labels
        for j in range(chunk_masks.shape[0]):
            visible_mask = chunk_masks[j]
            
            if torch.any(visible_mask):
                visible_points = points[visible_mask]
                visible_labels = labels[visible_mask]
                grid_pos = self.grid_points[i+j]
                
                # Process this grid position
                process_fn(grid_pos, visible_points, visible_labels)
```

### 5. Create Direct Vector Operations for Processing

```python
def create_update_function(self, quadrant_memory):
    """
    Create a function to update quadrant memory in batches.
    
    Args:
        quadrant_memory: The quadrant memory object to update
        
    Returns:
        Update function for batch processing
    """
    def update_memory(points, labels):
        # Directly update memory in one batch operation
        quadrant_memory.update_with_points(points, labels)
    
    return update_memory
```

### 6. Implement Grid-Based Memory Calculation

Rather than using the original sequential approach that calculates similarity scores individually, implement a grid-based approach that computes all similarity scores at once:

```python
def calculate_grid_probabilities(self, 
                                quadrant_memory,
                                resolution: float = None) -> Dict[str, torch.Tensor]:
    """
    Calculate occupancy probabilities for the entire grid in one operation.
    
    Args:
        quadrant_memory: The quadrant memory to query
        resolution: Optional custom resolution (uses default if None)
        
    Returns:
        Dictionary with probability grids for both occupied and free space
    """
    res = resolution if resolution is not None else self.sample_resolution
    
    # Use the entire grid at once
    return quadrant_memory.query_grid(res)
```

### 7. Create Direct Entropy Calculation

```python
def calculate_entropy_batch(self,
                           occupied_probs: torch.Tensor,
                           empty_probs: torch.Tensor,
                           entropy_extractor) -> Dict[str, torch.Tensor]:
    """
    Calculate entropy for the entire grid in one batch operation.
    
    Args:
        occupied_probs: Grid of occupancy probabilities
        empty_probs: Grid of emptiness probabilities
        entropy_extractor: The entropy extraction object
        
    Returns:
        Dictionary with entropy maps and classifications
    """
    # Process the entire grid in one batch
    return entropy_extractor.extract_features(occupied_probs, empty_probs)
```

### 8. Eliminate All Loops in Vector Processing

Replace all loops in the processing pipeline with vectorized operations:

```python
def vectorized_processing_pipeline(self,
                                  points: torch.Tensor,
                                  labels: torch.Tensor,
                                  quadrant_memory,
                                  entropy_extractor) -> Dict[str, torch.Tensor]:
    """
    Complete processing pipeline using batch tensor operations.
    
    Args:
        points: Point cloud tensor
        labels: Labels tensor
        quadrant_memory: Quadrant memory object
        entropy_extractor: Entropy extractor object
        
    Returns:
        Dictionary with all processing results
    """
    # Update memory with all points at once
    quadrant_memory.update_with_points(points, labels)
    
    # Calculate probabilities for entire grid
    prob_grids = self.calculate_grid_probabilities(quadrant_memory)
    
    # Calculate entropy in a single batch operation
    entropy_results = self.calculate_entropy_batch(
        prob_grids['occupied'],
        prob_grids['empty'],
        entropy_extractor
    )
    
    # Return all results
    return {
        'prob_grids': prob_grids,
        'entropy_results': entropy_results,
        'grid_coords': {
            'x': self.x_coords,
            'y': self.y_coords
        }
    }
```

### 9. Implement Fast Point-Cloud Processing Operators

```python
def create_point_cloud_operators(self) -> Dict[str, Callable]:
    """
    Create a set of fast tensor operators for point cloud processing.
    
    Returns:
        Dictionary with common point cloud operations
    """
    def filter_by_distance(points, center, radius):
        """Filter points within distance of center (batch operation)"""
        distances = torch.norm(points - center.unsqueeze(0), dim=1)
        return points[distances <= radius]
    
    def batch_find_nearest(points, queries, k=1):
        """Find k nearest neighbors for all query points (batch operation)"""
        # Calculate all pairwise distances
        dists = torch.cdist(queries, points)
        # Get k nearest for each query
        vals, indices = torch.topk(dists, k, dim=1, largest=False)
        return indices, vals
    
    return {
        'filter_by_distance': filter_by_distance,
        'batch_find_nearest': batch_find_nearest
    }
```

### 10. Update VSAMapper Integration

```python
# In VSAMapper class
def _init_components(self) -> None:
    """Initialize system components with batch-oriented implementations"""
    # Replace SequentialProcessor with BatchProcessor
    self.processor = BatchProcessor(
        world_bounds=self.world_bounds,
        sample_resolution=self.config.get("sequential", "sample_resolution"),
        device=self.device
    )
    
    # Other components...
    
def process_point_cloud(self, 
                       points: Union[np.ndarray, torch.Tensor],
                       labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    """
    Process a full point cloud using batch operations.
    
    Args:
        points: Point cloud as array/tensor of shape (N, 2)
        labels: Optional labels for points (1=occupied, 0=empty)
    """
    # Convert inputs to tensors
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float().to(self.device)
        
    if labels is None:
        # Default to all occupied
        labels = torch.ones(points.shape[0], dtype=torch.int, device=self.device)
    elif isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).to(self.device)
    
    # Create memory update function
    update_fn = self.processor.create_update_function(self.quadrant_memory)
    
    # Process entire point cloud in one batch
    self.processor.process_point_cloud(points, labels, update_fn)
    
    # Normalize only at the end of processing
    self.quadrant_memory.normalize_memories()
    
    # Generate maps using vectorized operations
    results = self.processor.vectorized_processing_pipeline(
        points, labels, self.quadrant_memory, self.entropy_extractor
    )
    
    # Store results
    self.entropy_grid = results['entropy_results']['global_entropy']
    self.classification = results['entropy_results']['classification']
    self.occupancy_grid = self.entropy_extractor.get_occupancy_grid(self.classification)
    self.grid_coords = results['grid_coords']
```

This plan completely rewrites the sequential processing architecture to use batch tensor operations throughout. All loops are eliminated where possible, and operations are vectorized to take advantage of GPU acceleration. There are no compatibility layers, and the focus is entirely on performance optimization using modern PyTorch tensor operations.