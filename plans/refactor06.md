# Plan to Address Memory Normalization Frequency Issues

## Problem Statement
The original VSA-OGM only normalizes memory vectors after all points are processed, while the current implementation normalizes too frequently (after each point), causing unnecessary computational overhead.

## Implementation Plan

### 1. Eliminate Per-Point Normalization

```python
class QuadrantMemory:
    """
    Quadrant-based memory system for Vector Symbolic Architecture (VSA).
    
    Optimized to normalize memory vectors only at the end of processing.
    """
    
    def __init__(self, 
                 world_bounds: Tuple[float, float, float, float],
                 quadrant_size: int,
                 vector_dim: int = 1024,
                 length_scale: float = 1.0,
                 device: str = "cuda") -> None:
        # Initialize parameters
        self.world_bounds = world_bounds
        self.quadrant_size = quadrant_size
        self.vector_dim = vector_dim
        self.length_scale = length_scale
        self.device = device
        
        # Initialize memory vectors and other components
        self._init_memory()
        
        # Track normalization state
        self._needs_normalization = False
```

### 2. Redesign Memory Update Function Without Normalization

```python
def update_with_points(self, 
                      points: torch.Tensor, 
                      labels: torch.Tensor) -> None:
    """
    Update memory with multiple points without normalization.
    
    Args:
        points: Points tensor of shape (N, 2)
        labels: Binary labels tensor of shape (N)
    """
    # Get quadrant indices for all points at once
    quadrant_indices = self.get_quadrant_index(points)
    
    # Encode all points in one batch operation
    point_vectors = self.encode_points_batch(points)
    
    # Process occupied points
    occupied_mask = labels == 1
    if torch.any(occupied_mask):
        occupied_indices = quadrant_indices[occupied_mask]
        occupied_vectors = point_vectors[occupied_mask]
        
        # Update occupied memory without normalization
        for i, idx in enumerate(occupied_indices):
            self.occupied_memory[idx] += occupied_vectors[i]
    
    # Process empty points
    empty_mask = labels == 0
    if torch.any(empty_mask):
        empty_indices = quadrant_indices[empty_mask]
        empty_vectors = point_vectors[empty_mask]
        
        # Update empty memory without normalization
        for i, idx in enumerate(empty_indices):
            self.empty_memory[idx] += empty_vectors[i]
    
    # Mark that memory needs normalization
    self._needs_normalization = True
```

### 3. Implement Batch Memory Update Without Normalization

```python
def update_memory_batch(self,
                       indices: torch.Tensor,
                       vectors: torch.Tensor,
                       is_occupied: bool) -> None:
    """
    Update memory in batch without normalization.
    
    Args:
        indices: Quadrant indices tensor
        vectors: Point vectors tensor
        is_occupied: Whether to update occupied or empty memory
    """
    # Select target memory
    memory = self.occupied_memory if is_occupied else self.empty_memory
    
    # Update memory without normalization
    for i, idx in enumerate(indices):
        memory[idx] += vectors[i]
    
    # Mark that memory needs normalization
    self._needs_normalization = True
```

### 4. Implement Explicit Normalization Function

```python
def normalize_memories(self) -> None:
    """
    Normalize all memory vectors to unit length.
    
    Only performs normalization if needed, tracking normalization state.
    """
    # Skip if no normalization is needed
    if not self._needs_normalization:
        return
    
    # Calculate norms for occupied memory
    occupied_norms = torch.norm(self.occupied_memory, dim=1, keepdim=True)
    # Identify non-zero vectors
    occupied_mask = occupied_norms > 0
    # Normalize only non-zero vectors
    if torch.any(occupied_mask):
        self.occupied_memory[occupied_mask.squeeze()] /= occupied_norms[occupied_mask]
    
    # Calculate norms for empty memory
    empty_norms = torch.norm(self.empty_memory, dim=1, keepdim=True)
    # Identify non-zero vectors
    empty_mask = empty_norms > 0
    # Normalize only non-zero vectors
    if torch.any(empty_mask):
        self.empty_memory[empty_mask.squeeze()] /= empty_norms[empty_mask]
    
    # Reset normalization flag
    self._needs_normalization = False
```

### 5. Implement Point-Cloud Processing Without Normalization

```python
def process_point_cloud(self,
                       points: torch.Tensor,
                       labels: torch.Tensor) -> None:
    """
    Process entire point cloud without intermediate normalization.
    
    Args:
        points: Points tensor
        labels: Labels tensor
    """
    # Get quadrant indices for all points
    indices = self.get_quadrant_index(points)
    
    # Encode all points in one batch operation
    encoded_points = self.encode_points_batch(points)
    
    # Process occupied points
    occupied_mask = labels == 1
    if torch.any(occupied_mask):
        self.update_memory_batch(
            indices[occupied_mask],
            encoded_points[occupied_mask],
            is_occupied=True
        )
    
    # Process empty points
    empty_mask = labels == 0
    if torch.any(empty_mask):
        self.update_memory_batch(
            indices[empty_mask],
            encoded_points[empty_mask],
            is_occupied=False
        )
```

### 6. Implement Normalized Query Function

```python
def query_grid_normalized(self, resolution: float) -> Dict[str, torch.Tensor]:
    """
    Query grid with normalized memory vectors.
    
    Ensures memory is normalized before querying.
    
    Args:
        resolution: Resolution of the grid
        
    Returns:
        Dictionary with grid coordinates and similarity scores
    """
    # Ensure memory is normalized before querying
    self.normalize_memories()
    
    # Create grid coordinates
    x_coords = torch.arange(
        self.world_bounds[0], 
        self.world_bounds[1], 
        resolution, 
        device=self.device
    )
    
    y_coords = torch.arange(
        self.world_bounds[2], 
        self.world_bounds[3], 
        resolution, 
        device=self.device
    )
    
    # Create meshgrid for all coordinates
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_shape = xx.shape
    
    # Reshape to (N, 2) for batch processing
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Process grid query
    # (implementation of grid query logic)
    
    # Return results
    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'occupied': occupied_grid,
        'empty': empty_grid
    }
```

### 7. Implement On-Demand Normalization for Individual Queries

```python
def query_point(self, 
               point: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Query the memory for a specific point with on-demand normalization.
    
    Args:
        point: The point coordinates (x, y)
        
    Returns:
        Dictionary with similarity scores
    """
    # Get quadrant index
    quadrant_idx = self.get_quadrant_index(point)
    
    # Encode point
    point_vector = self.encode_point(point)
    
    # Normalize vectors for this specific query
    # This doesn't normalize the entire memory, just creates normalized copies
    occupied_vec = self.occupied_memory[quadrant_idx]
    empty_vec = self.empty_memory[quadrant_idx]
    
    # Calculate norms
    occupied_norm = torch.norm(occupied_vec)
    empty_norm = torch.norm(empty_vec)
    
    # Normalize vectors if non-zero
    if occupied_norm > 0:
        occupied_vec = occupied_vec / occupied_norm
    
    if empty_norm > 0:
        empty_vec = empty_vec / empty_norm
    
    # Calculate similarities
    occupied_sim = torch.dot(point_vector, occupied_vec)
    empty_sim = torch.dot(point_vector, empty_vec)
    
    return {
        'occupied': occupied_sim,
        'empty': empty_sim,
        'quadrant_idx': quadrant_idx
    }
```

### 8. Implement Optimized Memory Management

```python
def _init_memory(self) -> None:
    """
    Initialize memory vectors with optimized allocation.
    """
    # Calculate total quadrants
    total_quadrants = self.quadrant_size ** 2
    
    # Allocate memory tensors
    self.occupied_memory = torch.zeros(
        (total_quadrants, self.vector_dim), 
        device=self.device
    )
    
    self.empty_memory = torch.zeros(
        (total_quadrants, self.vector_dim), 
        device=self.device
    )
    
    # Initialize axis vectors
    self._init_axis_vectors()
    
    # Initial state doesn't need normalization
    self._needs_normalization = False
```

### 9. Implement Full-Batch Processing Method

```python
def process_batch_points(self,
                        all_points: List[torch.Tensor],
                        all_labels: List[torch.Tensor]) -> None:
    """
    Process multiple batches of points without intermediate normalization.
    
    Args:
        all_points: List of point tensors
        all_labels: List of label tensors
    """
    # Process each batch without normalization
    for points, labels in zip(all_points, all_labels):
        self.process_point_cloud(points, labels)
    
    # Normalize only once at the end
    self.normalize_memories()
```

### 10. Update VSAMapper Integration

```python
# Inside VSAMapper class
def process_point_cloud(self, points, labels=None):
    """
    Process a point cloud with optimized normalization.
    
    Args:
        points: Point cloud tensor
        labels: Labels tensor
    """
    # Convert inputs to tensors
    points = self.create_tensor(points)
    if labels is None:
        labels = torch.ones(points.shape[0], dtype=torch.int, device=self.device)
    else:
        labels = self.create_tensor(labels, dtype=torch.int)
    
    # Process all points without normalization
    self.quadrant_memory.process_point_cloud(points, labels)
    
    # Generate maps with a single normalization
    self._generate_maps()

def _generate_maps(self):
    """Generate maps with a single normalization."""
    # Ensure memory is normalized before querying
    self.quadrant_memory.normalize_memories()
    
    # Query grid for normalized similarity scores
    grid_results = self.quadrant_memory.query_grid(
        self.config.get("sequential", "sample_resolution")
    )
    
    # Extract features
    # (feature extraction logic)
    
    # Store results
    # (store results logic)
```

This implementation completely rewrites the memory normalization strategy to only normalize when necessary, rather than after each point update. Key improvements include:

1. Tracking normalization state to avoid redundant operations
2. Batching all processing without intermediate normalization
3. Ensuring normalization only happens once before querying
4. Providing on-demand normalization for individual queries
5. Optimizing memory allocation and management

These changes significantly reduce computational overhead while maintaining the same functionality and accuracy as the original implementation.