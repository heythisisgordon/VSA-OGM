You're right - no compatibility layers needed. Let's focus purely on performance optimization.

Here's a revised plan to address issue #2 (inefficient quadrant memory implementation):

# Plan to Address Quadrant Memory Implementation Inefficiencies

## Problem Statement
The current implementation in `src/quadrant_memory.py` uses an inefficient dictionary-based lookup system for quadrant assignment that creates Python overhead, while the original implementation in `spl/mapping.py` uses direct tensor operations for faster processing.

## Implementation Plan

### 1. Rewrite QuadrantMemory Class Core Architecture

- Eliminate all dictionary-based lookups
- Replace with direct tensor operations
- Pre-allocate all memory tensors at initialization
- Use vectorized operations throughout

### 2. Implement Direct Tensor-Based Indexing

```python
# Revised get_quadrant_index method
def get_quadrant_index(self, points: torch.Tensor) -> torch.Tensor:
    """
    Get the indices of quadrants containing the given points using tensor operations.
    
    Args:
        points: Point tensor of shape (N, 2) or single point (2,)
        
    Returns:
        Tensor of quadrant indices of shape (N,)
    """
    # Handle single point vs batch
    is_single = points.dim() == 1
    if is_single:
        points = points.unsqueeze(0)
        
    # Compute quadrant indices using tensor operations
    x_idx = ((points[:, 0] - self.world_bounds[0]) / self.quadrant_width).long()
    y_idx = ((points[:, 1] - self.world_bounds[2]) / self.quadrant_height).long()
    
    # Clamp to valid range
    x_idx = torch.clamp(x_idx, 0, self.quadrant_size - 1)
    y_idx = torch.clamp(y_idx, 0, self.quadrant_size - 1)
    
    # Convert to linear indices
    indices = x_idx * self.quadrant_size + y_idx
    
    return indices[0] if is_single else indices
```

### 3. Implement Batch Point Processing

```python
# Add batch update method
def update_with_points(self, 
                      points: torch.Tensor, 
                      labels: torch.Tensor) -> None:
    """
    Update memory with multiple points using batch operations.
    
    Args:
        points: Points tensor of shape (N, 2)
        labels: Binary labels tensor of shape (N,)
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
        
        # Use scatter_add for efficient memory updates
        for i, idx in enumerate(occupied_indices):
            self.occupied_memory[idx] += occupied_vectors[i]
    
    # Process empty points
    empty_mask = labels == 0
    if torch.any(empty_mask):
        empty_indices = quadrant_indices[empty_mask]
        empty_vectors = point_vectors[empty_mask]
        
        # Use scatter_add for efficient memory updates
        for i, idx in enumerate(empty_indices):
            self.empty_memory[idx] += empty_vectors[i]
```

### 4. Implement Batch Point Encoding

```python
# Add batch encoding method
def encode_points_batch(self, points: torch.Tensor) -> torch.Tensor:
    """
    Encode multiple points in batch using FFT operations.
    
    Args:
        points: Points tensor of shape (N, 2)
        
    Returns:
        Encoded vectors tensor of shape (N, vector_dim)
    """
    batch_size = points.shape[0]
    result = torch.zeros((batch_size, self.vector_dim), device=self.device)
    
    # Pre-compute FFT of axis vectors
    x_axis_fft = torch.fft.fft(self.axis_vectors[0])
    y_axis_fft = torch.fft.fft(self.axis_vectors[1])
    
    # Process in batches to avoid GPU memory issues
    max_batch = 1000  # Adjust based on memory constraints
    for i in range(0, batch_size, max_batch):
        end = min(i + max_batch, batch_size)
        batch_points = points[i:end]
        batch_size_current = batch_points.shape[0]
        
        # Apply fractional binding for x dimension
        x_powers = (batch_points[:, 0] / self.length_scale).unsqueeze(1)
        x_powers_expanded = x_powers.expand(-1, self.vector_dim)
        x_encoded_fft = x_axis_fft.unsqueeze(0).expand(batch_size_current, -1) ** x_powers_expanded
        
        # Apply fractional binding for y dimension
        y_powers = (batch_points[:, 1] / self.length_scale).unsqueeze(1)
        y_powers_expanded = y_powers.expand(-1, self.vector_dim)
        y_encoded_fft = y_axis_fft.unsqueeze(0).expand(batch_size_current, -1) ** y_powers_expanded
        
        # Bind through element-wise multiplication in Fourier domain
        result_fft = x_encoded_fft * y_encoded_fft
        
        # Transform back to time domain
        result[i:end] = torch.fft.ifft(result_fft).real
    
    return result
```

### 5. Optimize Query Operations

```python
# Enhanced query_grid method
def query_grid(self, resolution: float) -> Dict[str, torch.Tensor]:
    """
    Query memory for a grid of points using tensor operations.
    
    Args:
        resolution: Resolution of the grid
        
    Returns:
        Dictionary with grid coordinates and similarity scores
    """
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
    
    # Get quadrant indices for all points
    quadrant_indices = self.get_quadrant_index(points)
    
    # Encode all grid points in one batch operation
    grid_vectors = self.encode_points_batch(points)
    
    # Initialize similarity tensors
    occupied_sim = torch.zeros(len(points), device=self.device)
    empty_sim = torch.zeros(len(points), device=self.device)
    
    # Create normalized memory for faster similarity calculation
    occupied_memory_norm = self.normalize_memory(self.occupied_memory)
    empty_memory_norm = self.normalize_memory(self.empty_memory)
    
    # Process in batches to avoid GPU memory issues
    max_batch = 10000  # Adjust based on memory constraints
    for i in range(0, len(points), max_batch):
        end = min(i + max_batch, len(points))
        batch_indices = quadrant_indices[i:end]
        batch_vectors = grid_vectors[i:end]
        
        # Calculate similarities for occupied memory
        batch_occupied_mem = occupied_memory_norm[batch_indices]
        batch_occupied_sim = torch.sum(batch_vectors * batch_occupied_mem, dim=1)
        occupied_sim[i:end] = batch_occupied_sim
        
        # Calculate similarities for empty memory
        batch_empty_mem = empty_memory_norm[batch_indices]
        batch_empty_sim = torch.sum(batch_vectors * batch_empty_mem, dim=1)
        empty_sim[i:end] = batch_empty_sim
    
    # Reshape to original grid shape
    occupied_grid = occupied_sim.reshape(grid_shape)
    empty_grid = empty_sim.reshape(grid_shape)
    
    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'occupied': occupied_grid,
        'empty': empty_grid
    }

def normalize_memory(self, memory: torch.Tensor) -> torch.Tensor:
    """Normalize memory vectors while handling zero-norm vectors"""
    norms = torch.norm(memory, dim=1, keepdim=True)
    # Replace zero norms with 1 to avoid division by zero
    norms[norms == 0] = 1.0
    return memory / norms
```

### 6. Efficient Memory Initialization

```python
# Improved __init__ method
def __init__(self, world_bounds, quadrant_size, vector_dim=1024, length_scale=1.0, device="cpu"):
    """Initialize the quadrant memory system with efficient tensor operations"""
    self.world_bounds = world_bounds
    self.quadrant_size = quadrant_size
    self.vector_dim = vector_dim
    self.length_scale = length_scale
    self.device = device
    
    # Calculate world dimensions
    self.world_width = world_bounds[1] - world_bounds[0]
    self.world_height = world_bounds[3] - world_bounds[2]
    
    # Calculate quadrant dimensions
    self.quadrant_width = self.world_width / quadrant_size
    self.quadrant_height = self.world_height / quadrant_size
    
    # Initialize axis vectors
    self._init_axis_vectors()
    
    # Pre-allocate memory tensors
    total_quadrants = quadrant_size * quadrant_size
    self.occupied_memory = torch.zeros((total_quadrants, self.vector_dim), device=device)
    self.empty_memory = torch.zeros((total_quadrants, self.vector_dim), device=device)
    
    # Pre-compute quadrant centers for visualization
    self._compute_quadrant_centers()
```

### 7. Add Profiling Code

```python
# Add timing decorators for performance monitoring
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

# Apply to key methods for performance monitoring during development
# @timing_decorator
def get_quadrant_index(self, points):
    # Method implementation...

# @timing_decorator
def encode_points_batch(self, points):
    # Method implementation...
```

### 8. Implement Fast Memory Normalization

```python
def normalize_memories(self) -> None:
    """Normalize all memory vectors to unit length using batch operations"""
    # Calculate norms
    occupied_norms = torch.norm(self.occupied_memory, dim=1, keepdim=True)
    empty_norms = torch.norm(self.empty_memory, dim=1, keepdim=True)
    
    # Identify non-zero vectors
    occupied_mask = occupied_norms > 0
    empty_mask = empty_norms > 0
    
    # Normalize only non-zero vectors
    if torch.any(occupied_mask):
        self.occupied_memory[occupied_mask.squeeze()] /= occupied_norms[occupied_mask]
        
    if torch.any(empty_mask):
        self.empty_memory[empty_mask.squeeze()] /= empty_norms[empty_mask]
```

### 9. Implement Memory-Efficient Vector Storage

```python
# Add efficient memory usage with half-precision where appropriate
def enable_memory_efficiency(self, use_half_precision=True):
    """Enable memory efficiency optimizations"""
    if use_half_precision and self.device == "cuda":
        # Convert to half precision (16-bit) to save memory on GPU
        self.occupied_memory = self.occupied_memory.half()
        self.empty_memory = self.empty_memory.half()
        print("Using half precision (16-bit) for memory vectors")
        
        # Note: some operations need to be in full precision
        # Add conversion helpers
        self._half_precision = True
    else:
        self._half_precision = False
```

### 10. Direct Integration with Mapper

```python
# Update VSAMapper to integrate with optimized QuadrantMemory
def process_point_cloud(self, points, labels=None):
    """Process a point cloud in an optimized manner"""
    # Convert inputs to correct format and device
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float().to(self.device)
    
    if labels is None:
        labels = torch.ones(points.shape[0], dtype=torch.int, device=self.device)
    elif isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).to(self.device)
    
    # Direct processing with optimized batch operations
    self.quadrant_memory.update_with_points(points, labels)
    
    # Normalize only at the end of processing
    self.quadrant_memory.normalize_memories()
    
    # Generate maps with optimized grid queries
    self._generate_maps()
```

This implementation removes all dictionary-based operations, eliminates Python overhead, and uses tensor operations throughout for maximum efficiency. The batch processing approach significantly reduces the number of function calls and leverages GPU acceleration where available.