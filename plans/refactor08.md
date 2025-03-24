# Plan to Address Python vs. C++ Bottlenecks

## Problem Statement
The current implementation has critical sections implemented in pure Python rather than using optimized libraries like NumPy/PyTorch, creating performance bottlenecks.

## Implementation Plan

### 1. Rewrite Core VSA Operations Using PyTorch Native Functions

```python
def make_unitary(vector_dim: int, device: str = "cuda") -> torch.Tensor:
    """
    Create a unitary vector using PyTorch native operations.
    
    Args:
        vector_dim: Dimensionality of the vector
        device: Device for tensor operations
        
    Returns:
        Unitary vector of shape [vector_dim]
    """
    # Generate frequency-domain representation directly
    fv = torch.zeros(vector_dim, dtype=torch.complex64, device=device)
    
    # Use PyTorch random functions for improved performance
    phi = torch.rand((vector_dim - 1) // 2, device=device) * torch.pi * 0.99 + 0.01 * torch.pi
    phi = phi * torch.randint(0, 2, ((vector_dim - 1) // 2,), device=device).float() * 2 - 1
    
    # Set frequency-domain components in a single operation
    fv[0] = 1.0
    fv[1:(vector_dim + 1) // 2] = torch.exp(1j * phi)
    fv[(vector_dim // 2) + 1:] = torch.flip(torch.conj(fv[1:(vector_dim + 1) // 2]), dims=[0])
    
    # Handle even-dimensional case
    if vector_dim % 2 == 0:
        fv[vector_dim // 2] = 1.0
    
    # Convert to time domain using torch.fft
    return torch.fft.ifft(fv).real
```

### 2. Implement Direct Tensor Operations for Binding

```python
def bind_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """
    Bind multiple vectors using PyTorch's native FFT operations.
    
    Args:
        vectors: Tensor of shape [num_vectors, vector_dim]
        
    Returns:
        Bound vector of shape [vector_dim]
    """
    # Transform to frequency domain
    vectors_fft = torch.fft.fft(vectors, dim=1)
    
    # Multiply in frequency domain using optimized reduction
    result_fft = vectors_fft.prod(dim=0)
    
    # Transform back to time domain
    return torch.fft.ifft(result_fft).real
```

### 3. Implement Matrix-Based Fractional Binding

```python
def fractional_bind(vector: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
    """
    Apply fractional binding to a vector with multiple powers in a single operation.
    
    Args:
        vector: Base vector of shape [vector_dim]
        powers: Powers tensor of shape [batch_size]
        
    Returns:
        Tensor of shape [batch_size, vector_dim]
    """
    # Get vector in frequency domain
    vector_fft = torch.fft.fft(vector)
    
    # Expand for broadcasting
    vector_fft = vector_fft.unsqueeze(0).expand(powers.shape[0], -1)
    powers = powers.unsqueeze(1).expand(-1, vector.shape[0])
    
    # Apply fractional binding using native exponentiation
    result_fft = vector_fft ** powers
    
    # Transform back to time domain
    return torch.fft.ifft(result_fft).real
```

### 4. Implement Direct GPU-Optimized Kernel Operations

```python
def create_optimized_filter(radius: int, device: str = "cuda") -> torch.Tensor:
    """
    Create an optimized disk filter with PyTorch's native operations.
    
    Args:
        radius: Radius of the disk filter
        device: Device for tensor operations
        
    Returns:
        Normalized disk filter
    """
    # Create coordinate grid directly
    size = 2 * radius + 1
    indices = torch.arange(-radius, radius + 1, device=device)
    y, x = torch.meshgrid(indices, indices, indexing='ij')
    
    # Create disk mask using efficient tensor operations
    distances = torch.sqrt(x.float() ** 2 + y.float() ** 2)
    disk = (distances <= radius).float()
    
    # Normalize in a single operation
    return disk / disk.sum()
```

### 5. Implement Optimized Entropy Calculation

```python
def calculate_entropy(probability_field: torch.Tensor, disk_filter: torch.Tensor) -> torch.Tensor:
    """
    Calculate Shannon entropy using PyTorch's optimized operations.
    
    Args:
        probability_field: Probability field tensor
        disk_filter: Disk filter for convolution
        
    Returns:
        Entropy tensor
    """
    # Clamp values to avoid log(0) using PyTorch's efficient operations
    prob = torch.clamp(probability_field, 1e-10, 1.0 - 1e-10)
    
    # Calculate entropy directly
    entropy = -prob * torch.log2(prob) - (1-prob) * torch.log2(1-prob)
    
    # Format tensors for convolution
    entropy_4d = entropy.unsqueeze(0).unsqueeze(0)
    filter_4d = disk_filter.unsqueeze(0).unsqueeze(0)
    
    # Use optimized conv2d function
    padding = disk_filter.shape[0] // 2
    local_entropy = torch.nn.functional.conv2d(entropy_4d, filter_4d, padding=padding)
    
    # Return to original shape
    return local_entropy.squeeze()
```

### 6. Implement Direct Tensor Manipulation for Memory Updates

```python
def update_memory_tensors(memory: torch.Tensor, 
                         indices: torch.Tensor, 
                         vectors: torch.Tensor) -> torch.Tensor:
    """
    Update memory tensors with vectors at specified indices using PyTorch's scatter operations.
    
    Args:
        memory: Memory tensor of shape [num_quadrants, vector_dim]
        indices: Indices tensor of shape [batch_size]
        vectors: Vectors tensor of shape [batch_size, vector_dim]
        
    Returns:
        Updated memory tensor
    """
    # Create index tensor for scatter_add
    index_tensor = indices.unsqueeze(1).expand(-1, vectors.shape[1])
    
    # Use scatter_add for efficient updates
    return memory.scatter_add(0, index_tensor, vectors)
```

### 7. Implement Contiguous Memory Layout and Strided Operations

```python
def ensure_optimal_memory_layout(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has optimal memory layout for computational efficiency.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor with optimal memory layout
    """
    # Check if tensor is contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    return tensor
```

### 8. Implement Native PyTorch Spatial Operations

```python
def spatial_distance(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise spatial distances using PyTorch's native operations.
    
    Args:
        points1: First set of points with shape [N, 2]
        points2: Second set of points with shape [M, 2]
        
    Returns:
        Distance matrix with shape [N, M]
    """
    # Use PyTorch's cdist for optimized distance calculation
    return torch.cdist(points1, points2)
```

### 9. Implement Optimized Quadrant Assignment

```python
def assign_points_to_quadrants(points: torch.Tensor, 
                              world_bounds: torch.Tensor,
                              quadrant_size: int) -> torch.Tensor:
    """
    Assign points to quadrants using PyTorch's native operations.
    
    Args:
        points: Points tensor with shape [N, 2]
        world_bounds: World bounds tensor [x_min, x_max, y_min, y_max]
        quadrant_size: Number of quadrants along each dimension
        
    Returns:
        Quadrant indices tensor with shape [N]
    """
    # Calculate quadrant sizes
    quadrant_width = (world_bounds[1] - world_bounds[0]) / quadrant_size
    quadrant_height = (world_bounds[3] - world_bounds[2]) / quadrant_size
    
    # Calculate quadrant indices in a single operation
    x_idx = ((points[:, 0] - world_bounds[0]) / quadrant_width).long()
    y_idx = ((points[:, 1] - world_bounds[2]) / quadrant_height).long()
    
    # Clamp indices to valid range
    x_idx = torch.clamp(x_idx, 0, quadrant_size - 1)
    y_idx = torch.clamp(y_idx, 0, quadrant_size - 1)
    
    # Calculate linear indices
    return x_idx * quadrant_size + y_idx
```

### 10. Implement Direct Integration in VSAMapper

```python
class VSAMapper:
    """
    Main VSA-OGM implementation with optimized PyTorch operations.
    """
    def __init__(self, config, world_bounds):
        # Initialize parameters
        self.config = config
        self.world_bounds = torch.tensor(world_bounds, device="cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components with optimized parameters
        self._init_components()
    
    def _init_components(self):
        """Initialize components with optimized PyTorch operations"""
        # Create optimized point encoder
        self.point_encoder = PointEncoder(
            vector_dim=self.config.get("vsa", "dimensions"),
            length_scale=self.config.get("vsa", "length_scale"),
            device=self.device
        )
        
        # Create optimized quadrant memory
        self.quadrant_memory = QuadrantMemory(
            world_bounds=self.world_bounds,
            quadrant_size=self.config.get("quadrant", "size"),
            vector_dim=self.config.get("vsa", "dimensions"),
            length_scale=self.config.get("vsa", "length_scale"),
            device=self.device
        )
        
        # Create optimized entropy extractor
        self.entropy_extractor = EntropyExtractor(
            disk_radius=self.config.get("entropy", "disk_radius"),
            occupied_threshold=self.config.get("entropy", "occupied_threshold"),
            empty_threshold=self.config.get("entropy", "empty_threshold"),
            device=self.device
        )
        
        # Create optimized batch processor
        self.processor = BatchProcessor(
            world_bounds=self.world_bounds,
            sample_resolution=self.config.get("sequential", "sample_resolution"),
            device=self.device
        )
    
    def process_point_cloud(self, points, labels=None):
        """Process point cloud with optimized PyTorch operations"""
        # Convert inputs to appropriate tensors
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype=torch.float32, device=self.device)
        else:
            points = points.to(device=self.device, dtype=torch.float32)
        
        if labels is None:
            labels = torch.ones(points.shape[0], dtype=torch.int32, device=self.device)
        elif isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        else:
            labels = labels.to(device=self.device, dtype=torch.int32)
        
        # Process point cloud
        self.processor.process_point_cloud(points, labels, self.quadrant_memory)
        
        # Generate maps
        self._generate_maps()
    
    def _generate_maps(self):
        """Generate maps with optimized PyTorch operations"""
        # Normalize memory vectors
        self.quadrant_memory.normalize_memories()
        
        # Query grid from quadrant memory
        grid_results = self.quadrant_memory.query_grid(
            self.config.get("sequential", "sample_resolution")
        )
        
        # Apply Born rule to convert similarity scores to probabilities
        occupied_probs = self.entropy_extractor.apply_born_rule(grid_results['occupied'])
        empty_probs = self.entropy_extractor.apply_born_rule(grid_results['empty'])
        
        # Extract features with optimized entropy calculation
        features = self.entropy_extractor.extract_features(occupied_probs, empty_probs)
        
        # Store results
        self.entropy_grid = features['global_entropy']
        self.classification = features['classification']
        self.occupancy_grid = self.entropy_extractor.get_occupancy_grid(self.classification)
        self.grid_coords = {
            'x': grid_results['x_coords'],
            'y': grid_results['y_coords']
        }
```

This implementation completely rewrites critical sections to use PyTorch's native operations instead of pure Python implementations. Key optimizations include:

1. Using PyTorch's native FFT operations for binding and fractional binding
2. Implementing direct tensor operations for memory updates
3. Using optimized convolution operations for entropy calculation
4. Implementing efficient spatial distance calculations
5. Using scatter operations for memory updates
6. Ensuring optimal memory layout for computational efficiency
7. Using vectorized operations throughout

These changes eliminate Python-level bottlenecks and leverage PyTorch's highly optimized C++ and CUDA implementations for core operations, resulting in significantly improved performance.