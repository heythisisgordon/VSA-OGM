# Plan to Address Inefficient Point Encoding

## Problem Statement
The original code uses Fourier-domain operations for encoding points, which is faster than time-domain implementations in the current code. The point encoding process needs a complete rewrite for better performance.

## Implementation Plan

### 1. Redesign Point Encoding Architecture

```python
class PointEncoder:
    """
    Optimized Fourier-domain point encoder for Vector Symbolic Architecture.
    Provides fast encoding of spatial coordinates into hyperdimensional vectors.
    """
    
    def __init__(self, 
                 vector_dim: int,
                 length_scale: float = 1.0,
                 device: str = "cuda") -> None:
        """
        Initialize the point encoder with optimized parameters.
        
        Args:
            vector_dim: Dimensionality of VSA vectors
            length_scale: Length scale parameter for fractional binding
            device: Device for tensor operations
        """
        self.vector_dim = vector_dim
        self.length_scale = length_scale
        self.device = device
        
        # Initialize axis vectors
        self._init_axis_vectors()
        
        # Pre-compute FFT of axis vectors for fast encoding
        self._precompute_fft()
```

### 2. Implement Optimized Axis Vector Initialization

```python
def _init_axis_vectors(self) -> None:
    """
    Initialize axis vectors with optimized generation.
    """
    # Generate orthogonal unitary vectors for x and y axes
    self.axis_vectors = torch.zeros((2, self.vector_dim), device=self.device)
    
    # Create first axis vector
    a = torch.rand((self.vector_dim - 1) // 2, device=self.device)
    sign = torch.randint(0, 2, ((self.vector_dim - 1) // 2,), device=self.device) * 2 - 1
    phi = sign * torch.pi * (0.001 + a * 0.998)
    
    # Construct frequency domain representation
    fv = torch.zeros(self.vector_dim, dtype=torch.complex64, device=self.device)
    fv[0] = 1
    fv[1:(self.vector_dim + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(self.vector_dim // 2) + 1:] = torch.flip(torch.conj(fv[1:(self.vector_dim + 1) // 2]), dims=[0])
    
    # Handle even-dimensional case
    if self.vector_dim % 2 == 0:
        fv[self.vector_dim // 2] = 1
    
    # Convert to time domain
    self.axis_vectors[0] = torch.fft.ifft(fv).real
    
    # Create second orthogonal axis vector (independent random phases)
    a = torch.rand((self.vector_dim - 1) // 2, device=self.device)
    sign = torch.randint(0, 2, ((self.vector_dim - 1) // 2,), device=self.device) * 2 - 1
    phi = sign * torch.pi * (0.001 + a * 0.998)
    
    # Construct frequency domain representation
    fv = torch.zeros(self.vector_dim, dtype=torch.complex64, device=self.device)
    fv[0] = 1
    fv[1:(self.vector_dim + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(self.vector_dim // 2) + 1:] = torch.flip(torch.conj(fv[1:(self.vector_dim + 1) // 2]), dims=[0])
    
    # Handle even-dimensional case
    if self.vector_dim % 2 == 0:
        fv[self.vector_dim // 2] = 1
    
    # Convert to time domain
    self.axis_vectors[1] = torch.fft.ifft(fv).real
```

### 3. Implement FFT Precomputation

```python
def _precompute_fft(self) -> None:
    """
    Precompute FFT of axis vectors for fast encoding.
    """
    # Transform axis vectors to frequency domain
    self.axis_vectors_fft = torch.fft.fft(self.axis_vectors, dim=1)
```

### 4. Implement Direct Fourier-Domain Point Encoding

```python
def encode_point(self, point: torch.Tensor) -> torch.Tensor:
    """
    Encode a single point into hyperdimensional space using Fourier domain operations.
    
    Args:
        point: Point coordinates [x, y]
        
    Returns:
        Encoded vector representation of the point
    """
    # Scale coordinates by length scale
    scaled_point = point / self.length_scale
    
    # Apply fractional binding in frequency domain
    x_encoded_fft = self.axis_vectors_fft[0] ** scaled_point[0]
    y_encoded_fft = self.axis_vectors_fft[1] ** scaled_point[1]
    
    # Bind through element-wise multiplication in frequency domain
    bound_fft = x_encoded_fft * y_encoded_fft
    
    # Convert back to time domain
    return torch.fft.ifft(bound_fft).real
```

### 5. Implement Optimized Batch Point Encoding

```python
def encode_points_batch(self, points: torch.Tensor) -> torch.Tensor:
    """
    Encode multiple points at once using vectorized Fourier domain operations.
    
    Args:
        points: Point coordinates tensor of shape [N, 2]
        
    Returns:
        Encoded vector representations of shape [N, vector_dim]
    """
    batch_size = points.shape[0]
    
    # Scale all points by length scale
    scaled_points = points / self.length_scale
    
    # Prepare for broadcasting
    x_fft = self.axis_vectors_fft[0].unsqueeze(0).expand(batch_size, -1)
    y_fft = self.axis_vectors_fft[1].unsqueeze(0).expand(batch_size, -1)
    
    x_powers = scaled_points[:, 0].unsqueeze(1).expand(-1, self.vector_dim)
    y_powers = scaled_points[:, 1].unsqueeze(1).expand(-1, self.vector_dim)
    
    # Apply fractional binding in frequency domain
    x_encoded_fft = x_fft ** x_powers
    y_encoded_fft = y_fft ** y_powers
    
    # Bind through element-wise multiplication in frequency domain
    bound_fft = x_encoded_fft * y_encoded_fft
    
    # Convert back to time domain
    return torch.fft.ifft(bound_fft).real
```

### 6. Implement Grid Encoding for Fast Querying

```python
def encode_grid(self, 
               x_coords: torch.Tensor, 
               y_coords: torch.Tensor) -> torch.Tensor:
    """
    Encode a grid of points for fast querying.
    
    Args:
        x_coords: X-coordinates tensor
        y_coords: Y-coordinates tensor
        
    Returns:
        Grid of encoded vectors of shape [len(x_coords), len(y_coords), vector_dim]
    """
    # Create meshgrid
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Reshape to [N, 2] for batch processing
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Encode all points
    encoded = self.encode_points_batch(points)
    
    # Reshape to grid
    return encoded.reshape(len(x_coords), len(y_coords), self.vector_dim)
```

### 7. Implement Vector Similarity Calculation

```python
def calculate_similarity(self, 
                        encoded_points: torch.Tensor, 
                        memory_vector: torch.Tensor) -> torch.Tensor:
    """
    Calculate similarity between encoded points and a memory vector.
    
    Args:
        encoded_points: Encoded points tensor of shape [..., vector_dim]
        memory_vector: Memory vector of shape [vector_dim]
        
    Returns:
        Similarity scores tensor of shape [...]
    """
    # Normalize memory vector if needed
    memory_norm = torch.norm(memory_vector)
    if memory_norm > 0:
        memory_vector = memory_vector / memory_norm
    
    # Calculate dot product along last dimension
    return torch.sum(encoded_points * memory_vector, dim=-1)
```

### 8. Implement Optimized Binding Operation

```python
def bind(self, vectors: torch.Tensor) -> torch.Tensor:
    """
    Bind multiple vectors together using Fourier domain operations.
    
    Args:
        vectors: Tensor of shape [num_vectors, vector_dim]
        
    Returns:
        Bound vector of shape [vector_dim]
    """
    # Transform to frequency domain
    vectors_fft = torch.fft.fft(vectors, dim=1)
    
    # Multiply in frequency domain
    result_fft = torch.prod(vectors_fft, dim=0)
    
    # Transform back to time domain
    return torch.fft.ifft(result_fft).real
```

### 9. Implement Direct Integration with QuadrantMemory

```python
# In QuadrantMemory class
def __init__(self, world_bounds, quadrant_size, vector_dim, length_scale, device):
    # Other initialization...
    
    # Initialize point encoder with optimized parameters
    self.point_encoder = PointEncoder(
        vector_dim=vector_dim,
        length_scale=length_scale,
        device=device
    )

def encode_point(self, point):
    """Encode a single point using optimized encoder"""
    return self.point_encoder.encode_point(point)

def encode_points_batch(self, points):
    """Encode multiple points using optimized batch encoder"""
    return self.point_encoder.encode_points_batch(points)
```

### 10. Optimize Query Grid with Precomputed Encodings

```python
def query_grid(self, resolution: float) -> Dict[str, torch.Tensor]:
    """
    Query memory for a grid of points using optimized encoding.
    
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
    
    # Encode entire grid at once
    grid_encoded = self.point_encoder.encode_grid(x_coords, y_coords)
    
    # Get quadrant indices for all grid points
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    quadrant_indices = self.get_quadrant_index(points)
    
    # Normalize memories if needed
    self.normalize_memories()
    
    # Initialize result tensors
    occupied_grid = torch.zeros((len(x_coords), len(y_coords)), device=self.device)
    empty_grid = torch.zeros((len(x_coords), len(y_coords)), device=self.device)
    
    # Reshape grid encoded to match quadrant indices
    grid_encoded_flat = grid_encoded.reshape(-1, self.vector_dim)
    
    # Calculate similarities for each quadrant
    for q_idx in range(len(self.occupied_memory)):
        # Find points in this quadrant
        q_mask = quadrant_indices == q_idx
        
        if not torch.any(q_mask):
            continue
        
        # Get points in this quadrant
        q_points = grid_encoded_flat[q_mask]
        
        # Calculate similarities
        if torch.norm(self.occupied_memory[q_idx]) > 0:
            q_occupied_sim = self.point_encoder.calculate_similarity(
                q_points, self.occupied_memory[q_idx]
            )
        else:
            q_occupied_sim = torch.zeros(q_points.shape[0], device=self.device)
            
        if torch.norm(self.empty_memory[q_idx]) > 0:
            q_empty_sim = self.point_encoder.calculate_similarity(
                q_points, self.empty_memory[q_idx]
            )
        else:
            q_empty_sim = torch.zeros(q_points.shape[0], device=self.device)
        
        # Map flat indices back to grid
        flat_indices = torch.where(q_mask)[0]
        grid_indices = np.unravel_index(
            flat_indices.cpu().numpy(), 
            (len(x_coords), len(y_coords))
        )
        
        # Update grids
        occupied_grid[grid_indices] = q_occupied_sim.cpu().numpy()
        empty_grid[grid_indices] = q_empty_sim.cpu().numpy()
    
    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'occupied': occupied_grid,
        'empty': empty_grid
    }
```

This implementation completely rewrites the point encoding system to use Fourier-domain operations consistently. Key optimizations include:

1. Precomputing the FFT of axis vectors to avoid redundant transformations
2. Implementing direct Fourier-domain fractional binding
3. Providing batch operations for encoding multiple points at once
4. Optimizing grid encoding for query operations
5. Using vectorized similarity calculations
6. Implementing optimized binding operations

These changes significantly improve encoding performance by eliminating redundant calculations and leveraging the efficiency of frequency-domain operations throughout the encoding process.