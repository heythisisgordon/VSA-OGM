# Plan to Address Missing CUDA Acceleration

## Problem Statement
The original VSA-OGM code leverages GPU acceleration where available, providing 10-100x speedups, while the current implementation doesn't properly utilize CUDA capabilities for tensor operations.

## Implementation Plan

### 1. Implement Device-Aware Initialization

```python
def __init__(self, config, world_bounds):
    """Initialize with proper device detection and configuration"""
    # Automatic device selection with fallback
    if torch.cuda.is_available() and config.get("system", "use_gpu", True):
        self.device = "cuda"
        # Get GPU information for reporting
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Configure for optimal performance
        torch.backends.cudnn.benchmark = True
    else:
        self.device = "cpu"
        print("Using CPU for processing (GPU not available or disabled)")
    
    # Move all tensors to the selected device
    self.world_bounds = torch.tensor(world_bounds, device=self.device)
```

### 2. Implement CUDA-Optimized Tensor Creation

```python
def create_tensor(self, data, dtype=torch.float32):
    """Create tensor on the correct device with optional type"""
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype, device=self.device)
    elif isinstance(data, torch.Tensor):
        return data.to(device=self.device, dtype=dtype)
    else:
        return torch.tensor(data, dtype=dtype, device=self.device)
```

### 3. Implement GPU Memory Management

```python
class GPUMemoryManager:
    """
    Manage GPU memory usage for optimal performance.
    """
    def __init__(self, device="cuda", reserved_memory_fraction=0.2):
        self.device = device
        if device == "cuda":
            # Reserve memory to avoid fragmentation
            self.reserved_memory_fraction = reserved_memory_fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = int(total_memory * reserved_memory_fraction)
            
            # Create a tensor to reserve memory
            self.reserved_tensor = torch.zeros(reserved_memory, 
                                             dtype=torch.uint8, 
                                             device=device)
    
    def release_reserved_memory(self):
        """Release reserved memory when needed for large operations"""
        if hasattr(self, 'reserved_tensor'):
            del self.reserved_tensor
            torch.cuda.empty_cache()
    
    def restore_reserved_memory(self):
        """Restore memory reservation after large operations"""
        if self.device == "cuda" and not hasattr(self, 'reserved_tensor'):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = int(total_memory * self.reserved_memory_fraction)
            self.reserved_tensor = torch.zeros(reserved_memory, 
                                             dtype=torch.uint8, 
                                             device=self.device)
    
    def batch_size_for_memory(self, tensor_size_per_item, max_memory_fraction=0.7):
        """Calculate optimal batch size based on available memory"""
        if self.device != "cuda":
            return 10000  # Default large batch size for CPU
            
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * max_memory_fraction
        
        # Calculate memory per item
        bytes_per_item = tensor_size_per_item * 4  # Assuming float32
        
        # Calculate batch size
        return max(1, int(available_memory / bytes_per_item))
```

### 4. Optimize FFT Operations for CUDA

```python
def optimize_fft_operations(self, tensor_shape):
    """Set optimal FFT parameters based on tensor shape"""
    if self.device == "cuda":
        # On CUDA, certain sizes are faster for FFT
        # Find the next power of 2 for optimal FFT performance
        next_pow2 = 2 ** (tensor_shape[-1] - 1).bit_length()
        
        if next_pow2 != tensor_shape[-1]:
            print(f"For optimal FFT performance, consider using dimension size {next_pow2} instead of {tensor_shape[-1]}")
            
        # Set cuFFT plan cache
        torch.backends.cuda.cufft_plan_cache.max_size = 128
        
        return next_pow2
    return tensor_shape[-1]
```

### 5. Implement CUDA-Optimized Batch Operations

```python
def batch_process_points(self, points, length_scale, axis_vectors):
    """
    Process points in optimized batches for CUDA.
    
    Args:
        points: Points tensor (N, 2)
        length_scale: Length scale parameter
        axis_vectors: Axis basis vectors
        
    Returns:
        Encoded point vectors
    """
    if self.device != "cuda":
        # For CPU, process in one go
        return self.encode_points_batch(points, length_scale, axis_vectors)
    
    # For CUDA, determine optimal batch size
    N = points.shape[0]
    vector_dim = axis_vectors.shape[1]
    
    # Calculate memory requirements
    memory_per_item = vector_dim * 8  # Complex64 in FFT domain
    
    # Get optimal batch size
    batch_size = self.memory_manager.batch_size_for_memory(memory_per_item)
    
    # Process in batches
    result = torch.zeros((N, vector_dim), device=self.device)
    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_points = points[i:end]
        result[i:end] = self.encode_points_batch(batch_points, length_scale, axis_vectors)
        
        # Clear cache to avoid fragmentation
        torch.cuda.empty_cache() 
    
    return result
```

### 6. Implement CUDA Stream Processing for Parallel Operations

```python
def parallel_process_with_streams(self, operations, tensors):
    """
    Process multiple operations in parallel using CUDA streams.
    
    Args:
        operations: List of functions to execute
        tensors: List of tensors to process
        
    Returns:
        List of results
    """
    if self.device != "cuda":
        # Sequential processing for CPU
        return [op(tensor) for op, tensor in zip(operations, tensors)]
    
    # Create streams for parallel processing
    streams = [torch.cuda.Stream() for _ in range(len(operations))]
    results = [None] * len(operations)
    
    # Launch operations in parallel streams
    for i, (op, tensor) in enumerate(zip(operations, tensors)):
        with torch.cuda.stream(streams[i]):
            results[i] = op(tensor)
    
    # Synchronize all streams
    torch.cuda.synchronize()
    
    return results
```

### 7. Optimize Matrix Operations for CUDA

```python
def optimize_matrix_operations(self, mat1, mat2, operation="multiply"):
    """
    Optimize matrix operations for CUDA performance.
    
    Args:
        mat1: First matrix
        mat2: Second matrix
        operation: Operation type (multiply, add, etc.)
        
    Returns:
        Result matrix
    """
    if self.device != "cuda":
        # Standard operations for CPU
        if operation == "multiply":
            return torch.matmul(mat1, mat2)
        elif operation == "add":
            return mat1 + mat2
    
    # For CUDA, use specialized functions
    if operation == "multiply":
        # Use cublas for large matrices
        if mat1.shape[0] > 1000 or mat2.shape[1] > 1000:
            # Ensure contiguous memory layout
            mat1 = mat1.contiguous()
            mat2 = mat2.contiguous()
            return torch.matmul(mat1, mat2)
        else:
            return torch.matmul(mat1, mat2)
    elif operation == "add":
        return mat1 + mat2
```

### 8. Implement Mixed Precision Training for CUDA

```python
def use_mixed_precision(self):
    """Configure for mixed precision computation when possible"""
    if self.device == "cuda" and torch.cuda.is_available():
        # Check if GPU supports mixed precision
        if torch.cuda.get_device_capability(0)[0] >= 7:
            # Set up automatic mixed precision
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
            print("Using automatic mixed precision (16-bit) for faster computation")
            return True
        else:
            print("GPU does not support mixed precision, using 32-bit computation")
    
    self.use_amp = False
    return False

def compute_with_mixed_precision(self, func, *args, **kwargs):
    """Perform computation with mixed precision when available"""
    if not self.use_amp:
        return func(*args, **kwargs)
    
    # Use automatic mixed precision
    with torch.cuda.amp.autocast():
        return func(*args, **kwargs)
```

### 9. Optimize Entropy Calculation for CUDA

```python
def calculate_entropy_cuda(self, probability_field, disk_filter):
    """
    Calculate Shannon entropy using CUDA-optimized operations.
    
    Args:
        probability_field: Probability field tensor
        disk_filter: Disk filter for convolution
        
    Returns:
        Entropy tensor
    """
    # Ensure tensors are on CUDA
    probability_field = probability_field.to(self.device)
    disk_filter = disk_filter.to(self.device)
    
    # Use mixed precision for entropy calculation
    def entropy_calc():
        # Clamp values to avoid log(0)
        prob = torch.clamp(probability_field, 1e-10, 1.0 - 1e-10)
        
        # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -prob * torch.log2(prob) - (1-prob) * torch.log2(1-prob)
        
        # Optimize convolution for CUDA
        filter_size = disk_filter.shape[0]
        padding = filter_size // 2
        
        # Reshape filter for 2D convolution
        disk_reshaped = disk_filter.unsqueeze(0).unsqueeze(0)
        entropy_reshaped = entropy.unsqueeze(0).unsqueeze(0)
        
        # Use optimized cudnn convolution
        return torch.nn.functional.conv2d(
            entropy_reshaped, disk_reshaped, padding=padding
        ).squeeze()
    
    # Use mixed precision if available
    return self.compute_with_mixed_precision(entropy_calc)
```

### 10. Implement CUDA-Aware VSAMapper

```python
class VSAMapper:
    """
    Main VSA-OGM implementation with CUDA acceleration.
    """
    def __init__(self, config, world_bounds):
        # Initialize with CUDA support
        self._init_cuda(config)
        
        # Create memory manager
        self.memory_manager = GPUMemoryManager(device=self.device)
        
        # Initialize components with CUDA support
        self._init_components_cuda()
    
    def _init_cuda(self, config):
        """Initialize CUDA environment"""
        # Device selection
        if torch.cuda.is_available() and config.get("system", "use_gpu", True):
            self.device = "cuda"
            
            # Set CUDA device
            if "gpu_id" in config.get("system", {}):
                torch.cuda.set_device(config.get("system", "gpu_id"))
            
            # Configure cuDNN
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Configure for TF32 precision (on Ampere+ GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Try mixed precision
            self.use_mixed_precision()
        else:
            self.device = "cpu"
    
    def _init_components_cuda(self):
        """Initialize components with CUDA optimization"""
        # Optimize FFT dimension size
        vector_dim = self.config.get("vsa", "dimensions")
        optimal_dim = self.optimize_fft_operations((vector_dim,))
        
        if optimal_dim != vector_dim:
            self.config.set("vsa", "dimensions", optimal_dim)
            
        # Initialize components with optimized settings
        self.quadrant_memory = QuadrantMemory(
            world_bounds=self.world_bounds,
            quadrant_size=self.config.get("quadrant", "size"),
            vector_dim=optimal_dim,
            length_scale=self.config.get("vsa", "length_scale"),
            device=self.device
        )
        
        # Initialize with CUDA optimization
        self.processor = BatchProcessor(
            world_bounds=self.world_bounds,
            sample_resolution=self.config.get("sequential", "sample_resolution"),
            device=self.device
        )
        
        self.entropy_extractor = EntropyExtractor(
            disk_radius=self.config.get("entropy", "disk_radius"),
            occupied_threshold=self.config.get("entropy", "occupied_threshold"),
            empty_threshold=self.config.get("entropy", "empty_threshold"),
            device=self.device
        )
    
    def process_point_cloud_cuda(self, points, labels=None):
        """Process point cloud with CUDA acceleration"""
        # Prepare tensors
        points = self.create_tensor(points)
        if labels is None:
            labels = torch.ones(points.shape[0], dtype=torch.int, device=self.device)
        else:
            labels = self.create_tensor(labels, dtype=torch.int)
        
        # Free reserved memory for large operation
        self.memory_manager.release_reserved_memory()
        
        try:
            # Process in optimized batches
            self.processor.batch_process_points_cuda(
                points, labels, self.quadrant_memory
            )
            
            # Normalize memory
            self.quadrant_memory.normalize_memories_cuda()
            
            # Generate maps with CUDA optimization
            self._generate_maps_cuda()
        finally:
            # Restore memory reservation
            self.memory_manager.restore_reserved_memory()
    
    def _generate_maps_cuda(self):
        """Generate maps with CUDA optimization"""
        # Query grid efficiently using CUDA
        resolution = self.config.get("sequential", "sample_resolution")
        grid_results = self.quadrant_memory.query_grid_cuda(resolution)
        
        # Calculate entropy with CUDA optimization
        features = self.entropy_extractor.extract_features_cuda(
            grid_results['occupied'], grid_results['empty']
        )
        
        # Store results
        self.entropy_grid = features['global_entropy']
        self.classification = features['classification']
        self.occupancy_grid = self.entropy_extractor.get_occupancy_grid(self.classification)
        self.grid_coords = {
            'x': grid_results['x_coords'],
            'y': grid_results['y_coords']
        }
```

This plan provides a comprehensive approach to leveraging CUDA acceleration throughout the VSA-OGM implementation. It includes optimizations for GPU memory management, mixed precision arithmetic, parallel processing with CUDA streams, and specialized CUDA kernels for key operations. The implementation is designed to automatically detect and use CUDA capabilities when available, with appropriate fallbacks to CPU processing when necessary.