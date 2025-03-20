# CUDA Acceleration for VSA-OGM

## Overview

This document outlines a plan to implement CUDA acceleration for the VSA-OGM system to leverage the RTX 4070 GPU. The current implementation is CPU-bound, with the primary bottleneck being the construction of the XY axis matrix. By moving these computationally intensive operations to the GPU, we can achieve significant speedups.

## Implementation Steps

### 1. Update Device Configuration

- Modify the configuration to properly detect and use CUDA when available
- Update the device assignment in the OGM2D_V4 class to use CUDA by default when available

```python
# Current code
"device": "cpu" if torch.cuda.is_available() else "cpu",

# Updated code
"device": "cuda" if torch.cuda.is_available() else "cpu",
```

### 2. Optimize Tensor Operations for GPU

- Ensure all tensor operations are CUDA-compatible
- Move tensors to the GPU at creation time rather than after operations
- Batch operations where possible to maximize GPU utilization

### 3. Parallelize XY Axis Matrix Construction

- Refactor the `_build_xy_axis_matrix` method to use CUDA parallelism
- Replace the nested for-loops with vectorized operations that can run on the GPU
- Implement a batched approach for large matrices to avoid GPU memory limitations

```python
def _build_xy_axis_matrix(self) -> None:
    """
    Build the XY axis matrix using GPU acceleration.
    """
    if self.verbose:
        print("Building XY axis matrix...")

    x_shape: int = self.xy_axis_linspace[0].shape[0]
    y_shape: int = self.xy_axis_linspace[1].shape[0]

    # Create a meshgrid of all x,y coordinates
    x_coords = self.xy_axis_linspace[0].unsqueeze(1).repeat(1, y_shape)
    y_coords = self.xy_axis_linspace[1].unsqueeze(0).repeat(x_shape, 1)
    
    # Reshape to [num_points, 2]
    coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
    
    # Process in batches to avoid GPU memory issues
    batch_size = 10000  # Adjust based on GPU memory
    num_batches = (coords.shape[0] + batch_size - 1) // batch_size
    
    # Initialize the output matrix
    self.xy_axis_matrix = torch.zeros(
        (x_shape, y_shape, self.vsa_dimensions),
        device=self.device
    )
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, coords.shape[0])
        batch_coords = coords[start_idx:end_idx]
        
        # Compute vectors for this batch
        x_vectors = spf.power(self.xy_axis_vectors[0], batch_coords[:, 0], self.length_scale)
        y_vectors = spf.power(self.xy_axis_vectors[1], batch_coords[:, 1], self.length_scale)
        
        # Bind vectors
        batch_result = spf.bind_batch([x_vectors, y_vectors], self.device)
        
        # Place results in the output matrix
        batch_indices = torch.arange(start_idx, end_idx)
        x_indices = batch_indices // y_shape
        y_indices = batch_indices % y_shape
        
        for j in range(len(batch_indices)):
            self.xy_axis_matrix[x_indices[j], y_indices[j], :] = batch_result[j]
    
    if self.verbose:
        print("Finished building XY axis matrix.")
```

### 4. Implement Batch Processing for Vector Operations

- Add a new `bind_batch` function to `spl.functional` that processes multiple vector bindings in parallel
- Optimize the `power` function for batch operations on the GPU

```python
def bind_batch(vectors_list: list[torch.tensor], device: torch.device) -> torch.tensor:
    """
    Bind a batch of vectors using element-wise multiplication in the frequency domain.
    
    Args:
        vectors_list: A list of tensors, each of shape [batch_size, vsa_dimensions]
        device: The device to perform the operation on
        
    Returns:
        A tensor of shape [batch_size, vsa_dimensions] containing the bound vectors
    """
    # Convert to frequency domain
    vectors_fd = [torch.fft.fft(v) for v in vectors_list]
    
    # Multiply in frequency domain (element-wise)
    result_fd = vectors_fd[0]
    for v in vectors_fd[1:]:
        result_fd = result_fd * v
    
    # Convert back to time domain
    result = torch.fft.ifft(result_fd).real
    
    return result
```

### 5. Optimize Quadrant Memory Updates

- Refactor the point vector encoding and quadrant memory updates to use batched operations
- Implement parallel updates for occupied and empty quadrant memory vectors

### 6. Update Testing Infrastructure

- Add GPU memory monitoring to track memory usage during processing
- Create benchmarking scripts to compare CPU vs. GPU performance
- Implement fallback mechanisms for systems without CUDA support

## Performance Considerations

- Monitor GPU memory usage to avoid out-of-memory errors
- Use mixed precision (FP16) where appropriate to improve performance
- Consider using CUDA streams for overlapping computation and memory transfers
- Profile the code to identify and optimize remaining bottlenecks

## Testing Plan

1. Create unit tests for the GPU-accelerated functions
2. Benchmark performance on different dataset sizes
3. Compare results between CPU and GPU implementations to ensure correctness
4. Test on the RTX 4070 GPU to verify performance improvements

## Expected Outcomes

- Significant speedup (10x or more) for the XY axis matrix construction
- Overall processing time reduced by at least 5x
- Ability to process larger datasets more efficiently
- Maintained accuracy compared to the CPU implementation
