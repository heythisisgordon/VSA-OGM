# Systems Architecture Document: Pythonic Sequential VSA-OGM System

## 1. System Overview

The Sequential VSA-OGM system provides a pythonic implementation for converting static 2D point clouds into probabilistic occupancy grid maps. It utilizes hyperdimensional computing via Vector Symbolic Architectures (VSA) and employs Shannon entropy for feature extraction.

This system processes point clouds by sampling observation points sequentially across a grid, simulating how a robot with sensors would build a map while exploring an environment. At each sample location, only points within sensor range are processed, updating quadrant-based memory vectors.

## 2. Core Components

### 2.1 VSAMapper
- **Main entry point** for the system
- Manages overall processing and configuration
- Coordinates interactions between other components

### 2.2 QuadrantMemory
- **Memory storage system** based on vector symbolic architectures
- Divides the world into quadrants for efficient memory usage
- Maintains occupancy and emptiness representations per quadrant

### 2.3 SequentialProcessor 
- **Processes point clouds sequentially**
- Samples observation points on a grid
- Simulates sensor readings at each location

### 2.4 SpatialIndex
- **Indexes spatial data** for efficient querying
- Provides methods to find points within a specific range
- Optimized for quadrant-based lookups

### 2.5 EntropyExtractor
- **Implements Shannon entropy** for feature extraction
- Applies optimized disk filters to probability fields
- Provides classification mechanisms for occupied/empty determination

### 2.6 VectorOperations
- **Performs VSA operations** (binding, power, etc.)
- Optimized for batch processing
- CUDA-accelerated where possible

## 3. Data Flow

1. **Initialization**
   - Configure system parameters (vector dimensionality, decision thresholds, etc.)
   - Initialize quadrant-based memory structure
   - Create spatial index for efficient point lookup
   
2. **Sequential Processing**
   - Generate grid of sample positions
   - For each position:
     - Query points within sensor range
     - Filter to relevant points (within current sensor view)
     - Encode points into hyperdimensional vectors
     - Update quadrant memories with new vectors

3. **Entropy-Based Feature Extraction**
   - Calculate probability fields from memory vectors
   - Apply Shannon entropy extraction using disk filters
   - Generate global entropy map by combining occupied and empty entropies

4. **Map Generation**
   - Apply decision thresholds to classify each cell
   - Create final occupancy grid map
   - Provide visualization of results

## 4. Key Algorithms

### 4.1 Quadrant Memory Update
```
For each point in sensor range:
    1. Determine which quadrant the point belongs to
    2. Create point vector using fractional binding (via FFT)
    3. Add vector to appropriate quadrant memory (occupied or empty)
    4. Normalize memory vector to maintain unit modulus
```

### 4.2 Shannon Entropy Extraction
```
For each quadrant memory:
    1. Calculate dot product between memory vector and query vectors
    2. Apply Born rule to convert to true probabilities
    3. Apply disk filters to compute local entropy
    4. Calculate global entropy by subtracting empty entropy from occupied entropy
    5. Apply decision thresholds to classify cells
```

### 4.3 Sequential Processing
```
1. Create grid of sample positions
2. For each position:
    a. Find points within sensor range
    b. Update quadrant memories with these points
    c. Recalculate entropy maps for affected quadrants
3. Combine all updates into final map
```

## 5. Performance Considerations

- **Memory Efficiency**: Constant memory complexity regardless of point cloud size
- **Computational Efficiency**: Leverages batch operations and CUDA acceleration
- **Scalability**: Processes large environments by dividing into manageable quadrants

## 6. System Interface

```python
# Example of high-level public API
class VSAMapper:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the VSA mapper with configuration parameters."""
        pass
        
    def process_point_cloud(self, points: np.ndarray, labels: np.ndarray) -> None:
        """Process a full point cloud in sequential manner."""
        pass
        
    def process_incrementally(self, 
                             sample_resolution: float = 1.0,
                             sensor_range: float = 10.0) -> None:
        """Process the point cloud incrementally from sample positions."""
        pass
        
    def get_occupancy_grid(self) -> np.ndarray:
        """Get the current occupancy grid."""
        pass
        
    def get_entropy_grid(self) -> np.ndarray:
        """Get the entropy grid for visualization."""
        pass
```

## 7. Dependencies

- NumPy: Array operations and numerical processing
- PyTorch: Tensor operations and CUDA acceleration
- Matplotlib: Visualization and plotting (optional)
- tqdm: Progress monitoring (optional)

This architecture provides a foundation for an efficient, pythonic implementation of the VSA-OGM approach while maintaining the core concepts from the original paper that enable its high performance.