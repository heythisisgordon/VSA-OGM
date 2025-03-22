# Detailed Implementation Plan: Octree-Based Incremental VSA-OGM

## Overview

This document provides a detailed implementation plan for integrating a quadtree-based spatial indexing system with incremental mapping capabilities into the existing VSA-OGM framework. The goal is to efficiently process 2D point clouds by simulating robot exploration, sampling unoccupied points, and building a probabilistic occupancy map incrementally.

## Core Components

### 1. Quadtree for Spatial Indexing

#### 1.1 Quadtree Node Structure

```python
class QuadtreeNode:
    def __init__(self, bounds, depth=0, max_depth=8, max_points=10):
        self.bounds = bounds  # [x_min, x_max, y_min, y_max]
        self.depth = depth
        self.max_depth = max_depth
        self.max_points = max_points
        self.children = None  # Will be a list of 4 children when split
        self.points = []  # List of (point, label) tuples
        self.contains_occupied = False  # Flag to track if node contains occupied points
        self.center = [
            (bounds[0] + bounds[1]) / 2,  # x center
            (bounds[2] + bounds[3]) / 2   # y center
        ]
```

#### 1.2 Quadtree Class

```python
class Quadtree:
    def __init__(self, bounds, max_depth=8, max_points=10):
        """
        Initialize a quadtree for efficient spatial indexing.
        
        Args:
            bounds: World bounds [x_min, x_max, y_min, y_max]
            max_depth: Maximum depth of the tree
            max_points: Maximum points per leaf node before splitting
        """
        self.root = QuadtreeNode(bounds, max_depth=max_depth, max_points=max_points)
        self.bounds = bounds
        self.max_depth = max_depth
        self.max_points = max_points
    
    def insert_point(self, point, label):
        """Insert a point with label into the quadtree"""
        # Implementation details in section 1.3
    
    def query_range(self, center, radius):
        """Find all points within a given radius of center"""
        # Implementation details in section 1.4
    
    def is_region_free(self, bounds, safety_margin):
        """Check if a region is free of occupied points"""
        # Implementation details in section 1.5
```

#### 1.3 Point Insertion Method

```python
def insert_point(self, point, label):
    """
    Insert a point with label into the quadtree.
    
    Args:
        point: [x, y] coordinates
        label: 0 for free, 1 for occupied
    """
    def _insert_point_recursive(node, point, label):
        # Check if point is within node bounds
        if not (node.bounds[0] <= point[0] <= node.bounds[1] and
                node.bounds[2] <= point[1] <= node.bounds[3]):
            return False
        
        # Update occupied flag if needed
        if label == 1:
            node.contains_occupied = True
        
        # If node is a leaf with space or at max depth, add point
        if node.children is None:
            if len(node.points) < node.max_points or node.depth >= node.max_depth:
                node.points.append((point, label))
                return True
            else:
                # Split node if full and not at max depth
                self._split_node(node)
                # Continue with insertion after split
        
        # Find appropriate child and insert there
        for child in node.children:
            if _insert_point_recursive(child, point, label):
                return True
        
        return False
    
    return _insert_point_recursive(self.root, point, label)

def _split_node(self, node):
    """Split a node into four children"""
    x_min, x_max, y_min, y_max = node.bounds
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    # Create four children (SW, SE, NW, NE)
    node.children = [
        QuadtreeNode([x_min, x_mid, y_min, y_mid], node.depth + 1, node.max_depth, node.max_points),
        QuadtreeNode([x_mid, x_max, y_min, y_mid], node.depth + 1, node.max_depth, node.max_points),
        QuadtreeNode([x_min, x_mid, y_mid, y_max], node.depth + 1, node.max_depth, node.max_points),
        QuadtreeNode([x_mid, x_max, y_mid, y_max], node.depth + 1, node.max_depth, node.max_points)
    ]
    
    # Redistribute points to children
    for point, label in node.points:
        for child in node.children:
            if (child.bounds[0] <= point[0] <= child.bounds[1] and
                child.bounds[2] <= point[1] <= child.bounds[3]):
                child.points.append((point, label))
                if label == 1:
                    child.contains_occupied = True
                break
    
    # Clear points from parent
    node.points = []
```

#### 1.4 Range Query Method

```python
def query_range(self, center, radius):
    """
    Find all points within a given radius of center.
    
    Args:
        center: [x, y] coordinates of query center
        radius: Search radius
        
    Returns:
        List of (point, label) tuples within the radius
    """
    result = []
    
    def _query_range_recursive(node, center, radius):
        # Check if node intersects with query circle
        if not self._intersects_circle(node.bounds, center, radius):
            return
        
        # If leaf node, check all points
        if node.children is None:
            for point, label in node.points:
                dist = ((point[0] - center[0])**2 + (point[1] - center[1])**2)**0.5
                if dist <= radius:
                    result.append((point, label))
        else:
            # Recursively check children
            for child in node.children:
                _query_range_recursive(child, center, radius)
    
    _query_range_recursive(self.root, center, radius)
    return result

def _intersects_circle(self, bounds, center, radius):
    """Check if a bounding box intersects with a circle"""
    # Find closest point on rectangle to circle center
    closest_x = max(bounds[0], min(center[0], bounds[1]))
    closest_y = max(bounds[2], min(center[1], bounds[3]))
    
    # Calculate distance from closest point to circle center
    dist = ((closest_x - center[0])**2 + (closest_y - center[1])**2)**0.5
    
    return dist <= radius
```

#### 1.5 Region Safety Check Method

```python
def is_region_free(self, bounds, safety_margin):
    """
    Check if a region is free of occupied points with a safety margin.
    
    Args:
        bounds: Region bounds [x_min, x_max, y_min, y_max]
        safety_margin: Minimum distance from occupied points
        
    Returns:
        True if region is free, False otherwise
    """
    # Expand bounds by safety margin
    expanded_bounds = [
        bounds[0] - safety_margin,
        bounds[1] + safety_margin,
        bounds[2] - safety_margin,
        bounds[3] + safety_margin
    ]
    
    def _check_region_recursive(node):
        # If node doesn't intersect expanded bounds, it's safe
        if not self._intersects_rectangle(node.bounds, expanded_bounds):
            return True
        
        # If node contains occupied points, check distances
        if node.contains_occupied:
            if node.children is None:
                # Leaf node - check all occupied points
                for point, label in node.points:
                    if label == 1:  # Occupied point
                        # Check if point is within safety margin of bounds
                        if (bounds[0] - safety_margin <= point[0] <= bounds[1] + safety_margin and
                            bounds[2] - safety_margin <= point[1] <= bounds[3] + safety_margin):
                            # Calculate minimum distance to bounds
                            dx = max(bounds[0] - point[0], 0, point[0] - bounds[1])
                            dy = max(bounds[2] - point[1], 0, point[1] - bounds[3])
                            dist = (dx**2 + dy**2)**0.5
                            if dist < safety_margin:
                                return False
            else:
                # Check children
                for child in node.children:
                    if not _check_region_recursive(child):
                        return False
        
        return True
    
    return _check_region_recursive(self.root)

def _intersects_rectangle(self, bounds1, bounds2):
    """Check if two rectangles intersect"""
    return not (bounds1[1] < bounds2[0] or bounds1[0] > bounds2[1] or
                bounds1[3] < bounds2[2] or bounds1[2] > bounds2[3])
```

### 2. Robot Position Sampling

#### 2.1 Grid-Based Sampling

```python
def generate_sample_grid(self, resolution, world_bounds, safety_margin):
    """
    Generate a grid of sample positions avoiding occupied areas.
    
    Args:
        resolution: Sampling resolution
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        safety_margin: Minimum distance from occupied points
        
    Returns:
        List of valid sample positions
    """
    x_min, x_max, y_min, y_max = world_bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate number of samples in each dimension
    nx = int(x_range / resolution) + 1
    ny = int(y_range / resolution) + 1
    
    valid_samples = []
    
    # Check each grid point
    for i in range(nx):
        for j in range(ny):
            x = x_min + i * resolution
            y = y_min + j * resolution
            
            # Create a small region around the sample point
            sample_bounds = [
                x - resolution/2,
                x + resolution/2,
                y - resolution/2,
                y + resolution/2
            ]
            
            # Check if region is free of occupied points
            if self.is_region_free(sample_bounds, safety_margin):
                valid_samples.append([x, y])
    
    return valid_samples
```

#### 2.2 Adaptive Sampling

```python
def generate_adaptive_samples(self, min_resolution, max_resolution, world_bounds, safety_margin, max_samples=1000):
    """
    Generate adaptive sample positions with higher density in complex areas.
    
    Args:
        min_resolution: Minimum sampling resolution for complex areas
        max_resolution: Maximum sampling resolution for open areas
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        safety_margin: Minimum distance from occupied points
        max_samples: Maximum number of samples to generate
        
    Returns:
        List of valid sample positions
    """
    # Start with coarse grid
    samples = self.generate_sample_grid(max_resolution, world_bounds, safety_margin)
    
    # Refine grid in areas with high point density
    refined_samples = []
    
    for sample in samples:
        # Check point density in region
        points_in_range = self.query_range(sample, max_resolution * 2)
        point_density = len(points_in_range) / (np.pi * (max_resolution * 2)**2)
        
        # If density is high, add more samples
        if point_density > 0.1:  # Threshold can be adjusted
            # Calculate adaptive resolution based on density
            adaptive_res = max(min_resolution, max_resolution / (1 + point_density * 10))
            
            # Add refined samples
            for dx in [-adaptive_res/2, adaptive_res/2]:
                for dy in [-adaptive_res/2, adaptive_res/2]:
                    new_sample = [sample[0] + dx, sample[1] + dy]
                    
                    # Check if new sample is valid
                    sample_bounds = [
                        new_sample[0] - adaptive_res/4,
                        new_sample[0] + adaptive_res/4,
                        new_sample[1] - adaptive_res/4,
                        new_sample[1] + adaptive_res/4
                    ]
                    
                    if self.is_region_free(sample_bounds, safety_margin):
                        refined_samples.append(new_sample)
        
        # Add original sample
        refined_samples.append(sample)
        
        # Limit number of samples
        if len(refined_samples) >= max_samples:
            break
    
    return refined_samples[:max_samples]
```

#### 2.3 Sample Position Selection

```python
def get_next_sample_position(self, processed_samples, remaining_samples, current_position=None):
    """
    Get the next best sample position based on distance and coverage.
    
    Args:
        processed_samples: List of already processed sample positions
        remaining_samples: List of remaining sample positions
        current_position: Current position (if None, choose the first sample)
        
    Returns:
        Next sample position
    """
    if not remaining_samples:
        return None
    
    if current_position is None or not processed_samples:
        # Start with the sample closest to the origin
        distances = [np.sqrt(s[0]**2 + s[1]**2) for s in remaining_samples]
        return remaining_samples[np.argmin(distances)]
    
    # Find the closest unprocessed sample to current position
    distances = [np.sqrt((s[0] - current_position[0])**2 + 
                         (s[1] - current_position[1])**2) 
                 for s in remaining_samples]
    
    return remaining_samples[np.argmin(distances)]
```

### 3. Incremental VSA Mapper

#### 3.1 Extended VSAMapper Class

```python
class IncrementalVSAMapper(VSAMapper):
    """
    Extended VSA Mapper with incremental processing capabilities.
    """
    
    def __init__(self, config, device=None):
        """
        Initialize the incremental VSA mapper.
        
        Args:
            config: Configuration dictionary with additional parameters:
                - horizon_distance: Maximum distance to consider points
                - safety_margin: Minimum distance from occupied points
                - sample_resolution: Resolution for sampling
            device: Device to use for computation
        """
        super().__init__(config, device)
        
        # Extract additional configuration parameters
        self.horizon_distance = config.get("horizon_distance", 10.0)
        self.safety_margin = config.get("safety_margin", 0.5)
        self.sample_resolution = config.get("sample_resolution", self.resolution * 5)
        
        # Initialize quadtree
        self.quadtree = None
        
        # Track processed regions
        self.processed_samples = []
        self.remaining_samples = []
        self.current_position = None
        
        # Track incremental processing state
        self.is_initialized = False
```

#### 3.2 Quadtree Initialization

```python
def initialize_with_pointcloud(self, points, labels):
    """
    Initialize the quadtree with a point cloud.
    
    Args:
        points: Tensor of shape [N, 2] containing point coordinates
        labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels
    
    # Initialize quadtree
    self.quadtree = Quadtree(
        bounds=[
            self.world_bounds[0],
            self.world_bounds[1],
            self.world_bounds[2],
            self.world_bounds[3]
        ],
        max_depth=8,
        max_points=50
    )
    
    # Insert points into quadtree
    for i in range(len(points_np)):
        self.quadtree.insert_point(points_np[i], labels_np[i])
    
    # Generate sample positions
    self.remaining_samples = self.quadtree.generate_sample_grid(
        resolution=self.sample_resolution,
        world_bounds=self.world_bounds,
        safety_margin=self.safety_margin
    )
    
    self.is_initialized = True
    
    if self.verbose:
        print(f"Initialized quadtree with {len(points_np)} points")
        print(f"Generated {len(self.remaining_samples)} sample positions")
```

#### 3.3 Incremental Processing Method

```python
def process_incrementally(self, max_samples=None):
    """
    Process the point cloud incrementally from sample positions.
    
    Args:
        max_samples: Maximum number of samples to process (None for all)
    """
    if not self.is_initialized:
        raise ValueError("Mapper not initialized. Call initialize_with_pointcloud first.")
    
    # Limit number of samples if specified
    if max_samples is not None:
        max_samples = min(max_samples, len(self.remaining_samples))
    else:
        max_samples = len(self.remaining_samples)
    
    # Process samples one by one
    for i in range(max_samples):
        # Get next sample position
        self.current_position = self.quadtree.get_next_sample_position(
            self.processed_samples,
            self.remaining_samples,
            self.current_position
        )
        
        if self.current_position is None:
            break
        
        # Process at current position
        self.process_at_position(self.current_position)
        
        # Update processed and remaining samples
        self.processed_samples.append(self.current_position)
        self.remaining_samples.remove(self.current_position)
        
        if self.verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{max_samples} sample positions")
    
    if self.verbose:
        print(f"Completed incremental processing with {len(self.processed_samples)} samples")
```

#### 3.4 Position-Based Processing Method

```python
def process_at_position(self, position):
    """
    Process points visible from a specific position.
    
    Args:
        position: [x, y] coordinates of the observation position
    """
    # Query points within horizon distance
    visible_points = self.quadtree.query_range(position, self.horizon_distance)
    
    if not visible_points:
        return
    
    # Convert to torch tensors
    points = np.array([p[0] for p in visible_points])
    labels = np.array([p[1] for p in visible_points])
    
    points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
    labels_tensor = torch.tensor(labels, dtype=torch.int, device=self.device)
    
    # Process these points with the standard VSA mapper
    super().process_observation(points_tensor, labels_tensor)
```

### 4. Main Processing Function Updates

#### 4.1 Updated pointcloud_to_ogm Function

```python
def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    axis_resolution: float = 0.5,
    vsa_dimensions: int = 16000,
    use_cuda: bool = True,
    verbose: bool = False,
    incremental: bool = False,
    horizon_distance: float = 10.0,
    safety_margin: float = 0.5,
    sample_resolution: Optional[float] = None,
    max_samples: Optional[int] = None
) -> None:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npy file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        axis_resolution: Resolution for axis vectors (typically 5x grid resolution)
        vsa_dimensions: Dimensionality of VSA vectors
        use_cuda: Whether to use CUDA if available
        verbose: Whether to print verbose output
        incremental: Whether to use incremental processing
        horizon_distance: Maximum distance from sample point to consider points
        safety_margin: Minimum distance from occupied points for sampling
        sample_resolution: Resolution of the sampling grid (default: 5x resolution)
        max_samples: Maximum number of sample positions to process
    """
    # Set device (default to CUDA when available)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading point cloud from {input_file}")
    
    # Load point cloud
    points, labels = io.load_pointcloud(input_file, device)
    
    if verbose:
        print(f"Loaded point cloud with {points.shape[0]} points")
    
    # Set default world bounds if not provided
    if world_bounds is None:
        # Calculate from point cloud with padding
        x_min, y_min = points.min(dim=0).values.cpu().numpy() - 5.0
        x_max, y_max = points.max(dim=0).values.cpu().numpy() + 5.0
        world_bounds = [float(x_min), float(x_max), float(y_min), float(y_max)]
        
        if verbose:
            print(f"Automatically determined world bounds: {world_bounds}")
    
    # Set default sample resolution if not provided
    if sample_resolution is None:
        sample_resolution = resolution * 5
    
    # Create mapper configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "axis_resolution": axis_resolution,
        "vsa_dimensions": vsa_dimensions,
        "quadrant_hierarchy": [4],
        "length_scale": 2.0,
        "use_query_normalization": True,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose,
        "horizon_distance": horizon_distance,
        "safety_margin": safety_margin,
        "sample_resolution": sample_resolution
    }
    
    if verbose:
        print("Initializing VSA mapper...")
    
    # Create appropriate mapper based on incremental flag
    if incremental:
        mapper = IncrementalVSAMapper(config, device=device)
        
        # Initialize and process incrementally
        if verbose:
            print("Initializing quadtree with point cloud...")
        
        mapper.initialize_with_pointcloud(points, labels)
        
        if verbose:
            print("Processing point cloud incrementally...")
        
        mapper.process_incrementally(max_samples)
    else:
        # Use standard mapper
        mapper = VSAMapper(config, device=device)
        
        # Process point cloud
        if verbose:
            print("Processing point cloud...")
        
        mapper.process_observation(points, labels)
    
    # Get and save occupancy grid
    if verbose:
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    io.save_occupancy_grid(grid, output_file, metadata={
        "world_bounds": world_bounds, 
        "resolution": resolution,
        "incremental": incremental,
        "horizon_distance": horizon_distance if incremental else None,
        "safety_margin": safety_margin if incremental else None,
        "sample_resolution": sample_resolution if incremental else None,
        "processed_samples": len(mapper.processed_samples) if incremental else None
    })
    
    if verbose:
        print(f"Occupancy grid saved to {output_file}")
```

#### 4.2 Updated CLI Interface

```python
def cli_main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid map")
    parser.add_argument("input", help="Input point cloud file (.npy)")
    parser.add_argument("output", help="Output occupancy grid file (.npy)")
    parser.add_argument("--bounds", "-b", nargs=4, type=float, help="World bounds [x_min x_max y_min y_max]")
    parser.add_argument("--resolution", "-r", type=float, default=0.1, help="Grid resolution in meters")
    parser.add_argument("--axis-resolution", "-a", type=float, default=0.5, help="Axis resolution in meters")
    parser.add_argument("--dimensions", "-d", type=int, default=16000, help="VSA dimensions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Add incremental processing options
    parser.add_argument("--incremental", "-i", action="store_true", help="Use incremental processing")
    parser.add_argument("--horizon", type=float, default=10.0, help="Horizon distance for incremental processing")
    parser.add_argument("--safety", type=float, default=0.5, help="Safety margin for sampling")
    parser.add_argument("--sample-resolution", type=float, help="Resolution for sampling grid")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    pointcloud_to_ogm(
        args.input,
        args.output,
        world_bounds=args.bounds,
        resolution=args.resolution,
        axis_resolution=args.axis_resolution,
        vsa_dimensions=args.dimensions,
        use_cuda=not args.cpu,
        verbose=args.verbose,
        incremental=args.incremental,
        horizon_distance=args.horizon,
        safety_margin=args.safety,
        sample_resolution=args.sample_resolution,
        max_samples=args.max_samples
    )
```

## Implementation Strategy

### Phase 1: Quadtree Implementation

1. Create a new file `src/spatial.py` to implement the Quadtree class
2. Implement the QuadtreeNode and Quadtree classes with basic functionality
3. Add methods for point insertion, range queries, and region safety checks
4. Write unit tests to verify quadtree functionality

### Phase 2: Robot Position Sampling

1. Add methods to the Quadtree class for generating sample positions
2. Implement grid-based and adaptive sampling strategies
3. Add method for selecting the next best sample position
4. Write unit tests to verify sampling functionality

### Phase 3: Incremental VSA Mapper

1. Create the IncrementalVSAMapper class extending VSAMapper
2. Implement methods for initializing with a point cloud
3. Add incremental processing methods
4. Write unit tests to verify incremental mapping functionality

### Phase 4: Integration and CLI Updates

1. Update the main processing function to support incremental processing
2. Add command-line options for incremental processing parameters
3. Update documentation and examples
4. Perform integration testing with different point clouds

## Performance Considerations

1. **Memory Efficiency**: The quadtree structure reduces memory usage by only storing points at leaf nodes, avoiding redundant storage.

2. **Computational Efficiency**:
   - Range queries are O(log n) in balanced cases, significantly faster than linear search
   - Processing only points within horizon distance reduces computational load
   - Batch processing of points within each sample position maintains GPU efficiency

3. **Parallelization**:
   - Sample positions can be processed in parallel if needed
   - GPU acceleration is maintained for vector operations within each sample

4. **Scalability**:
   - The approach scales well with point cloud size due to logarithmic search complexity
   - Adaptive sampling ensures efficient coverage of complex environments

## Expected Outcomes

1. **Improved Memory Usage**: Reduced memory footprint for large point clouds
2. **Faster Processing**: More efficient spatial queries and incremental processing
3. **Realistic Simulation**: Better modeling of robot exploration with limited sensing
4. **Flexible Configuration**: Adjustable parameters for different environments and requirements

## Validation Approach

1. **Unit Testing**: Test each component individually
2. **Integration Testing**: Test the complete pipeline with different point clouds
3. **Performance Benchmarking**: Compare memory usage and processing time with the original implementation
4. **Visualization**: Generate visualizations of the incremental mapping process

## Future Extensions

1. **Dynamic Updates**: Support for adding new points to an existing map
2. **Path Planning**: Integrate with path planning algorithms for exploration
3. **3D Extension**: Extend to 3D point clouds using octrees
4. **Multi-Resolution**: Support for multi-resolution mapping with different detail levels
