# Efficient Full-Resolution Point Cloud Processing for VSA-OGM

## Overview

This document outlines a plan to implement an efficient algorithm for processing full-resolution 2D point clouds using the VSA-OGM approach. The current implementation processes all points in the occupancy grid, which leads to performance issues with large grids. This refactoring will focus on algorithmic improvements to handle full-resolution data efficiently.

## Current Bottlenecks

1. **XY Axis Matrix Construction**: Building the matrix of query points is computationally expensive and memory-intensive.
2. **Sequential Processing**: Many operations are performed sequentially that could be parallelized.
3. **Redundant Computations**: The same vector operations are repeated for points in the same vicinity.
4. **Memory Usage**: The full matrix of all possible query points is stored in memory.

## Implementation Strategy

### 1. Hierarchical Processing with Adaptive Resolution

Implement a multi-resolution approach that processes different regions of the point cloud at different resolutions based on information content:

```python
def process_observation_hierarchical(self, point_cloud, labels):
    # Divide the world into regions
    regions = self._divide_into_regions(point_cloud, labels)
    
    # Process each region with appropriate resolution
    for region in regions:
        if region.information_content > self.high_info_threshold:
            # Process high-information regions at full resolution
            self._process_region(region, resolution_factor=1.0)
        elif region.information_content > self.medium_info_threshold:
            # Process medium-information regions at half resolution
            self._process_region(region, resolution_factor=0.5)
        else:
            # Process low-information regions at quarter resolution
            self._process_region(region, resolution_factor=0.25)
```

### 2. Sparse Matrix Representation

Replace the dense XY axis matrix with a sparse representation that only computes and stores values for relevant query points:

```python
def _build_sparse_xy_axis_matrix(self, query_points):
    """
    Build a sparse XY axis matrix only for the specified query points.
    
    Args:
        query_points: Tensor of shape [N, 2] containing the query points
        
    Returns:
        Sparse tensor containing the XY axis matrix values
    """
    num_points = query_points.shape[0]
    
    # Initialize sparse matrix
    indices = []
    values = torch.zeros((num_points, self.vsa_dimensions), device=self.device)
    
    # Compute values for each query point
    for i, point in enumerate(query_points):
        x, y = point
        
        # Convert to grid indices
        x_idx = int((x - self.world_bounds[0]) / self.axis_resolution)
        y_idx = int((y - self.world_bounds[2]) / self.axis_resolution)
        
        # Compute vector for this point
        vs = [
            spf.power(self.xy_axis_vectors[0], x, self.length_scale),
            spf.power(self.xy_axis_vectors[1], y, self.length_scale)
        ]
        values[i] = spf.bind(vs, self.device)
        indices.append([x_idx, y_idx])
    
    # Create sparse tensor
    indices = torch.tensor(indices, device=self.device)
    shape = (self.grid_size_x, self.grid_size_y, self.vsa_dimensions)
    
    return torch.sparse_coo_tensor(
        indices.t(), values, shape, device=self.device
    )
```

### 3. Importance Sampling

Implement importance sampling to focus computational resources on the most informative regions of the point cloud:

```python
def _sample_query_points(self, point_cloud, labels):
    """
    Sample query points based on information content.
    
    Args:
        point_cloud: Tensor of shape [N, 2] containing point coordinates
        labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
        
    Returns:
        Tensor of shape [M, 2] containing sampled query points
    """
    # Calculate information density
    density = self._calculate_information_density(point_cloud, labels)
    
    # Sample points with probability proportional to information density
    sample_probs = density / density.sum()
    num_samples = min(self.max_query_points, point_cloud.shape[0])
    
    # Sample indices
    indices = torch.multinomial(sample_probs, num_samples, replacement=False)
    
    # Return sampled points
    return point_cloud[indices]
```

### 4. Quadtree-Based Spatial Partitioning

Replace the fixed quadrant hierarchy with an adaptive quadtree that subdivides regions based on point density:

```python
class AdaptiveQuadtree:
    def __init__(self, max_depth=6, min_points=10, max_points=100):
        self.max_depth = max_depth
        self.min_points = min_points
        self.max_points = max_points
        self.root = None
        
    def build(self, points):
        """Build the quadtree from a set of points"""
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        y_min = points[:, 1].min()
        y_max = points[:, 1].max()
        
        self.root = QuadtreeNode(x_min, x_max, y_min, y_max, 0)
        self.root.build(points, self.max_depth, self.min_points, self.max_points)
        
    def query(self, point):
        """Find the leaf node containing the point"""
        return self.root.query(point)
        
class QuadtreeNode:
    def __init__(self, x_min, x_max, y_min, y_max, depth):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.depth = depth
        self.children = None
        self.points = None
        
    def build(self, points, max_depth, min_points, max_points):
        """Recursively build the quadtree"""
        # Filter points that belong to this node
        mask = ((points[:, 0] >= self.x_min) & 
                (points[:, 0] < self.x_max) & 
                (points[:, 1] >= self.y_min) & 
                (points[:, 1] < self.y_max))
        node_points = points[mask]
        
        # Store points if this is a leaf node
        if self.depth >= max_depth or len(node_points) <= min_points:
            self.points = node_points
            return
            
        # Split if we have too many points
        if len(node_points) > max_points:
            self.children = []
            x_mid = (self.x_min + self.x_max) / 2
            y_mid = (self.y_min + self.y_max) / 2
            
            # Create four children (NW, NE, SW, SE)
            self.children.append(QuadtreeNode(self.x_min, x_mid, y_mid, self.y_max, self.depth + 1))
            self.children.append(QuadtreeNode(x_mid, self.x_max, y_mid, self.y_max, self.depth + 1))
            self.children.append(QuadtreeNode(self.x_min, x_mid, self.y_min, y_mid, self.depth + 1))
            self.children.append(QuadtreeNode(x_mid, self.x_max, self.y_min, y_mid, self.depth + 1))
            
            # Recursively build children
            for child in self.children:
                child.build(node_points, max_depth, min_points, max_points)
        else:
            self.points = node_points
```

### 5. Incremental Processing

Implement incremental processing to update the map with new observations without recomputing everything:

```python
def update_map_incrementally(self, new_point_cloud, new_labels):
    """
    Update the map incrementally with new observations.
    
    Args:
        new_point_cloud: Tensor of shape [N, 2] containing new point coordinates
        new_labels: Tensor of shape [N] containing new point labels
    """
    # Determine which quadrants are affected by the new points
    affected_quadrants = self._find_affected_quadrants(new_point_cloud)
    
    # Update only the affected quadrants
    for quadrant_idx in affected_quadrants:
        # Filter points that belong to this quadrant
        mask = self._points_in_quadrant(new_point_cloud, quadrant_idx)
        quadrant_points = new_point_cloud[mask]
        quadrant_labels = new_labels[mask]
        
        # Update quadrant memory vectors
        self._update_quadrant_memory(quadrant_idx, quadrant_points, quadrant_labels)
        
    # Update heatmaps only for affected regions
    self._update_heatmaps(affected_quadrants)
```

### 6. Vector Quantization for Memory Efficiency

Implement vector quantization to reduce memory usage while maintaining accuracy:

```python
def _quantize_vectors(self, vectors, num_centroids=256):
    """
    Quantize vectors to reduce memory usage.
    
    Args:
        vectors: Tensor of shape [N, D] containing vectors to quantize
        num_centroids: Number of centroids to use for quantization
        
    Returns:
        Tuple of (centroids, indices) where:
            centroids: Tensor of shape [num_centroids, D] containing centroid vectors
            indices: Tensor of shape [N] containing indices of centroids for each vector
    """
    # Reshape vectors for kmeans
    vectors_flat = vectors.reshape(-1, vectors.shape[-1])
    
    # Run k-means clustering
    kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit(vectors_flat.cpu().numpy())
    
    # Get centroids and indices
    centroids = torch.tensor(kmeans.cluster_centers_, device=self.device)
    indices = torch.tensor(kmeans.labels_, device=self.device)
    
    return centroids, indices
```

## Implementation Plan

### Phase 1: Sparse Matrix Representation

1. Implement the sparse XY axis matrix representation
2. Modify the query functions to work with sparse matrices
3. Update the heatmap generation to use sparse operations

### Phase 2: Adaptive Resolution and Importance Sampling

1. Implement the information density calculation
2. Develop the importance sampling algorithm
3. Create the hierarchical processing framework

### Phase 3: Quadtree-Based Spatial Partitioning

1. Implement the AdaptiveQuadtree class
2. Replace the fixed quadrant hierarchy with the adaptive quadtree
3. Update the quadrant memory vector storage to work with the quadtree

### Phase 4: Incremental Processing

1. Implement the affected quadrant detection
2. Develop the incremental update algorithm
3. Modify the process_observation method to use incremental updates when possible

### Phase 5: Memory Optimization

1. Implement vector quantization
2. Add support for compressed storage of memory vectors
3. Optimize memory usage throughout the pipeline

## Testing and Validation

1. Create benchmark datasets of varying sizes and complexities
2. Measure processing time and memory usage for each implementation phase
3. Compare accuracy between the original and optimized implementations
4. Validate on real-world datasets

## Expected Outcomes

- Ability to process full-resolution point clouds with millions of points
- Reduced memory usage by at least 80%
- Processing time reduced by at least 10x (combined with CUDA acceleration)
- Maintained or improved accuracy compared to the original implementation
- Support for incremental updates with minimal recomputation
