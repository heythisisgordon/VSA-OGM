# Grid-Based Sequential VSA-OGM Mapping Pipeline

Here's an outline for a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. Grid-Based Sequential VSA-OGM Mapping offers a systematic approach to converting a single, final 2D point cloud into a probabilistic occupancy grid mapping (OGM) using Vector Symbolic Architectures (VSA). Unlike batch processing of an entire point cloud, this method samples observation points at regular intervals across a grid pattern, processing only local points within range of an emulated sensor at each location. By treating each grid point as a robot pose keyframe, it mimics realistic exploration while avoiding redundant computation. The approach leverages VSA's efficient incremental update capabilities and Shannon entropy feature extraction, maintaining quadrant-level processing for computational efficiency. This sequential method significantly reduces memory requirements and processing time compared to global batch processing, making it suitable for large-scale mapping applications while preserving the real-time performance benefits highlighted in the original VSA-OGM research.

## 1. Initialization
- Set up environment bounds (X meters × Y meters - note: expect large values like 200 m x 300 m for large mine tunnel systems, for example.)
- Define grid sampling resolution (e.g., every 5m)
- Set emulated sensor characteristics. We will emulate a 2D lidar with 10 meters of range, 360 degrees of coverage, and 1 degree beam spacing.
- Initialize empty global map structure
- Create VSA components (axis vectors only - we'll create quadrant memories on demand)
- Define VSA dimensionality and length scale parameters

## 2. Generate Sampling Grid
- Create grid of potential sampling points
- Filter out points that would fall in known occupied spaces
- Create raster traversal order

## 3. Sequential Processing Loop
For each sampling point in the traversal order:
  - Extract local point cloud (points within sensor range of 10m)
  - Filter points based on sensor model (visibility, ray-casting for occlusion)
  - Process only these local points using VSA operations:
    - Encode points into VSA representation
    - If a quadrant is encountered for the first time, initialize its memory vectors
    - Update only the quadrant memories affected by these local observations
    - Maintain separate VSA quadrant grid system (distinct from sampling grid)
  - Update global map with new information
  - Update occupancy knowledge to refine future sampling locations

## 4. Map Maintenance
- Normalize memory vectors after updates
- Implement memory cleanup for areas with high confidence
- Apply Shannon entropy to enhance feature extraction and reduce noise
- Maintain sparse representation for efficiency (only store occupied quadrants)

## 5. Visualization & Output
- Generate occupancy grid maps for visualization
- Output probability heatmaps
- Calculate confidence metrics

## Implementation Considerations

### Local Processing
```python
def process_sample_point(self, sample_point, global_point_cloud):
    # Extract local point cloud within sensor range
    local_points = extract_points_within_range(sample_point, global_point_cloud, self.sensor_range)
    
    # Filter based on visibility/occlusion (ray casting)
    visible_points = apply_visibility_filter(sample_point, local_points)
    
    # Process only these points (much smaller batch)
    self.process_observation(visible_points, visible_points_labels)
    
    # Only update affected quadrants (not the entire map)
    affected_quadrants = identify_affected_quadrants(visible_points)
    update_quadrant_memories(affected_quadrants)
```

### Incremental Map Building
```python
def build_map_sequentially(self, point_cloud, sampling_resolution=5.0):
    # Create sampling grid
    sampling_points = generate_sampling_grid(self.world_bounds, sampling_resolution)
    
    # Filter out occupied sampling locations
    valid_sampling_points = filter_unoccupied_locations(sampling_points, self.current_map)
    
    # Process each sampling point sequentially
    for sample_point in valid_sampling_points:
        self.process_sample_point(sample_point, point_cloud)
        
        # Optional: visualize incremental map building
        if self.verbose:
            self.visualize_current_map()
```

### Sparse Quadrant Memory Management
```python
def get_or_create_quadrant_memory(self, quadrant_index):
    """Retrieve existing quadrant memory or create if it doesn't exist"""
    if quadrant_index not in self.quadrant_memories:
        # Initialize new memory vectors for this quadrant
        self.quadrant_memories[quadrant_index] = {
            'occupied': torch.zeros(self.vsa_dimensions, device=self.device),
            'empty': torch.zeros(self.vsa_dimensions, device=self.device)
        }
    return self.quadrant_memories[quadrant_index]
```

This pipeline maintains the key advantages of VSA-OGM (efficient updates, probabilistic representation) while processing data in a way that mimics robot exploration, greatly improving performance over batch processing the entire point cloud at once.