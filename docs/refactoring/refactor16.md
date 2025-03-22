# Implementation Plan: Octree-Based Incremental VSA-OGM

## Overview
This plan combines an octree with incremental mapping to efficiently process a 2D point cloud. It simulates robot exploration by sampling unoccupied points and building a probabilistic occupancy map incrementally.

## Core Components

### 1. Quadtree for Spatial Indexing
- Create a quadtree class to efficiently index the point cloud
- Support insertion of points with labels (occupied=1, free=0)
- Implement range queries to find points within a given radius
- Track whether nodes contain occupied points for safe sampling

### 2. Robot Position Sampling
- Implement a method to find valid sampling positions (unoccupied areas)
- Ensure a minimum safety distance from occupied points
- Generate a reasonable coverage of the explorable space
- Support both grid-based and adaptive sampling strategies

### 3. Incremental VSA Mapper
- Extend the existing VSAMapper class
- Add methods for incremental processing of observations
- Support horizon distance limiting for each sample position
- Process only visible points at each step

## Implementation Plan

1. Create the Quadtree class:
   - Implement node structure with bounds, children, and points
   - Add methods for point insertion and range queries
   - Include a method to check if a region is free of occupied points

2. Extend the VSAMapper class:
   - Add quadtree initialization from point cloud
   - Implement incremental processing method
   - Add tracking of processed regions

3. Modify the main processing function:
   - Accept additional parameters for incremental processing
   - Support horizon distance and safety margin settings
   - Add option to limit number of samples

4. Implement sampling strategy:
   - Start with simple grid-based sampling with occupied-space checking
   - Sample at resolution consistent with desired map detail
   - Skip samples that would be in occupied space

5. Update the CLI interface:
   - Add command-line options for incremental processing
   - Include parameters for horizon distance and safety margin
   - Add option to specify sampling approach

## Key Functions to Implement

```python
# Quadtree implementation for efficient spatial queries
def insert_point(self, point, label)
def query_range(self, center, radius)
def is_region_free(self, bounds, safety_margin)

# Sample valid robot positions
def get_next_sample_position(self, quadtree, safety_margin)
def generate_sample_grid(self, resolution, world_bounds)

# Incremental processing methods
def initialize_with_pointcloud(self, points, labels)
def process_incrementally(self, sample_positions=None, horizon_distance=10.0)
def process_at_position(self, position, horizon_distance)
```

## Parameters to Add

- `horizon_distance`: Maximum distance from sample point to consider points (default: 10.0)
- `safety_margin`: Minimum distance from occupied points for sampling (default: 0.5)
- `sample_resolution`: Resolution of the sampling grid (default: map resolution)
- `max_samples`: Maximum number of sample positions to process (optional)

## Expected Behavior

1. Initialize quadtree with full point cloud
2. Generate or load sample positions avoiding occupied areas
3. For each sample position:
   - Find all points within horizon distance
   - Process only these points with VSA-OGM
   - Update the global occupancy map
4. Continue until all samples are processed or coverage is complete
5. Return the final occupancy grid

## Performance Benefits

- Quadtree enables O(log n) point retrieval versus O(n) linear search
- Processing only points within horizon reduces computational load
- Incremental approach mimics real-world sensor limitations
- Avoiding occupied sample points ensures realistic exploration