# Plan to Address Redundant Similarity Calculations

## Problem Statement
The original VSA-OGM code avoids recalculating similarities for the same areas by using spatial indexing, while your implementation may be performing redundant calculations.

## Implementation Plan

### 1. Implement Spatial Grid Caching System

```python
class SpatialGridCache:
    """
    Efficient spatial grid caching system to avoid redundant similarity calculations.
    Stores and retrieves similarity values based on spatial coordinates.
    """
    
    def __init__(self, world_bounds, resolution, device="cuda"):
        """
        Initialize spatial grid cache.
        
        Args:
            world_bounds: World boundaries [x_min, x_max, y_min, y_max]
            resolution: Spatial resolution of the grid
            device: Device for tensor operations
        """
        self.world_bounds = world_bounds
        self.resolution = resolution
        self.device = device
        
        # Calculate grid dimensions
        self.grid_shape = self._calculate_grid_shape()
        
        # Initialize cache storage - None means no value calculated yet
        self.occupied_cache = None
        self.empty_cache = None
    
    def _calculate_grid_shape(self):
        """Calculate grid shape based on world bounds and resolution"""
        x_size = int((self.world_bounds[1] - self.world_bounds[0]) / self.resolution) + 1
        y_size = int((self.world_bounds[3] - self.world_bounds[2]) / self.resolution) + 1
        return (x_size, y_size)
```

### 2. Implement Direct Grid Coordination System

```python
def get_grid_coordinates(self):
    """Get grid coordinate tensors"""
    x_coords = torch.arange(
        self.world_bounds[0], 
        self.world_bounds[1] + 0.5 * self.resolution, 
        self.resolution, 
        device=self.device
    )
    
    y_coords = torch.arange(
        self.world_bounds[2], 
        self.world_bounds[3] + 0.5 * self.resolution, 
        self.resolution, 
        device=self.device
    )
    
    return x_coords, y_coords
```

### 3. Implement Efficient Cache Updates

```python
def update_cache(self, 
                updated_regions, 
                similarity_values, 
                is_occupied: bool):
    """
    Update cache with new similarity values for specific regions.
    
    Args:
        updated_regions: Region indices to update [num_regions, 2]
        similarity_values: New similarity values [num_regions]
        is_occupied: Whether to update occupied or empty cache
    """
    # Initialize cache if not already done
    if is_occupied and self.occupied_cache is None:
        self.occupied_cache = torch.zeros(self.grid_shape, device=self.device)
    elif not is_occupied and self.empty_cache is None:
        self.empty_cache = torch.zeros(self.grid_shape, device=self.device)
    
    # Get target cache
    cache = self.occupied_cache if is_occupied else self.empty_cache
    
    # Update cache values
    for i in range(updated_regions.shape[0]):
        x, y = updated_regions[i]
        cache[x, y] = similarity_values[i]
```

### 4. Implement Direct Spatial Query System

```python
def get_cache_value(self, 
                   x_idx: int, 
                   y_idx: int, 
                   is_occupied: bool) -> torch.Tensor:
    """
    Get cached similarity value for a specific grid cell.
    
    Args:
        x_idx: X index in the grid
        y_idx: Y index in the grid
        is_occupied: Whether to query occupied or empty cache
        
    Returns:
        Cached similarity value or None if not calculated
    """
    cache = self.occupied_cache if is_occupied else self.empty_cache
    
    if cache is None:
        return None
    
    # Check if indices are within bounds
    if 0 <= x_idx < self.grid_shape[0] and 0 <= y_idx < self.grid_shape[1]:
        return cache[x_idx, y_idx]
    
    return None
```

### 5. Implement Efficient Grid-Based Vector Encoding

```python
def precompute_encoded_grid(self, point_encoder):
    """
    Precompute encoded vectors for the entire grid.
    
    Args:
        point_encoder: Point encoder object
        
    Returns:
        Encoded grid with shape [grid_x, grid_y, vector_dim]
    """
    # Get grid coordinates
    x_coords, y_coords = self.get_grid_coordinates()
    
    # Create meshgrid
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Encode all points
    encoded = point_encoder.encode_points_batch(points)
    
    # Reshape to grid
    return encoded.reshape(self.grid_shape[0], self.grid_shape[1], -1)
```

### 6. Implement Selective Region Query and Update

```python
def get_outdated_regions(self, 
                        updated_quadrants, 
                        quadrant_to_grid_map, 
                        is_occupied: bool) -> torch.Tensor:
    """
    Get regions that need recalculation based on updated quadrants.
    
    Args:
        updated_quadrants: Indices of quadrants that have been updated
        quadrant_to_grid_map: Mapping from quadrant indices to grid regions
        is_occupied: Whether to check occupied or empty cache
        
    Returns:
        Indices of grid cells that need recalculation
    """
    # Get all grid cells that belong to updated quadrants
    outdated_regions = []
    
    for q_idx in updated_quadrants:
        if q_idx in quadrant_to_grid_map:
            outdated_regions.append(quadrant_to_grid_map[q_idx])
    
    if not outdated_regions:
        return torch.tensor([], dtype=torch.long, device=self.device)
    
    return torch.cat(outdated_regions, dim=0)
```

### 7. Implement Direct Integration with QuadrantMemory

```python
# In QuadrantMemory class
def __init__(self, world_bounds, quadrant_size, vector_dim, length_scale, device):
    # Other initialization...
    
    # Initialize spatial grid cache
    self.grid_cache = SpatialGridCache(
        world_bounds=world_bounds,
        resolution=length_scale / 2,  # Higher resolution for accuracy
        device=device
    )
    
    # Track which quadrants have been updated
    self.updated_quadrants = set()
    
    # Create mapping from quadrants to grid regions
    self.quadrant_to_grid_map = {}
    self._build_quadrant_grid_mapping()
```

### 8. Implement Optimized Query Grid Function

```python
def query_grid(self, resolution: float) -> Dict[str, torch.Tensor]:
    """
    Query grid with optimized caching to avoid redundant calculations.
    
    Args:
        resolution: Resolution of the grid
        
    Returns:
        Dictionary with grid data
    """
    # Check if we can use cached grid directly
    if resolution == self.grid_cache.resolution:
        # Ensure caches are fully populated
        self._ensure_full_cache_calculation()
        
        return {
            'x_coords': self.grid_cache.get_grid_coordinates()[0],
            'y_coords': self.grid_cache.get_grid_coordinates()[1],
            'occupied': self.grid_cache.occupied_cache,
            'empty': self.grid_cache.empty_cache
        }
    
    # Create new grid at requested resolution
    x_coords = torch.arange(
        self.world_bounds[0], 
        self.world_bounds[1] + 0.5 * resolution, 
        resolution, 
        device=self.device
    )
    
    y_coords = torch.arange(
        self.world_bounds[2], 
        self.world_bounds[3] + 0.5 * resolution, 
        resolution, 
        device=self.device
    )
    
    # Initialize result grids
    occupied_grid = torch.zeros((len(x_coords), len(y_coords)), device=self.device)
    empty_grid = torch.zeros((len(x_coords), len(y_coords)), device=self.device)
    
    # Create meshgrid
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Get quadrant indices
    quadrant_indices = self.get_quadrant_index(points)
    
    # Ensure memory is normalized
    self.normalize_memories()
    
    # Encode points
    encoded_points = self.point_encoder.encode_points_batch(points)
    
    # Calculate similarities only for needed quadrants
    unique_quadrants = torch.unique(quadrant_indices)
    for q_idx in unique_quadrants:
        # Find points in this quadrant
        mask = quadrant_indices == q_idx
        
        if not torch.any(mask):
            continue
        
        # Get encoded points for this quadrant
        q_points = encoded_points[mask]
        
        # Calculate similarities
        if torch.norm(self.occupied_memory[q_idx]) > 0:
            occ_sim = torch.sum(q_points * self.occupied_memory[q_idx], dim=1)
        else:
            occ_sim = torch.zeros(q_points.shape[0], device=self.device)
            
        if torch.norm(self.empty_memory[q_idx]) > 0:
            emp_sim = torch.sum(q_points * self.empty_memory[q_idx], dim=1)
        else:
            emp_sim = torch.zeros(q_points.shape[0], device=self.device)
        
        # Get grid indices
        flat_indices = torch.where(mask)[0]
        grid_indices = torch.tensor(np.unravel_index(
            flat_indices.cpu().numpy(), 
            (len(x_coords), len(y_coords))
        ))
        
        # Update grids
        occupied_grid[grid_indices[0], grid_indices[1]] = occ_sim
        empty_grid[grid_indices[0], grid_indices[1]] = emp_sim
    
    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'occupied': occupied_grid,
        'empty': empty_grid
    }
```

### 9. Implement Selective Cache Updates

```python
def update_with_points(self, points, labels):
    """Update memory with points and track updated quadrants"""
    # Get quadrant indices
    quadrant_indices = self.get_quadrant_index(points)
    
    # Track which quadrants are updated
    self.updated_quadrants.update(quadrant_indices.cpu().numpy())
    
    # Standard update logic...
    
def _ensure_full_cache_calculation(self):
    """Ensure entire cache is calculated"""
    # Get grid coordinates
    x_coords, y_coords = self.grid_cache.get_grid_coordinates()
    
    # Check if caches are initialized
    if self.grid_cache.occupied_cache is None:
        self.grid_cache.occupied_cache = torch.zeros(
            (len(x_coords), len(y_coords)), 
            device=self.device
        )
    
    if self.grid_cache.empty_cache is None:
        self.grid_cache.empty_cache = torch.zeros(
            (len(x_coords), len(y_coords)), 
            device=self.device
        )
    
    # Get points that need calculation
    grid_points = []
    grid_indices = []
    
    # Find points that need calculation in each cache
    for cache_type in ['occupied', 'empty']:
        cache = self.grid_cache.occupied_cache if cache_type == 'occupied' else self.grid_cache.empty_cache
        
        # Create meshgrid for this cache
        xx, yy = torch.meshgrid(
            torch.arange(cache.shape[0], device=self.device),
            torch.arange(cache.shape[1], device=self.device),
            indexing='ij'
        )
        
        # Find points that need calculation (zeros in cache)
        need_calc = cache == 0
        
        if torch.any(need_calc):
            # Get indices
            x_idx = xx[need_calc]
            y_idx = yy[need_calc]
            
            # Convert to actual coordinates
            x_coords_actual = x_coords[x_idx]
            y_coords_actual = y_coords[y_idx]
            
            # Combine into points
            points = torch.stack([x_coords_actual, y_coords_actual], dim=1)
            
            grid_points.append(points)
            grid_indices.append(torch.stack([x_idx, y_idx], dim=1))
    
    # If nothing needs calculation, return
    if not grid_points:
        return
    
    # Combine all points
    grid_points = torch.cat(grid_points, dim=0)
    grid_indices = torch.cat(grid_indices, dim=0)
    
    # Calculate for all points at once
    self._calculate_similarities_for_grid_points(grid_points, grid_indices)
```

### 10. Implement Optimized Similarity Calculation

```python
def _calculate_similarities_for_grid_points(self, points, grid_indices):
    """
    Calculate similarities for grid points and update cache.
    
    Args:
        points: Points to calculate similarities for
        grid_indices: Corresponding grid indices
    """
    # Get quadrant indices
    quadrant_indices = self.get_quadrant_index(points)
    
    # Encode points
    encoded_points = self.point_encoder.encode_points_batch(points)
    
    # Initialize result arrays
    occupied_sims = torch.zeros(points.shape[0], device=self.device)
    empty_sims = torch.zeros(points.shape[0], device=self.device)
    
    # Calculate similarities for each quadrant
    unique_quadrants = torch.unique(quadrant_indices)
    for q_idx in unique_quadrants:
        # Find points in this quadrant
        mask = quadrant_indices == q_idx
        
        if not torch.any(mask):
            continue
        
        # Calculate similarities
        if torch.norm(self.occupied_memory[q_idx]) > 0:
            q_occ_mem = self.occupied_memory[q_idx] / torch.norm(self.occupied_memory[q_idx])
            occupied_sims[mask] = torch.sum(encoded_points[mask] * q_occ_mem, dim=1)
        
        if torch.norm(self.empty_memory[q_idx]) > 0:
            q_emp_mem = self.empty_memory[q_idx] / torch.norm(self.empty_memory[q_idx])
            empty_sims[mask] = torch.sum(encoded_points[mask] * q_emp_mem, dim=1)
    
    # Split by cache type
    occ_needed = grid_indices[:, 0] < self.grid_cache.occupied_cache.shape[0]
    occ_indices = grid_indices[occ_needed]
    
    # Update caches
    for i in range(occ_indices.shape[0]):
        x_idx, y_idx = occ_indices[i]
        self.grid_cache.occupied_cache[x_idx, y_idx] = occupied_sims[i]
        
    emp_needed = grid_indices[:, 0] < self.grid_cache.empty_cache.shape[0]
    emp_indices = grid_indices[emp_needed]
    
    for i in range(emp_indices.shape[0]):
        x_idx, y_idx = emp_indices[i]
        self.grid_cache.empty_cache[x_idx, y_idx] = empty_sims[i]
```

This implementation completely rewrites the system to avoid redundant similarity calculations by implementing a spatial grid caching system. Key optimizations include:

1. Implementing a dedicated spatial grid cache to store similarity values
2. Tracking which quadrants have been updated to selectively update the cache
3. Precomputing encoded vectors for the grid
4. Implementing selective region queries and updates
5. Using direct tensor operations for cache management
6. Calculating similarities only for needed quadrants
7. Ensuring full cache calculation only when necessary
8. Optimizing similarity calculation with batched operations

These changes significantly reduce computational overhead by avoiding recalculation of similarity values for unchanged regions of the map, resulting in much faster processing times for large environments.