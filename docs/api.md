# VSA-OGM API Reference

This document provides a comprehensive reference for the VSA-OGM API.

## Main Interface

### pointcloud_to_ogm

```python
def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    vsa_dimensions: int = 16000,
    use_cuda: bool = True,
    verbose: bool = False,
    incremental: bool = False,
    horizon_distance: float = 10.0,
    sample_resolution: Optional[float] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    length_scale: float = 2.0,
    min_cell_resolution: Optional[float] = None,
    max_cell_resolution: Optional[float] = None,
    cache_size: int = 10000,
    memory_threshold: float = 0.8,
    safety_margin: float = 0.5,
    occupied_disk_radius: int = 2,
    empty_disk_radius: int = 4,
    save_entropy_grids: bool = False,
    save_stats: bool = False,
    visualize: bool = False
) -> Dict[str, Any]
```

Convert a point cloud to an occupancy grid map.

**Parameters:**

- `input_file`: Path to input point cloud (.npy file)
- `output_file`: Path to save output occupancy grid (.npz file)
- `world_bounds`: World bounds [x_min, x_max, y_min, y_max]
- `resolution`: Grid resolution in meters
- `vsa_dimensions`: Dimensionality of VSA vectors
- `use_cuda`: Whether to use CUDA if available
- `verbose`: Whether to print verbose output
- `incremental`: Whether to use incremental processing
- `horizon_distance`: Maximum distance from sample point to consider points
- `sample_resolution`: Resolution for sampling grid (default: 10x resolution)
- `max_samples`: Maximum number of sample positions to process
- `batch_size`: Batch size for processing points
- `length_scale`: Length scale for power operation
- `min_cell_resolution`: Minimum resolution for spatial indexing (default: 5x resolution)
- `max_cell_resolution`: Maximum resolution for spatial indexing (default: 20x resolution)
- `cache_size`: Maximum size of vector cache
- `memory_threshold`: Threshold for GPU memory usage (0.0-1.0)
- `safety_margin`: Minimum distance from occupied points for sampling
- `occupied_disk_radius`: Radius for occupied disk filter in entropy calculation
- `empty_disk_radius`: Radius for empty disk filter in entropy calculation
- `save_entropy_grids`: Whether to save entropy grids
- `save_stats`: Whether to save processing statistics
- `visualize`: Whether to visualize results

**Returns:**

A dictionary with the following keys:
- `grid`: The occupancy grid
- `class_grid`: The class grid (1 for occupied, -1 for empty, 0 for unknown)
- `occupied_entropy`: The occupied entropy grid
- `empty_entropy`: The empty entropy grid
- `global_entropy`: The global entropy grid
- `metadata`: Metadata about the processing
- `processing_time`: Total processing time in seconds
- `stats`: Processing statistics

## VSAMapper Class

```python
class VSAMapper:
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = None
    )
```

The VSAMapper class is the core of the VSA-OGM implementation.

**Parameters:**

- `config`: Configuration dictionary with the following keys:
  - `world_bounds`: World bounds [x_min, x_max, y_min, y_max]
  - `resolution`: Grid resolution in meters
  - `min_cell_resolution`: Minimum resolution for spatial indexing
  - `max_cell_resolution`: Maximum resolution for spatial indexing
  - `vsa_dimensions`: Dimensionality of VSA vectors
  - `length_scale`: Length scale for power operation
  - `decision_thresholds`: Thresholds for classification [lower, upper]
  - `verbose`: Whether to print verbose output
  - `batch_size`: Batch size for processing points
  - `cache_size`: Maximum size of vector cache
  - `memory_threshold`: Threshold for GPU memory usage (0.0-1.0)
  - `occupied_disk_radius`: Radius for occupied disk filter in entropy calculation
  - `empty_disk_radius`: Radius for empty disk filter in entropy calculation
- `device`: PyTorch device to use (default: CUDA if available, otherwise CPU)

### Methods

#### process_observation

```python
def process_observation(
    self,
    points: torch.Tensor,
    labels: torch.Tensor
) -> None
```

Process a point cloud observation.

**Parameters:**

- `points`: Point cloud as a tensor of shape (N, 2)
- `labels`: Labels as a tensor of shape (N,) with values 0 (empty) or 1 (occupied)

#### process_incrementally

```python
def process_incrementally(
    self,
    horizon_distance: float = 10.0,
    sample_resolution: Optional[float] = None,
    max_samples: Optional[int] = None,
    safety_margin: float = 0.5
) -> None
```

Process the point cloud incrementally.

**Parameters:**

- `horizon_distance`: Maximum distance from sample point to consider points
- `sample_resolution`: Resolution for sampling grid (default: 10x resolution)
- `max_samples`: Maximum number of sample positions to process
- `safety_margin`: Minimum distance from occupied points for sampling

#### get_occupancy_grid

```python
def get_occupancy_grid(self) -> torch.Tensor
```

Get the occupancy grid.

**Returns:**

A tensor of shape (H, W) with values in the range [-1, 1].

#### get_class_grid

```python
def get_class_grid(self) -> torch.Tensor
```

Get the class grid.

**Returns:**

A tensor of shape (H, W) with values -1 (empty), 0 (unknown), or 1 (occupied).

#### get_occupied_entropy_grid

```python
def get_occupied_entropy_grid(self) -> torch.Tensor
```

Get the occupied entropy grid.

**Returns:**

A tensor of shape (H, W) with values in the range [0, 1].

#### get_empty_entropy_grid

```python
def get_empty_entropy_grid(self) -> torch.Tensor
```

Get the empty entropy grid.

**Returns:**

A tensor of shape (H, W) with values in the range [0, 1].

#### get_global_entropy_grid

```python
def get_global_entropy_grid(self) -> torch.Tensor
```

Get the global entropy grid.

**Returns:**

A tensor of shape (H, W) with values in the range [-1, 1].

#### get_stats

```python
def get_stats(self) -> Dict[str, Any]
```

Get processing statistics.

**Returns:**

A dictionary with the following keys:
- `total_time`: Total processing time in seconds
- `init_time`: Initialization time in seconds
- `process_time`: Process observation time in seconds
- `incremental_time`: Incremental processing time in seconds (if applicable)
- `total_points_processed`: Total number of points processed
- `total_samples_processed`: Total number of samples processed (if incremental)
- `points_per_second`: Points processed per second
- `cache_hit_rate`: Cache hit rate (0.0-1.0)
- `current_memory_gb`: Current GPU memory usage in GB (if CUDA)
- `max_memory_gb`: Maximum GPU memory available in GB (if CUDA)

## AdaptiveSpatialIndex Class

```python
class AdaptiveSpatialIndex:
    def __init__(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        min_resolution: float,
        max_resolution: float,
        device: torch.device = None
    )
```

The AdaptiveSpatialIndex class provides efficient spatial indexing for point clouds.

**Parameters:**

- `points`: Point cloud as a tensor of shape (N, 2)
- `labels`: Labels as a tensor of shape (N,) with values 0 (empty) or 1 (occupied)
- `min_resolution`: Minimum cell resolution
- `max_resolution`: Maximum cell resolution
- `device`: PyTorch device to use (default: same as points)

### Methods

#### query_range

```python
def query_range(
    self,
    center: torch.Tensor,
    radius: float
) -> Tuple[torch.Tensor, torch.Tensor]
```

Query points within a range.

**Parameters:**

- `center`: Center point as a tensor of shape (2,)
- `radius`: Radius in meters

**Returns:**

A tuple of (points, labels) within the range.

#### is_region_free

```python
def is_region_free(
    self,
    bounds: List[float],
    safety_margin: float = 0.0
) -> bool
```

Check if a region is free of occupied points.

**Parameters:**

- `bounds`: Region bounds [x_min, x_max, y_min, y_max]
- `safety_margin`: Safety margin in meters

**Returns:**

True if the region is free, False otherwise.

## VectorCache Class

```python
class VectorCache:
    def __init__(
        self,
        axis_vectors: torch.Tensor,
        length_scale: float,
        device: torch.device = None,
        grid_resolution: float = 0.1,
        max_size: int = 10000
    )
```

The VectorCache class provides caching for VSA vectors.

**Parameters:**

- `axis_vectors`: Axis vectors as a tensor of shape (2, D)
- `length_scale`: Length scale for power operation
- `device`: PyTorch device to use (default: same as axis_vectors)
- `grid_resolution`: Resolution for discretizing points
- `max_size`: Maximum cache size

### Methods

#### get_batch_vectors

```python
def get_batch_vectors(
    self,
    points: torch.Tensor
) -> torch.Tensor
```

Get VSA vectors for a batch of points.

**Parameters:**

- `points`: Points as a tensor of shape (N, 2)

**Returns:**

VSA vectors as a tensor of shape (N, D).

#### precompute_grid_vectors

```python
def precompute_grid_vectors(
    self,
    world_bounds: List[float],
    grid_resolution: float
) -> None
```

Precompute VSA vectors for a grid.

**Parameters:**

- `world_bounds`: World bounds [x_min, x_max, y_min, y_max]
- `grid_resolution`: Grid resolution in meters

#### clear

```python
def clear(self) -> None
```

Clear the cache.

#### get_cache_stats

```python
def get_cache_stats(self) -> Dict[str, Any]
```

Get cache statistics.

**Returns:**

A dictionary with the following keys:
- `cache_size`: Current cache size
- `hits`: Number of cache hits
- `misses`: Number of cache misses
- `hit_rate`: Cache hit rate (0.0-1.0)

## Utility Functions

### visualize_occupancy_grid

```python
def visualize_occupancy_grid(
    grid: Union[np.ndarray, torch.Tensor],
    output_file: Optional[str] = None,
    world_bounds: Optional[List[float]] = None,
    colormap: str = 'viridis',
    show: bool = False
) -> None
```

Visualize an occupancy grid.

**Parameters:**

- `grid`: Occupancy grid as a numpy array or torch tensor
- `output_file`: Path to save the visualization (optional)
- `world_bounds`: World bounds [x_min, x_max, y_min, y_max] (optional)
- `colormap`: Matplotlib colormap to use
- `show`: Whether to show the plot

### visualize_class_grid

```python
def visualize_class_grid(
    grid: Union[np.ndarray, torch.Tensor],
    output_file: Optional[str] = None,
    world_bounds: Optional[List[float]] = None,
    show: bool = False
) -> None
```

Visualize a class grid.

**Parameters:**

- `grid`: Class grid as a numpy array or torch tensor
- `output_file`: Path to save the visualization (optional)
- `world_bounds`: World bounds [x_min, x_max, y_min, y_max] (optional)
- `show`: Whether to show the plot

### visualize_entropy_grid

```python
def visualize_entropy_grid(
    grid: Union[np.ndarray, torch.Tensor],
    output_file: Optional[str] = None,
    world_bounds: Optional[List[float]] = None,
    colormap: str = 'plasma',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = False
) -> None
```

Visualize an entropy grid.

**Parameters:**

- `grid`: Entropy grid as a numpy array or torch tensor
- `output_file`: Path to save the visualization (optional)
- `world_bounds`: World bounds [x_min, x_max, y_min, y_max] (optional)
- `colormap`: Matplotlib colormap to use
- `vmin`: Minimum value for colormap scaling (optional)
- `vmax`: Maximum value for colormap scaling (optional)
- `show`: Whether to show the plot

### visualize_entropy_comparison

```python
def visualize_entropy_comparison(
    occupied_entropy: Union[np.ndarray, torch.Tensor],
    empty_entropy: Union[np.ndarray, torch.Tensor],
    global_entropy: Union[np.ndarray, torch.Tensor],
    output_file: Optional[str] = None,
    world_bounds: Optional[List[float]] = None,
    show: bool = False
) -> None
```

Visualize a comparison of occupied, empty, and global entropy grids.

**Parameters:**

- `occupied_entropy`: Occupied entropy grid
- `empty_entropy`: Empty entropy grid
- `global_entropy`: Global entropy grid
- `output_file`: Path to save the visualization (optional)
- `world_bounds`: World bounds [x_min, x_max, y_min, y_max] (optional)
- `show`: Whether to show the plot

## I/O Functions

### load_pointcloud

```python
def load_pointcloud(
    file_path: str,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Load a point cloud from a file.

**Parameters:**

- `file_path`: Path to the point cloud file (.npy)
- `device`: PyTorch device to use (default: CPU)

**Returns:**

A tuple of (points, labels).

### save_occupancy_grid

```python
def save_occupancy_grid(
    grid: torch.Tensor,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

Save an occupancy grid to a file.

**Parameters:**

- `grid`: Occupancy grid as a tensor
- `file_path`: Path to save the occupancy grid (.npz)
- `metadata`: Metadata to save with the grid (optional)

### convert_occupancy_grid_to_pointcloud

```python
def convert_occupancy_grid_to_pointcloud(
    grid: np.ndarray,
    world_bounds: List[float],
    resolution: float
) -> Tuple[torch.Tensor, torch.Tensor]
```

Convert an occupancy grid to a point cloud.

**Parameters:**

- `grid`: Occupancy grid as a numpy array
- `world_bounds`: World bounds [x_min, x_max, y_min, y_max]
- `resolution`: Grid resolution in meters

**Returns:**

A tuple of (points, labels).
