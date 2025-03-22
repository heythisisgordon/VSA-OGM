# Phase 4: Shannon Entropy Feature Extraction Implementation

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

1. Efficient spatial indexing with adaptive cell sizing (Phase 1)
2. Optimized vector computation with parallel processing and caching (Phase 2)
3. Memory-aware processing with GPU memory monitoring (Phase 3)
4. Shannon entropy feature extraction (Phase 4 - Current)
5. Enhanced class grid generation based on entropy values (Phase 4 - Current)
6. Incremental processing with horizon-limited visibility
7. Enhanced VSA mapper with direct spatial processing

## Phase 4 Focus: Shannon Entropy Feature Extraction

In this phase, we will implement the Shannon entropy-based feature extraction capability described in the original paper. This is a critical component of the VSA-OGM approach that significantly enhances the mapper's ability to extract features from noisy hyperdimensional computing (HDC) representations. The implementation will extend the basic structure of the enhanced VSA mapper from Phase 3.

### Current Implementation Analysis

The current implementation from Phase 3 uses a simple thresholding approach to classify voxels based on normalized occupancy and emptiness values. This approach doesn't fully leverage the feature extraction capabilities described in the original paper, which uses Shannon entropy to extract rich information from the HDC representations.

### Implementation Plan

1. **Shannon Entropy Utility Functions**

```python
def _create_disk_filter(self, radius: int) -> torch.Tensor:
    """
    Create a disk filter for Shannon entropy calculation.
    
    Args:
        radius: Radius of the disk filter in voxels
        
    Returns:
        Binary disk filter as a tensor
    """
    diameter = 2 * radius + 1
    center = radius
    
    # Create grid coordinates
    y, x = torch.meshgrid(
        torch.arange(diameter, device=self.device),
        torch.arange(diameter, device=self.device),
        indexing="ij"
    )
    
    # Calculate squared distance from center
    squared_distance = (x - center) ** 2 + (y - center) ** 2
    
    # Create disk filter (1 inside disk, 0 outside)
    disk = (squared_distance <= radius ** 2).float()
    
    return disk

def _shannon_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
    """
    Calculate Shannon entropy of probability values.
    
    Args:
        probabilities: Tensor of probability values (0.0-1.0)
        
    Returns:
        Entropy values
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    
    # Ensure probabilities are valid (between 0 and 1)
    probabilities = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    
    # Calculate entropy: -p*log2(p) - (1-p)*log2(1-p)
    entropy = -probabilities * torch.log2(probabilities) - (1 - probabilities) * torch.log2(1 - probabilities)
    
    return entropy

def _normalize_grid(self, grid: torch.Tensor) -> torch.Tensor:
    """
    Normalize a grid to the range [0, 1].
    
    Args:
        grid: Grid to normalize
        
    Returns:
        Normalized grid
    """
    # Get min and max values
    min_val = torch.min(grid)
    max_val = torch.max(grid)
    
    # Normalize grid
    if max_val > min_val:
        normalized_grid = (grid - min_val) / (max_val - min_val)
    else:
        normalized_grid = torch.zeros_like(grid)
    
    return normalized_grid
```

2. **Extend Initialization with Shannon Entropy Parameters**

```python
def __init__(
    self, 
    config: Dict[str, Any],
    device: Optional[torch.device] = None
):
    """
    Initialize the enhanced VSA mapper with Shannon entropy parameters.
    
    Args:
        config: Configuration dictionary with parameters:
            - ... (existing parameters)
            - occupied_disk_radius: Radius for occupied disk filter in entropy calculation
            - empty_disk_radius: Radius for empty disk filter in entropy calculation
    """
    # ... (existing initialization code)
    
    # Shannon entropy parameters
    self.occupied_disk_radius = config.get("occupied_disk_radius", 2)
    self.empty_disk_radius = config.get("empty_disk_radius", 4)
    
    # Initialize entropy grids
    self.occupied_entropy_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
    self.empty_entropy_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
    self.global_entropy_grid = torch.zeros((self.grid_height, self.grid_width), device=self.device)
    
    # ... (remaining initialization code)
    
    if self.verbose:
        print(f"Shannon entropy disk radii: occupied={self.occupied_disk_radius}, empty={self.empty_disk_radius}")
```

3. **Local Entropy Calculation**

```python
def _apply_local_entropy(self, probability_grid: torch.Tensor, disk_filter: torch.Tensor) -> torch.Tensor:
    """
    Apply local Shannon entropy calculation with disk filter.
    
    Args:
        probability_grid: Grid of probability values
        disk_filter: Disk filter to use for local entropy calculation
        
    Returns:
        Grid of local entropy values
    """
    # Get disk dimensions
    filter_height, filter_width = disk_filter.shape
    padding_y = filter_height // 2
    padding_x = filter_width // 2
    
    # Pad probability grid
    padded_grid = torch.nn.functional.pad(
        probability_grid,
        (padding_x, padding_x, padding_y, padding_y),
        mode='constant',
        value=0
    )
    
    # Initialize entropy grid
    entropy_grid = torch.zeros_like(probability_grid)
    
    # Apply disk filter to each position
    for y in range(self.grid_height):
        for x in range(self.grid_width):
            # Extract local region
            local_region = padded_grid[y:y + filter_height, x:x + filter_width]
            
            # Apply disk filter
            masked_region = local_region * disk_filter
            
            # Count non-zero elements in mask
            num_elements = torch.sum(disk_filter).item()
            
            # Calculate mean probability within disk
            if num_elements > 0:
                mean_prob = torch.sum(masked_region) / num_elements
            else:
                mean_prob = 0.0
            
            # Calculate entropy
            entropy_grid[y, x] = self._shannon_entropy(torch.tensor([mean_prob], device=self.device)).item()
    
    return entropy_grid
```

4. **Shannon Entropy Feature Extraction**

```python
def _apply_shannon_entropy(self) -> None:
    """
    Apply Shannon entropy-based feature extraction as described in the paper.
    
    This method implements the Shannon entropy approach described in the paper:
    1. Convert quasi-probabilities to true probabilities using the Born rule
    2. Apply disk filters to compute local entropy for both occupied and empty grids
    3. Calculate global entropy as the difference between occupied and empty entropy
    """
    if self.verbose:
        print("Applying Shannon entropy-based feature extraction...")
    
    # Create disk filters
    occupied_disk = self._create_disk_filter(self.occupied_disk_radius)
    empty_disk = self._create_disk_filter(self.empty_disk_radius)
    
    # Normalize occupied and empty grids to get probability maps
    occupied_prob = self._normalize_grid(self.occupied_grid)
    empty_prob = self._normalize_grid(self.empty_grid)
    
    # Apply Born rule: true probability = squared quasi-probability
    occupied_prob = occupied_prob ** 2
    empty_prob = empty_prob ** 2
    
    # Calculate local entropy for occupied and empty grids
    self.occupied_entropy_grid = self._apply_local_entropy(occupied_prob, occupied_disk)
    self.empty_entropy_grid = self._apply_local_entropy(empty_prob, empty_disk)
    
    # Calculate global entropy as the difference between occupied and empty entropy
    self.global_entropy_grid = self.occupied_entropy_grid - self.empty_entropy_grid
    
    if self.verbose:
        print("Shannon entropy-based feature extraction completed")
```

5. **Update Process Observation Method**

```python
def process_observation(
    self, 
    points: torch.Tensor, 
    labels: torch.Tensor
) -> None:
    """
    Process a point cloud observation with memory monitoring and Shannon entropy.
    
    Args:
        points: Tensor of shape [N, 2] containing point coordinates
        labels: Tensor of shape [N] containing point labels (0=empty, 1=occupied)
    """
    # ... (existing process_observation code)
    
    # Process points directly using spatial grid
    self._process_points_spatially(normalized_points, labels)
    
    # Apply Shannon entropy for feature extraction
    self._apply_shannon_entropy()
    
    # Update class grid based on entropy values
    self._update_class_grid_from_entropy()
    
    # ... (remaining process_observation code)
```

6. **Update Incremental Processing Method**

```python
def process_incrementally(
    self, 
    horizon_distance: float = 10.0, 
    sample_resolution: Optional[float] = None, 
    max_samples: Optional[int] = None,
    safety_margin: float = 0.5
) -> None:
    """
    Process the point cloud incrementally with Shannon entropy.
    """
    # ... (existing process_incrementally code)
    
    # Process each sample position
    # ... (existing processing code)
    
    # Apply Shannon entropy for feature extraction
    self._apply_shannon_entropy()
    
    # Update the class grid based on entropy values
    self._update_class_grid_from_entropy()
    
    # ... (remaining process_incrementally code)
```

7. **Entropy-Based Class Grid Update**

```python
def _update_class_grid_from_entropy(self) -> None:
    """
    Update class grid based on Shannon entropy values as described in the paper.
    
    This method implements the classification approach described in the paper,
    using the global entropy grid to classify each voxel as occupied, empty, or unknown.
    """
    # Initialize with unknown (0)
    self.class_grid = torch.zeros_like(self.global_entropy_grid)
    
    # Set occupied (1) where global entropy is above upper threshold
    self.class_grid[self.global_entropy_grid > self.decision_thresholds[1]] = 1
    
    # Set empty (-1) where global entropy is below lower threshold
    self.class_grid[self.global_entropy_grid < self.decision_thresholds[0]] = -1
    
    if self.verbose:
        occupied_count = torch.sum(self.class_grid == 1).item()
        empty_count = torch.sum(self.class_grid == -1).item()
        unknown_count = torch.sum(self.class_grid == 0).item()
        
        print(f"Class grid updated from entropy: {occupied_count} occupied, "
              f"{empty_count} empty, {unknown_count} unknown voxels")
```

8. **Additional Grid Retrieval Methods**

```python
def get_occupied_entropy_grid(self) -> torch.Tensor:
    """
    Get the occupied entropy grid.
    
    Returns:
        Tensor of shape [H, W] containing occupied entropy values
    """
    return self.occupied_entropy_grid

def get_empty_entropy_grid(self) -> torch.Tensor:
    """
    Get the empty entropy grid.
    
    Returns:
        Tensor of shape [H, W] containing empty entropy values
    """
    return self.empty_entropy_grid

def get_global_entropy_grid(self) -> torch.Tensor:
    """
    Get the global entropy grid.
    
    Returns:
        Tensor of shape [H, W] containing global entropy values
    """
    return self.global_entropy_grid
```

9. **Update Statistics Method**

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get processing statistics with entropy parameters.
    """
    # Get basic statistics
    combined_stats = super().get_stats()
    
    # Add entropy parameters
    combined_stats.update({
        "occupied_disk_radius": self.occupied_disk_radius,
        "empty_disk_radius": self.empty_disk_radius
    })
    
    return combined_stats
```

### Implementation Notes

This phase focuses on implementing the Shannon entropy-based feature extraction capability described in the original paper. The key components include:

1. **Disk Filters**: Creation of disk-shaped filters with configurable radii for both occupied and empty classes.

2. **Born Rule Conversion**: Conversion of quasi-probabilities from the VSA system to true probabilities using the Born rule (squaring the values).

3. **Local Entropy Calculation**: Computation of local Shannon entropy within disk neighborhoods for both occupied and empty probability maps.

4. **Global Entropy Map**: Calculation of the global entropy map by subtracting empty entropy from occupied entropy.

5. **Entropy-Based Classification**: Classification of voxels based on the global entropy map rather than directly using normalized grid values.

These enhancements significantly improve the mapper's ability to extract features from noisy HDC representations, particularly in sparse areas of the environment, and closely follow the approach described in the original paper.

### Next Steps

After implementing the Shannon entropy-based feature extraction, we will proceed to Phase 5, which will focus on updating the main interface and CLI to support the enhanced VSA mapper with Shannon entropy and provide access to the new features.