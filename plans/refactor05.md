# Plan to Address Suboptimal Entropy Calculation

## Problem Statement
The original implementation uses optimized disk filters with convolutions for entropy extraction, which is much faster than the naive implementation in your code. We need to completely rewrite the entropy calculation architecture.

## Implementation Plan

### 1. Rewrite the Entropy Extraction Core

```python
class EntropyExtractor:
    """
    High-performance entropy calculation for feature extraction from probability fields.
    Uses optimized convolution operations for maximum throughput.
    """
    
    def __init__(self, 
                 disk_radius: int = 3,
                 occupied_threshold: float = 0.6,
                 empty_threshold: float = 0.3,
                 device: str = "cuda") -> None:
        """Initialize the entropy extractor with optimized parameters"""
        self.disk_radius = disk_radius
        self.occupied_threshold = occupied_threshold
        self.empty_threshold = empty_threshold
        self.device = device
        
        # Create and optimize disk filter immediately
        self._create_optimized_disk_filter()
```

### 2. Implement Optimized Disk Filter Creation

```python
def _create_optimized_disk_filter(self) -> None:
    """
    Create an optimized disk filter for convolution-based entropy calculation.
    Pre-compute the filter once during initialization for maximum performance.
    """
    # Create a square grid with optimal size
    size = 2 * self.disk_radius + 1
    
    # Use linear space for precise control
    x = torch.linspace(-self.disk_radius, self.disk_radius, size, device=self.device)
    y = torch.linspace(-self.disk_radius, self.disk_radius, size, device=self.device)
    
    # Create meshgrid with explicit indexing for clarity
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Create disk mask (1 inside disk, 0 outside) using efficient tensor operations
    disk = (xx**2 + yy**2 <= self.disk_radius**2).float()
    
    # Normalize the disk filter for consistent weighting
    self.disk_filter = disk / torch.sum(disk)
    
    # Format filter for efficient 2D convolution (shape: 1, 1, height, width)
    self.conv_filter = self.disk_filter.unsqueeze(0).unsqueeze(0)
```

### 3. Implement Direct Convolution-Based Entropy Calculation

```python
def calculate_entropy(self, probability_field: torch.Tensor) -> torch.Tensor:
    """
    Calculate Shannon entropy for a probability field using optimized convolution.
    
    Args:
        probability_field: 2D tensor of probabilities in range [0, 1]
        
    Returns:
        2D tensor of entropy values
    """
    # Ensure probability field is on the correct device
    probability_field = probability_field.to(self.device)
    
    # Clamp probabilities to avoid log(0) with minimal overhead
    prob = torch.clamp(probability_field, 1e-10, 1.0 - 1e-10)
    
    # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
    # Direct formula implementation for maximum clarity and performance
    entropy = -prob * torch.log2(prob) - (1 - prob) * torch.log2(1 - prob)
    
    # Apply convolution directly with pre-formatted filter
    # Reshape probability field for 2D convolution (shape: 1, 1, height, width)
    entropy_reshaped = entropy.unsqueeze(0).unsqueeze(0)
    
    # Perform optimized convolution with appropriate padding
    padding = self.disk_radius  # Padding based on filter radius
    local_entropy = torch.nn.functional.conv2d(
        entropy_reshaped, 
        self.conv_filter,
        padding=padding
    )
    
    # Return to original shape
    return local_entropy.squeeze()
```

### 4. Implement Optimized Born Rule Application

```python
def apply_born_rule(self, similarity_scores: torch.Tensor) -> torch.Tensor:
    """
    Apply Born rule to convert similarity scores to probabilities.
    Optimized implementation with minimal operations.
    
    Args:
        similarity_scores: Tensor of similarity scores in range [-1, 1]
        
    Returns:
        Tensor of probabilities in range [0, 1]
    """
    # Normalize to [0, 1] range and apply Born rule (square)
    # Direct formula using efficient tensor operations
    normalized = (similarity_scores + 1) / 2
    return normalized ** 2
```

### 5. Implement Batch Feature Extraction

```python
def extract_features(self, 
                    occupied_probs: torch.Tensor,
                    empty_probs: torch.Tensor) -> dict:
    """
    Extract entropy features from probability fields in a single batch operation.
    
    Args:
        occupied_probs: 2D tensor of occupancy probabilities
        empty_probs: 2D tensor of emptiness probabilities
        
    Returns:
        Dictionary containing entropy and classification tensors
    """
    # Calculate entropy for occupied and empty probabilities in one pass
    occupied_entropy = self.calculate_entropy(occupied_probs)
    empty_entropy = self.calculate_entropy(empty_probs)
    
    # Calculate global entropy (occupied - empty) directly
    global_entropy = occupied_entropy - empty_entropy
    
    # Classify cells based on global entropy with efficient tensor operations
    classification = torch.zeros_like(global_entropy, dtype=torch.int)
    classification[global_entropy > self.occupied_threshold] = 1  # Occupied
    classification[global_entropy < -self.empty_threshold] = -1   # Empty
    
    # Return all results in a single dictionary
    return {
        'occupied_entropy': occupied_entropy,
        'empty_entropy': empty_entropy,
        'global_entropy': global_entropy,
        'classification': classification
    }
```

### 6. Implement Direct Classification

```python
def classify_grid(self, global_entropy: torch.Tensor) -> torch.Tensor:
    """
    Classify a grid based on global entropy values.
    Optimized implementation using direct tensor operations.
    
    Args:
        global_entropy: 2D tensor of global entropy values
        
    Returns:
        2D tensor of classifications (-1=empty, 0=unknown, 1=occupied)
    """
    # Create output tensor directly with correct size and device
    classification = torch.zeros_like(global_entropy, dtype=torch.int)
    
    # Apply classification thresholds in a single operation
    classification = torch.where(
        global_entropy > self.occupied_threshold,
        torch.ones_like(classification),
        classification
    )
    
    classification = torch.where(
        global_entropy < -self.empty_threshold,
        -torch.ones_like(classification),
        classification
    )
    
    return classification
```

### 7. Implement Direct Binary Occupancy Grid Generation

```python
def get_occupancy_grid(self, classification: torch.Tensor) -> torch.Tensor:
    """
    Convert classification to binary occupancy grid with optimized operations.
    
    Args:
        classification: 2D tensor of classifications (-1=empty, 0=unknown, 1=occupied)
        
    Returns:
        Binary occupancy grid (0=free/unknown, 1=occupied)
    """
    # Use direct tensor comparison for maximum efficiency
    return (classification == 1).int()
```

### 8. Implement Optimized Confidence Map Generation

```python
def get_confidence_map(self, global_entropy: torch.Tensor) -> torch.Tensor:
    """
    Generate a confidence map from global entropy using optimized operations.
    
    Args:
        global_entropy: 2D tensor of global entropy values
        
    Returns:
        Confidence map in range [0, 1]
    """
    # Direct calculation of confidence from absolute entropy values
    confidence = torch.abs(global_entropy)
    
    # Normalize to [0, 1] using efficient operations
    max_confidence = torch.max(confidence)
    if max_confidence > 0:
        confidence = confidence / max_confidence
    
    return confidence
```

### 9. Implement Multi-Resolution Entropy Calculation

```python
def calculate_multi_resolution_entropy(self, 
                                      probability_field: torch.Tensor, 
                                      resolutions: list) -> torch.Tensor:
    """
    Calculate entropy at multiple resolutions and combine for increased detail.
    
    Args:
        probability_field: 2D tensor of probabilities
        resolutions: List of disk filter radii to use
        
    Returns:
        Combined multi-resolution entropy map
    """
    # Store the original disk filter temporarily
    original_filter = self.conv_filter
    original_radius = self.disk_radius
    
    # Calculate entropy at each resolution
    entropy_maps = []
    for radius in resolutions:
        # Update filter for this resolution
        self.disk_radius = radius
        self._create_optimized_disk_filter()
        
        # Calculate entropy at this resolution
        entropy = self.calculate_entropy(probability_field)
        entropy_maps.append(entropy)
    
    # Restore original filter
    self.disk_radius = original_radius
    self.conv_filter = original_filter
    
    # Combine entropy maps (weighted average)
    weights = torch.tensor([1.0 / len(resolutions)] * len(resolutions), device=self.device)
    combined_entropy = torch.zeros_like(entropy_maps[0])
    
    for i, entropy in enumerate(entropy_maps):
        combined_entropy += weights[i] * entropy
    
    return combined_entropy
```

### 10. Optimize Integration with VSAMapper

```python
# Inside VSAMapper class:
def _init_components(self) -> None:
    """Initialize all components with optimized entropy extraction"""
    # Other initializations...
    
    # Initialize optimized entropy extractor
    self.entropy_extractor = EntropyExtractor(
        disk_radius=self.config.get("entropy", "disk_radius"),
        occupied_threshold=self.config.get("entropy", "occupied_threshold"),
        empty_threshold=self.config.get("entropy", "empty_threshold"),
        device=self.device
    )

def _generate_maps(self) -> None:
    """Generate all maps using optimized entropy calculation"""
    # Query grid from quadrant memory
    grid_results = self.quadrant_memory.query_grid(self.config.get("sequential", "sample_resolution"))
    
    # Convert similarity scores to probabilities using Born rule
    occupied_probs = self.entropy_extractor.apply_born_rule(grid_results['occupied'])
    empty_probs = self.entropy_extractor.apply_born_rule(grid_results['empty'])
    
    # Extract all features in a single optimized operation
    features = self.entropy_extractor.extract_features(occupied_probs, empty_probs)
    
    # Store results directly
    self.entropy_grid = features['global_entropy']
    self.classification = features['classification']
    self.occupancy_grid = self.entropy_extractor.get_occupancy_grid(self.classification)
    
    # Store grid coordinates for visualization
    self.grid_coords = {
        'x': grid_results['x_coords'],
        'y': grid_results['y_coords']
    }
```

This implementation completely rewrites the entropy calculation architecture to use optimized disk filters with convolutions, providing much faster performance than the naive approach. Key optimizations include pre-computing the disk filter during initialization, using direct tensor operations throughout, and implementing efficient convolution-based entropy calculation. The implementation also provides multi-resolution entropy calculation for increased detail in complex environments.