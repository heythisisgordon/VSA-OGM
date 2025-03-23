# Phase 6: Comprehensive Testing and Documentation

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

Phase 1: Adaptive Spatial Indexing Implementation
Phase 2: Simplified Vector Caching Implementation 
Phase 3: Enhanced VSA Mapper Implementation - Core Structure
Phase 4: Shannon Entropy Feature Extraction Implementation
Phase 5: Main Interface and CLI Updates
Phase 6: Comprehensive Testing and Documentation (Current Phase)

## Phase 6 Focus: Comprehensive Testing and Documentation

In this phase, we will focus on ensuring the enhanced VSA-OGM implementation is robust, reliable, and well-documented. This includes comprehensive testing, performance benchmarking, and detailed documentation to help users understand and leverage the new features.

### Current Implementation Analysis

The current testing and documentation are limited to basic unit tests and minimal documentation. To ensure the enhanced VSA-OGM implementation is production-ready, we need to:
- Implement comprehensive unit tests for all components
- Conduct integration tests to ensure components work together correctly
- Perform performance benchmarking to validate improvements
- Create detailed documentation, including API reference, usage examples, and performance guidelines

### Implementation Plan

1. **Comprehensive Unit Tests**

```python
# tests/test_enhanced.py

import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import unittest

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src directory
from src.spatial import AdaptiveSpatialIndex
from src.cache import VectorCache
from src.enhanced_mapper import EnhancedVSAMapper
from src.main import pointcloud_to_ogm
from src.io import load_pointcloud, save_occupancy_grid, convert_occupancy_grid_to_pointcloud
from src.utils import visualize_occupancy_grid, visualize_class_grid
from src.functional import SSPGenerator, bind, bind_batch, power, invert

class TestAdaptiveSpatialIndex(unittest.TestCase):
    """Test cases for the AdaptiveSpatialIndex class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a small test point cloud
        self.points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])
        
        self.labels = torch.tensor([0, 1, 0, 1, 0])
        self.device = torch.device("cpu")
    
    def test_initialization(self):
        """Test initialization of AdaptiveSpatialIndex."""
        # Create spatial index
        min_resolution = 0.1
        max_resolution = 0.5
        
        spatial_index = AdaptiveSpatialIndex(
            self.points,
            self.labels,
            min_resolution,
            max_resolution,
            self.device
        )
        
        # Check cell size is within bounds
        self.assertGreaterEqual(spatial_index.cell_size, min_resolution)
        self.assertLessEqual(spatial_index.cell_size, max_resolution)
        
        # Check grid is initialized
        self.assertIsNotNone(spatial_index.grid)
        self.assertGreater(len(spatial_index.grid), 0)
    
    def test_query_range(self):
        """Test range query functionality."""
        # Create spatial index
        spatial_index = AdaptiveSpatialIndex(
            self.points,
            self.labels,
            0.1,
            0.5,
            self.device
        )
        
        # Test with large radius (should return all points)
        center = torch.tensor([0.5, 0.5], device=self.device)
        radius = 1.0
        
        query_points, query_labels = spatial_index.query_range(center, radius)
        
        self.assertEqual(query_points.shape[0], 5)
        
        # Test with small radius (should return only center point)
        radius = 0.1
        
        query_points, query_labels = spatial_index.query_range(center, radius)
        
        self.assertEqual(query_points.shape[0], 1)
        
        # Test with empty result
        center = torch.tensor([10.0, 10.0], device=self.device)
        radius = 1.0
        
        query_points, query_labels = spatial_index.query_range(center, radius)
        
        self.assertEqual(query_points.shape[0], 0)
    
    def test_is_region_free(self):
        """Test region safety check functionality."""
        # Create spatial index
        spatial_index = AdaptiveSpatialIndex(
            self.points,
            self.labels,
            0.1,
            0.5,
            self.device
        )
        
        # Test region with no occupied points
        bounds = [0.0, 0.1, 0.0, 0.1]
        safety_margin = 0.1
        
        self.assertTrue(spatial_index.is_region_free(bounds, safety_margin))
        
        # Test region with occupied points
        bounds = [0.9, 1.1, 0.9, 1.1]
        
        self.assertFalse(spatial_index.is_region_free(bounds, safety_margin))
        
        # Test region with occupied points outside safety margin
        bounds = [0.9, 1.1, 0.9, 1.1]
        safety_margin = 0.0
        
        self.assertFalse(spatial_index.is_region_free(bounds, safety_margin))
        
        # Test region with occupied points within safety margin
        bounds = [0.7, 0.9, 0.7, 0.9]
        safety_margin = 0.2
        
        self.assertFalse(spatial_index.is_region_free(bounds, safety_margin))

class TestVectorCache(unittest.TestCase):
    """Test cases for the VectorCache class."""
    
    def setUp(self):
        """Set up test data."""
        # Create random axis vectors for testing
        self.device = torch.device("cpu")
        self.vsa_dimensions = 1000
        self.length_scale = 2.0
        
        # Create SSP generator
        self.ssp_generator = SSPGenerator(
            dimensionality=self.vsa_dimensions,
            device=self.device,
            length_scale=self.length_scale
        )
        
        # Generate axis vectors
        self.xy_axis_vectors = self.ssp_generator.generate(2)
        
        # Create test points
        self.points = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.1],  # Should be discretized to same as [0.0, 0.0]
            [1.0, 1.0],
            [1.1, 1.1],  # Should be discretized to same as [1.0, 1.0]
            [2.0, 2.0]
        ], device=self.device)
    
    def test_initialization(self):
        """Test initialization of VectorCache."""
        # Create vector cache
        cache = VectorCache(
            self.xy_axis_vectors,
            self.length_scale,
            self.device,
            grid_resolution=0.1,
            max_size=1000
        )
        
        # Check cache is initialized
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(cache.cache_hits, 0)
        self.assertEqual(cache.cache_misses, 0)
    
    def test_batch_vectors(self):
        """Test batch vector retrieval."""
        # Create vector cache
        cache = VectorCache(
            self.xy_axis_vectors,
            self.length_scale,
            self.device,
            grid_resolution=0.1,
            max_size=1000
        )
        
        # First batch - should all be cache misses
        batch1 = cache.get_batch_vectors(self.points)
        
        # Check batch shape
        self.assertEqual(batch1.shape, (5, self.vsa_dimensions))
        
        # Check cache statistics
        stats1 = cache.get_cache_stats()
        self.assertEqual(stats1["hits"], 0)
        self.assertEqual(stats1["misses"], 5)
        self.assertLessEqual(stats1["cache_size"], 5)  # May be less due to discretization
        
        # Second batch - should have some cache hits
        batch2 = cache.get_batch_vectors(self.points)
        
        # Check cache statistics
        stats2 = cache.get_cache_stats()
        self.assertGreater(stats2["hits"], 0)
        self.assertEqual(stats2["cache_size"], stats1["cache_size"])
    
    def test_cache_management(self):
        """Test cache management functionality."""
        # Create vector cache with small max size
        cache = VectorCache(
            self.xy_axis_vectors,
            self.length_scale,
            self.device,
            grid_resolution=0.1,
            max_size=2
        )
        
        # First batch - should fill cache to max size
        batch1 = cache.get_batch_vectors(self.points)
        
        # Check cache statistics
        stats1 = cache.get_cache_stats()
        self.assertEqual(stats1["cache_size"], 2)
        
        # Manage cache size
        removed = cache.manage_cache_size()
        
        # Check cache statistics
        stats2 = cache.get_cache_stats()
        self.assertLessEqual(stats2["cache_size"], 2)
        
        # Clear cache
        cache.clear()
        
        # Check cache statistics
        stats3 = cache.get_cache_stats()
        self.assertEqual(stats3["cache_size"], 0)
        self.assertEqual(stats3["hits"], stats2["hits"])  # Should preserve hit/miss stats

class TestEnhancedVSAMapper(unittest.TestCase):
    """Test cases for the EnhancedVSAMapper class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a small test grid
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[4:7, 4:7] = 1  # Add a small square in the middle
        
        # Convert to point cloud
        self.world_bounds = [0, 1, 0, 1]
        self.resolution = 0.1
        
        self.points, self.labels = convert_occupancy_grid_to_pointcloud(
            grid, self.world_bounds, self.resolution
        )
        
        # Create mapper configuration
        self.config = {
            "world_bounds": self.world_bounds,
            "resolution": self.resolution,
            "min_cell_resolution": self.resolution * 2,
            "max_cell_resolution": self.resolution * 5,
            "vsa_dimensions": 1000,  # Use smaller dimensions for testing
            "length_scale": 2.0,
            "decision_thresholds": [-0.99, 0.99],
            "verbose": False,
            "batch_size": 10,
            "cache_size": 100,
            "memory_threshold": 0.8
        }
        
        # Use CPU for testing to ensure compatibility
        self.device = torch.device("cpu")
    
    def test_initialization(self):
        """Test initialization of EnhancedVSAMapper."""
        # Create mapper
        mapper = EnhancedVSAMapper(self.config, device=self.device)
        
        # Check grid dimensions
        self.assertEqual(mapper.grid_width, 10)
        self.assertEqual(mapper.grid_height, 10)
        
        # Check grids are initialized
        self.assertEqual(mapper.occupied_grid.shape, (10, 10))
        self.assertEqual(mapper.empty_grid.shape, (10, 10))
        self.assertEqual(mapper.class_grid.shape, (10, 10))
        
        # Check vector cache is initialized
        self.assertIsNotNone(mapper.vector_cache)
        
        # Check statistics are initialized
        self.assertIsNotNone(mapper.stats)
        self.assertIn("init_time", mapper.stats)
    
    def test_process_observation(self):
        """Test processing of point cloud observation."""
        # Create mapper
        mapper = EnhancedVSAMapper(self.config, device=self.device)
        
        # Process observation
        mapper.process_observation(self.points, self.labels)
        
        # Check spatial index is initialized
        self.assertIsNotNone(mapper.spatial_index)
        
        # Check grids are updated
        self.assertGreater(torch.sum(mapper.occupied_grid), 0)
        self.assertGreater(torch.sum(mapper.empty_grid), 0)
        
        # Check statistics are updated
        self.assertGreater(mapper.stats["process_time"], 0)
        self.assertGreater(mapper.stats["total_points_processed"], 0)
    
    def test_incremental_processing(self):
        """Test incremental processing functionality."""
        # Create mapper
        mapper = EnhancedVSAMapper(self.config, device=self.device)
        
        # Process observation
        mapper.process_observation(self.points, self.labels)
        
        # Process incrementally
        mapper.process_incrementally(
            horizon_distance=0.5,
            sample_resolution=0.2,
            max_samples=5,
            safety_margin=0.1
        )
        
        # Check grids are updated
        self.assertGreater(torch.sum(mapper.occupied_grid), 0)
        self.assertGreater(torch.sum(mapper.empty_grid), 0)
        
        # Check statistics are updated
        self.assertGreater(mapper.stats["incremental_time"], 0)
        self.assertGreater(mapper.stats["total_samples_processed"], 0)
    
    def test_get_grids(self):
        """Test grid retrieval functionality."""
        # Create mapper
        mapper = EnhancedVSAMapper(self.config, device=self.device)
        
        # Process observation
        mapper.process_observation(self.points, self.labels)
        
        # Get occupancy grid
        occupancy_grid = mapper.get_occupancy_grid()
        
        # Check grid shape
        self.assertEqual(occupancy_grid.shape, (10, 10))
        
        # Get empty grid
        empty_grid = mapper.get_empty_grid()
        
        # Check grid shape
        self.assertEqual(empty_grid.shape, (10, 10))
        
        # Get class grid
        class_grid = mapper.get_class_grid()
        
        # Check grid shape
        self.assertEqual(class_grid.shape, (10, 10))
    
    def test_get_stats(self):
        """Test statistics retrieval functionality."""
        # Create mapper
        mapper = EnhancedVSAMapper(self.config, device=self.device)
        
        # Process observation
        mapper.process_observation(self.points, self.labels)
        
        # Get statistics
        stats = mapper.get_stats()
        
        # Check statistics
        self.assertIn("total_time", stats)
        self.assertIn("process_time", stats)
        self.assertIn("total_points_processed", stats)
        self.assertIn("cache_hit_rate", stats)

class TestMainInterface(unittest.TestCase):
    """Test cases for the main interface."""
    
    def setUp(self):
        """Set up test data."""
        # Create a small test grid
        self.grid = np.zeros((10, 10), dtype=np.int32)
        self.grid[4:7, 4:7] = 1  # Add a small square in the middle
        
        # Convert to point cloud
        self.world_bounds = [0, 1, 0, 1]
        self.resolution = 0.1
        
        # Create temporary files
        self.input_file = "test_input.npy"
        self.original_output = "test_original.npz"
        self.enhanced_output = "test_enhanced.npz"
        
        # Save test point cloud
        points, labels = convert_occupancy_grid_to_pointcloud(
            self.grid, self.world_bounds, self.resolution
        )
        points_np = points.cpu().numpy()
        labels_np = labels.cpu().numpy()
        combined = np.column_stack((points_np, labels_np))
        np.save(self.input_file, combined)
    
    def tearDown(self):
        """Clean up temporary files."""
        for file in [self.input_file, self.original_output, self.enhanced_output]:
            if os.path.exists(file):
                os.remove(file)
    
    def test_original_mapper(self):
        """Test main interface with original mapper."""
        # Process with original mapper
        result = pointcloud_to_ogm(
            input_file=self.input_file,
            output_file=self.original_output,
            world_bounds=self.world_bounds,
            resolution=self.resolution,
            vsa_dimensions=1000,  # Small for testing
            use_cuda=False,  # Use CPU for testing
            verbose=False,
            incremental=True,
            horizon_distance=0.5,
            sample_resolution=0.2,
            max_samples=5,
            use_enhanced_mapper=False
        )
        
        # Check result
        self.assertIn("grid", result)
        self.assertIn("metadata", result)
        self.assertIn("processing_time", result)
        
        # Check metadata
        self.assertEqual(result["metadata"]["use_enhanced_mapper"], False)
        
        # Check output file exists
        self.assertTrue(os.path.exists(self.original_output))
        
        # Load output file
        data = np.load(self.original_output)
        
        # Check data
        self.assertIn("grid", data)
    
    def test_enhanced_mapper(self):
        """Test main interface with enhanced mapper."""
        # Process with enhanced mapper
        result = pointcloud_to_ogm(
            input_file=self.input_file,
            output_file=self.enhanced_output,
            world_bounds=self.world_bounds,
            resolution=self.resolution,
            vsa_dimensions=1000,  # Small for testing
            use_cuda=False,  # Use CPU for testing
            verbose=False,
            incremental=True,
            horizon_distance=0.5,
            sample_resolution=0.2,
            max_samples=5,
            safety_margin=0.1,
            use_enhanced_mapper=True,
            save_stats=True,
            visualize=True
        )
        
        # Check result
        self.assertIn("grid", result)
        self.assertIn("metadata", result)
        self.assertIn("processing_time", result)
        self.assertIn("stats", result)
        
        # Check metadata
        self.assertEqual(result["metadata"]["use_enhanced_mapper"], True)
        
        # Check output file exists
        self.assertTrue(os.path.exists(self.enhanced_output))
        
        # Check visualization files exist
        self.assertTrue(os.path.exists(os.path.splitext(self.enhanced_output)[0] + "_visualization.png"))
        self.assertTrue(os.path.exists(os.path.splitext(self.enhanced_output)[0] + "_class.png"))
        
        # Load output file
        data = np.load(self.enhanced_output)
        
        # Check data
        self.assertIn("grid", data)
        self.assertIn("use_enhanced_mapper", data)
        self.assertTrue(data["use_enhanced_mapper"])
        
        # Clean up visualization files
        for file in [os.path.splitext(self.enhanced_output)[0] + "_visualization.png", 
                    os.path.splitext(self.enhanced_output)[0] + "_class.png"]:
            if os.path.exists(file):
                os.remove(file)

def run_all_tests():
    """Run all tests."""
    unittest.main()

if __name__ == "__main__":
    run_all_tests()
```

2. **Performance Benchmarking**

```python
# tests/benchmark.py

import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src directory
from src.mapper import VSAMapper
from src.enhanced_mapper import EnhancedVSAMapper
from src.main import pointcloud_to_ogm
from src.io import load_pointcloud, save_occupancy_grid

def benchmark_mappers(
    input_file: str,
    world_bounds: list,
    resolution: float = 0.1,
    vsa_dimensions: int = 8000,
    incremental: bool = True,
    horizon_distance: float = 10.0,
    sample_resolution: float = 1.0,
    max_samples: int = 100,
    batch_size: int = 1000,
    cache_size: int = 10000,
    memory_threshold: float = 0.8,
    safety_margin: float = 0.5,
    verbose: bool = True
):
    """
    Benchmark the original and enhanced mappers.
    
    Args:
        input_file: Path to input point cloud file
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
        incremental: Whether to use incremental processing
        horizon_distance: Maximum distance from sample point to consider points
        sample_resolution: Resolution for sampling grid
        max_samples: Maximum number of sample positions to process
        batch_size: Batch size for processing points
        cache_size: Maximum size of vector cache
        memory_threshold: Threshold for GPU memory usage
        safety_margin: Minimum distance from occupied points for sampling
        verbose: Whether to print verbose output
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping benchmark.")
        return
    
    # Set device
    device = torch.device("cuda")
    
    # Load point cloud
    points, labels = load_pointcloud(input_file, device)
    
    if verbose:
        print(f"Loaded point cloud with {points.shape[0]} points")
    
    # Common configuration
    common_config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "vsa_dimensions": vsa_dimensions,
        "length_scale": 2.0,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose,
        "batch_size": batch_size
    }
    
    # Original mapper configuration
    original_config = common_config.copy()
    
    # Enhanced mapper configuration
    enhanced_config = common_config.copy()
    enhanced_config.update({
        "min_cell_resolution": resolution * 5,
        "max_cell_resolution": resolution * 20,
        "cache_size": cache_size,
        "memory_threshold": memory_threshold
    })
    
    # Benchmark results
    results = {
        "original": {},
        "enhanced": {}
    }
    
    # Benchmark original mapper
    if verbose:
        print("\n=== Benchmarking Original VSAMapper ===\n")
    
    # Create original mapper
    start_time = time.time()
    original_mapper = VSAMapper(original_config, device=device)
    original_init_time = time.time() - start_time
    
    if verbose:
        print(f"Original initialization time: {original_init_time:.4f} seconds")
    
    results["original"]["init_time"] = original_init_time
    
    # Process observation
    start_time = time.time()
    original_mapper.process_observation(points, labels)
    original_process_time = time.time() - start_time
    
    if verbose:
        print(f"Original process time: {original_process_time:.4f} seconds")
    
    results["original"]["process_time"] = original_process_time
    
    # Process incrementally if requested
    if incremental:
        start_time = time.time()
        original_mapper.process_incrementally(
            horizon_distance=horizon_distance,
            sample_resolution=sample_resolution,
            max_samples=max_samples
        )
        original_incremental_time = time.time() - start_time
        
        if verbose:
            print(f"Original incremental time: {original_incremental_time:.4f} seconds")
        
        results["original"]["incremental_time"] = original_incremental_time
    
    # Get original grid
    original_grid = original_mapper.get_occupancy_grid()
    
    # Calculate total time
    original_total_time = original_init_time + original_process_time
    if incremental:
        original_total_time += original_incremental_time
    
    results["original"]["total_time"] = original_total_time
    
    if verbose:
        print(f"Original total time: {original_total_time:.4f} seconds")
    
    # Clear GPU memory
    del original_mapper
    torch.cuda.empty_cache()
    
    # Benchmark enhanced mapper
    if verbose:
        print("\n=== Benchmarking Enhanced VSAMapper ===\n")
    
    # Create enhanced mapper
    start_time = time.time()
    enhanced_mapper = EnhancedVSAMapper(enhanced_config, device=device)
    enhanced_init_time = time.time() - start_time
    
    if verbose:
        print(f"Enhanced initialization time: {enhanced_init_time:.4f} seconds")
    
    results["enhanced"]["init_time"] = enhanced_init_time
    
    # Process observation
    start_time = time.time()
    enhanced_mapper.process_observation(points, labels)
    enhanced_process_time = time.time() - start_time
    
    if verbose:
        print(f"Enhanced process time: {enhanced_process_time:.4f} seconds")
    
    results["enhanced"]["process_time"] = enhanced_process_time
    
    # Process incrementally if requested
    if incremental:
        start_time = time.time()
        enhanced_mapper.process_incrementally(
            horizon_distance=horizon_distance,
            sample_resolution=sample_resolution,
            max_samples=max_samples,
            safety_margin=safety_margin
        )
        enhanced_incremental_time = time.time() - start_time
        
        if verbose:
            print(f"Enhanced incremental time: {enhanced_incremental_time:.4f} seconds")
        
        results["enhanced"]["incremental_time"] = enhanced_incremental_time
    
    # Get enhanced grid
    enhanced_grid = enhanced_mapper.get_occupancy_grid()
    
    # Get enhanced mapper statistics
    enhanced_stats = enhanced_mapper.get_stats()
    results["enhanced"]["stats"] = enhanced_stats
    
    # Calculate total time
    enhanced_total_time = enhanced_stats["total_time"]
    results["enhanced"]["total_time"] = enhanced_total_time
    
    if verbose:
        print(f"Enhanced total time: {enhanced_total_time:.4f} seconds")
        print(f"Enhanced points per second: {enhanced_stats['points_per_second']:.2f}")
        print(f"Enhanced cache hit rate: {enhanced_stats['cache_hit_rate']*100:.2f}%")
    
    # Compare results
    if verbose:
        print("\n=== Performance Comparison ===\n")
        print(f"Original total time: {original_total_time:.4f} seconds")
        print(f"Enhanced total time: {enhanced_total_time:.4f} seconds")
    
    # Calculate speedup
    speedup = original_total_time / enhanced_total_time
    results["speedup"] = speedup
    
    if verbose:
        if speedup > 1:
            print(f"Enhanced mapper is {speedup:.2f}x faster than original mapper")
        else:
            print(f"Enhanced mapper is {1/speedup:.2f}x slower than original mapper")
    
    # Compare memory usage
    if "current_memory_gb" in enhanced_stats:
        if verbose:
            print(f"Enhanced memory usage: {enhanced_stats['current_memory_gb']:.2f} GB / {enhanced_stats['max_memory_gb']:.2f} GB")
        
        results["enhanced"]["memory_usage"] = enhanced_stats['current_memory_gb']
        results["enhanced"]["max_memory"] = enhanced_stats['max_memory_gb']
    
    # Compare grid similarity
    grid_diff = torch.abs(original_grid - enhanced_grid)
    grid_similarity = 1.0 - torch.mean(grid_diff).item()
    results["grid_similarity"] = grid_similarity
    
    if verbose:
        print(f"Grid similarity: {grid_similarity:.4f}")
    
    # Clear GPU memory
    del enhanced_mapper
    torch.cuda.empty_cache()
    
    return results

def run_benchmarks():
    """Run benchmarks with different parameters."""
    # Input file
    input_file = "inputs/obstacle_map.npy"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping benchmarks.")
        return
    
    # World bounds
    world_bounds = [-50, 50, -50, 50]
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run benchmark with default parameters
    print("\n=== Running Benchmark with Default Parameters ===\n")
    
    results = benchmark_mappers(
        input_file=input_file,
        world_bounds=world_bounds,
        resolution=0.1,
        vsa_dimensions=8000,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        batch_size=1000,
        cache_size=10000,
        memory_threshold=0.8,
        safety_margin=0.5,
        verbose=True
    )
    
    # Save results
    if results:
        np.save("outputs/benchmark_results.npy", results)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot timing comparison
        plt.subplot(1, 2, 1)
        
        original_times = [
            results["original"]["init_time"],
            results["original"]["process_time"],
            results["original"].get("incremental_time", 0),
            results["original"]["total_time"]
        ]
        
        enhanced_times = [
            results["enhanced"]["init_time"],
            results["enhanced"]["process_time"],
            results["enhanced"].get("incremental_time", 0),
            results["enhanced"]["total_time"]
        ]
        
        labels = ["Init", "Process", "Incremental", "Total"]
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, original_times, width, label='Original')
        plt.bar(x + width/2, enhanced_times
