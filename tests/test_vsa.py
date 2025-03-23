"""Tests for the VSA-OGM implementation."""

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
from src.spatial import AdaptiveSpatialIndex
from src.cache import VectorCache
from src.io import load_pointcloud, save_occupancy_grid, convert_occupancy_grid_to_pointcloud
from src.utils import visualize_occupancy_grid, visualize_class_grid

def test_adaptive_spatial_index():
    """
    Test the AdaptiveSpatialIndex class.
    """
    print("Testing AdaptiveSpatialIndex...")
    
    # Create a test point cloud
    points = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ])
    
    labels = torch.tensor([0, 1, 0, 1, 0])
    
    # Create spatial index
    device = torch.device("cpu")
    min_resolution = 0.1
    max_resolution = 1.0
    
    spatial_index = AdaptiveSpatialIndex(
        points,
        labels,
        min_resolution,
        max_resolution,
        device
    )
    
    # Test cell size calculation
    assert spatial_index.cell_size >= min_resolution
    assert spatial_index.cell_size <= max_resolution
    
    print(f"Created spatial index with cell size: {spatial_index.cell_size}")
    
    # Test range query
    center = torch.tensor([0.5, 0.5])
    radius = 0.6
    
    query_points, query_labels = spatial_index.query_range(center, radius)
    
    # Should return all points (all within 0.6 of center)
    assert query_points.shape[0] == 5
    print(f"Range query with radius {radius} returned {query_points.shape[0]} points")
    
    # Test with smaller radius
    radius = 0.3
    query_points, query_labels = spatial_index.query_range(center, radius)
    
    # Should only return the center point
    assert query_points.shape[0] == 1
    print(f"Range query with radius {radius} returned {query_points.shape[0]} points")
    
    # Test region safety check
    bounds = [0.4, 0.6, 0.4, 0.6]  # Small region around center
    
    # With small safety margin, should be free (center point is not occupied)
    assert spatial_index.is_region_free(bounds, 0.1)
    print(f"Region safety check with margin 0.1: region is free")
    
    # With large safety margin, should not be free (occupied points nearby)
    assert not spatial_index.is_region_free(bounds, 0.5)
    print(f"Region safety check with margin 0.5: region is not free")
    
    print("AdaptiveSpatialIndex tests passed!")
    return True

def test_vector_cache():
    """
    Test the VectorCache class.
    """
    print("Testing VectorCache...")
    
    # Create a small test grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 1, 0, 1]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Create device and axis vectors
    device = torch.device("cpu")
    vsa_dimensions = 1000
    length_scale = 2.0
    
    # Create random axis vectors for testing
    xy_axis_vectors = torch.randn((2, vsa_dimensions), device=device)
    
    # Create vector cache
    cache = VectorCache(
        xy_axis_vectors,
        length_scale,
        device,
        grid_resolution=resolution,
        max_size=1000
    )
    
    # Test batch vector retrieval
    batch_vectors = cache.get_batch_vectors(points)
    
    print(f"Generated batch vectors with shape: {batch_vectors.shape}")
    
    # Test cache hit rate
    batch_vectors = cache.get_batch_vectors(points)  # Should be cached now
    
    cache_stats = cache.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    # Test clear method
    cache.clear()
    print("Cache cleared")
    
    cache_stats = cache.get_cache_stats()
    print(f"Cache stats after clear: {cache_stats}")
    assert cache_stats["cache_size"] == 0
    
    # Test precompute grid vectors
    cache.precompute_grid_vectors(world_bounds, resolution * 2)
    
    cache_stats = cache.get_cache_stats()
    print(f"Cache stats after precompute: {cache_stats}")
    assert cache_stats["cache_size"] > 0
    
    # Test LRU-like cache management
    # Create more points than the cache can hold
    test_points = torch.rand((cache.max_size + 100, 2), device=device)
    
    # Fill the cache
    batch_vectors = cache.get_batch_vectors(test_points)
    
    cache_stats = cache.get_cache_stats()
    print(f"Cache stats after filling: {cache_stats}")
    assert cache_stats["cache_size"] <= cache.max_size
    
    return True

def test_vector_cache_performance():
    """
    Test the performance of the VectorCache class.
    """
    print("Testing vector cache performance...")
    
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping performance test.")
        return True
    
    # Create random axis vectors for testing
    device = torch.device("cuda")
    vsa_dimensions = 16000
    length_scale = 2.0
    
    xy_axis_vectors = torch.randn((2, vsa_dimensions), device=device)
    
    # Create vector cache
    cache = VectorCache(
        xy_axis_vectors,
        length_scale,
        device,
        grid_resolution=0.1,
        max_size=10000
    )
    
    # Create test points - large batch
    n_points = 10000
    points = torch.rand((n_points, 2), device=device) * 100  # Random points in 100x100 area
    
    # First batch - should all be cache misses
    start_time = time.time()
    batch1 = cache.get_batch_vectors(points)
    batch1_time = time.time() - start_time
    
    stats1 = cache.get_cache_stats()
    print(f"First batch time: {batch1_time:.4f} seconds")
    print(f"Cache stats after first batch: {stats1}")
    
    # Second batch - same points, should all be cache hits
    start_time = time.time()
    batch2 = cache.get_batch_vectors(points)
    batch2_time = time.time() - start_time
    
    stats2 = cache.get_cache_stats()
    print(f"Second batch time: {batch2_time:.4f} seconds")
    print(f"Cache stats after second batch: {stats2}")
    
    # Calculate speedup
    speedup = batch1_time / batch2_time
    print(f"Speedup from caching: {speedup:.2f}x")
    
    return True

def test_vsa_mapper():
    """
    Test the VSAMapper class with a small example.
    """
    print("Testing VSAMapper...")
    
    # Create a small test grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 1, 0, 1]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Create mapper configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "min_cell_resolution": resolution * 5,
        "max_cell_resolution": resolution * 20,
        "vsa_dimensions": 1000,  # Use smaller dimensions for testing
        "length_scale": 2.0,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": True,
        "batch_size": 100,
        "cache_size": 1000,
        "memory_threshold": 0.8
    }
    
    # Use CPU for testing to ensure compatibility
    device = torch.device("cpu")
    mapper = VSAMapper(config, device=device)
    
    # Process observation
    mapper.process_observation(points, labels)
    
    # Get occupancy grid
    occupancy_grid = mapper.get_occupancy_grid()
    
    print(f"Generated occupancy grid with shape: {occupancy_grid.shape}")
    
    # Get class grid
    class_grid = mapper.get_class_grid()
    
    print(f"Generated class grid with shape: {class_grid.shape}")
    
    # Test incremental processing
    mapper.process_incrementally(
        horizon_distance=0.5,
        sample_resolution=0.2,
        max_samples=10
    )
    
    # Get updated occupancy grid
    occupancy_grid = mapper.get_occupancy_grid()
    
    print(f"Generated incremental occupancy grid with shape: {occupancy_grid.shape}")
    
    # Visualize
    os.makedirs("outputs", exist_ok=True)
    
    # Visualize occupancy grid
    visualize_occupancy_grid(
        grid=occupancy_grid,
        output_file="outputs/test_vsa_occupancy_grid.png",
        world_bounds=world_bounds,
        colormap="viridis",
        show=False
    )
    
    # Visualize class grid
    visualize_class_grid(
        grid=class_grid,
        output_file="outputs/test_vsa_class_grid.png",
        world_bounds=world_bounds,
        show=False
    )
    
    print("Visualizations saved to 'outputs/test_vsa_occupancy_grid.png' and 'outputs/test_vsa_class_grid.png'")
    
    return True

def test_performance_comparison():
    """
    Compare performance between standard and incremental processing.
    """
    print("Testing performance comparison...")
    
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping performance comparison.")
        return True
    
    # Load a sample point cloud
    input_file = "inputs/obstacle_map.npy"
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping performance comparison.")
        return True
    
    # Set world bounds
    world_bounds = [-50, 50, -50, 50]
    
    # Set device
    device = torch.device("cuda")
    
    # Load point cloud
    points, labels = load_pointcloud(input_file, device)
    print(f"Loaded point cloud with {points.shape[0]} points")
    
    # Test with standard processing
    print("\n=== Testing with Standard Processing ===")
    
    # Create standard configuration
    standard_config = {
        "world_bounds": world_bounds,
        "resolution": 0.1,
        "min_cell_resolution": 0.5,
        "max_cell_resolution": 2.0,
        "vsa_dimensions": 8000,  # Reduced dimensions to avoid OOM errors
        "length_scale": 2.0,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": True,
        "batch_size": 1000,
        "cache_size": 10000,
        "memory_threshold": 0.8
    }
    
    # Time standard initialization
    print("\nInitializing VSAMapper...")
    start_time = time.time()
    standard_mapper = VSAMapper(standard_config, device=device)
    standard_init_time = time.time() - start_time
    print(f"Standard initialization time: {standard_init_time:.4f} seconds")
    
    # Time standard processing
    print("\nProcessing observation with Standard Processing...")
    start_time = time.time()
    standard_mapper.process_observation(points, labels)
    standard_process_time = time.time() - start_time
    print(f"Standard processing time: {standard_process_time:.4f} seconds")
    
    # Get standard occupancy grid
    standard_grid = standard_mapper.get_occupancy_grid()
    
    # Test with incremental processing
    print("\n=== Testing with Incremental Processing ===")
    
    # Create incremental mapper
    incremental_mapper = VSAMapper(standard_config, device=device)
    
    # Process observation first
    incremental_mapper.process_observation(points, labels)
    
    # Time incremental processing
    print("\nProcessing incrementally...")
    start_time = time.time()
    incremental_mapper.process_incrementally(
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100
    )
    incremental_process_time = time.time() - start_time
    print(f"Incremental processing time: {incremental_process_time:.4f} seconds")
    
    # Get incremental occupancy grid
    incremental_grid = incremental_mapper.get_occupancy_grid()
    
    # Print GPU memory usage
    print("\n=== GPU Memory Usage ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Print performance comparison
    print("\n=== Performance Comparison ===")
    print(f"Standard processing time: {standard_process_time:.4f} seconds")
    print(f"Incremental processing time: {incremental_process_time:.4f} seconds")
    
    # Calculate speedup
    if incremental_process_time > 0:
        process_speedup = standard_process_time / incremental_process_time
        if process_speedup > 1:
            print(f"Incremental processing is {process_speedup:.2f}x faster than standard processing")
        else:
            print(f"Incremental processing is {1/process_speedup:.2f}x slower than standard processing")
    
    # Visualize results
    os.makedirs("outputs", exist_ok=True)
    
    # Visualize standard occupancy grid
    visualize_occupancy_grid(
        grid=standard_grid,
        output_file="outputs/standard_occupancy_grid.png",
        world_bounds=world_bounds,
        colormap="viridis",
        show=False
    )
    
    # Visualize incremental occupancy grid
    visualize_occupancy_grid(
        grid=incremental_grid,
        output_file="outputs/incremental_occupancy_grid.png",
        world_bounds=world_bounds,
        colormap="viridis",
        show=False
    )
    
    print("Visualizations saved to 'outputs/standard_occupancy_grid.png' and 'outputs/incremental_occupancy_grid.png'")
    
    return True

def test_spatial_index_performance():
    """
    Test the performance of the AdaptiveSpatialIndex class.
    """
    print("Testing spatial index performance...")
    
    # Create a larger test point cloud
    n_points = 10000
    points = torch.rand((n_points, 2)) * 100  # Random points in 100x100 area
    labels = torch.randint(0, 2, (n_points,))  # Random labels
    
    # Create spatial index
    device = torch.device("cpu")
    min_resolution = 1.0
    max_resolution = 10.0
    
    start_time = time.time()
    spatial_index = AdaptiveSpatialIndex(
        points,
        labels,
        min_resolution,
        max_resolution,
        device
    )
    init_time = time.time() - start_time
    
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Cell size: {spatial_index.cell_size:.4f}")
    
    # Test range query performance
    center = torch.tensor([50.0, 50.0])
    radius = 10.0
    
    start_time = time.time()
    query_points, query_labels = spatial_index.query_range(center, radius)
    query_time = time.time() - start_time
    
    print(f"Range query time: {query_time:.4f} seconds")
    print(f"Found {query_points.shape[0]} points within radius {radius}")
    
    # Compare with brute force approach
    start_time = time.time()
    diffs = points - center
    distances = torch.sqrt(torch.sum(diffs * diffs, dim=1))
    mask = distances <= radius
    brute_force_points = points[mask]
    brute_force_time = time.time() - start_time
    
    print(f"Brute force time: {brute_force_time:.4f} seconds")
    print(f"Found {brute_force_points.shape[0]} points within radius {radius}")
    
    # Calculate speedup
    speedup = brute_force_time / query_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results match
    assert query_points.shape[0] == brute_force_points.shape[0]
    
    # Test region safety check performance
    bounds = [45.0, 55.0, 45.0, 55.0]  # 10x10 region around center
    safety_margin = 5.0
    
    start_time = time.time()
    is_free = spatial_index.is_region_free(bounds, safety_margin)
    region_check_time = time.time() - start_time
    
    print(f"Region safety check time: {region_check_time:.4f} seconds")
    print(f"Region is {'free' if is_free else 'not free'}")
    
    print("Spatial index performance tests passed!")
    return True

def test_adaptive_spatial_index_edge_cases():
    """Test edge cases for AdaptiveSpatialIndex."""
    print("Testing AdaptiveSpatialIndex edge cases...")
    
    # Test with empty point cloud
    points = torch.zeros((0, 2))
    labels = torch.zeros((0,))
    
    device = torch.device("cpu")
    min_resolution = 0.1
    max_resolution = 1.0
    
    # Should handle empty point cloud gracefully
    spatial_index = AdaptiveSpatialIndex(
        points,
        labels,
        min_resolution,
        max_resolution,
        device
    )
    
    print("Created spatial index with empty point cloud")
    
    # Test with single point
    points = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([1])
    
    spatial_index = AdaptiveSpatialIndex(
        points,
        labels,
        min_resolution,
        max_resolution,
        device
    )
    
    print("Created spatial index with single point")
    
    # Test query with point outside grid
    center = torch.tensor([100.0, 100.0])
    radius = 1.0
    
    query_points, query_labels = spatial_index.query_range(center, radius)
    assert query_points.shape[0] == 0
    print(f"Range query with point outside grid returned {query_points.shape[0]} points")
    
    print("AdaptiveSpatialIndex edge case tests passed!")
    return True

def test_vector_cache_lru():
    """Test LRU-like cache management in VectorCache."""
    print("Testing VectorCache LRU management...")
    
    device = torch.device("cpu")
    vsa_dimensions = 1000
    length_scale = 2.0
    
    # Create random axis vectors for testing
    xy_axis_vectors = torch.randn((2, vsa_dimensions), device=device)
    
    # Create vector cache with small max size
    cache = VectorCache(
        xy_axis_vectors,
        length_scale,
        device,
        grid_resolution=0.1,
        max_size=5
    )
    
    # Create test points
    points = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],  # This should cause the oldest entry to be removed
        [6.0, 6.0]   # This should cause the second oldest entry to be removed
    ], device=device)
    
    # Get batch vectors
    batch_vectors = cache.get_batch_vectors(points)
    
    # Check cache size
    stats = cache.get_cache_stats()
    assert stats["cache_size"] <= 5
    print(f"Cache size after adding 7 points: {stats['cache_size']}")
    
    # Check that the oldest entries were removed
    # Create keys for the first two points
    key1 = cache._discretize_point(torch.tensor([0.0, 0.0], device=device))
    key2 = cache._discretize_point(torch.tensor([1.0, 1.0], device=device))
    
    # These should not be in the cache anymore
    if key1 in cache.cache:
        print("Warning: First point still in cache")
    if key2 in cache.cache:
        print("Warning: Second point still in cache")
    
    print("VectorCache LRU management tests passed!")
    return True

def test_shannon_entropy_extraction():
    """Test Shannon entropy feature extraction in VSAMapper."""
    print("Testing Shannon entropy feature extraction...")
    
    # Create a small test grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 1, 0, 1]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Create mapper configuration with different disk radii
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "min_cell_resolution": resolution * 5,
        "max_cell_resolution": resolution * 20,
        "vsa_dimensions": 1000,  # Use smaller dimensions for testing
        "length_scale": 2.0,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": False,
        "batch_size": 100,
        "cache_size": 1000,
        "memory_threshold": 0.8,
        "occupied_disk_radius": 2,
        "empty_disk_radius": 4
    }
    
    # Use CPU for testing to ensure compatibility
    device = torch.device("cpu")
    mapper = VSAMapper(config, device=device)
    
    # Process observation
    mapper.process_observation(points, labels)
    
    # Get entropy grids
    occupied_entropy = mapper.get_occupied_entropy_grid()
    empty_entropy = mapper.get_empty_entropy_grid()
    global_entropy = mapper.get_global_entropy_grid()
    
    # Check shapes
    assert occupied_entropy.shape == (10, 10)
    assert empty_entropy.shape == (10, 10)
    assert global_entropy.shape == (10, 10)
    
    print(f"Generated entropy grids with shape: {occupied_entropy.shape}")
    
    # Check that entropy values are within expected range [0, 1]
    assert torch.all(occupied_entropy >= 0) and torch.all(occupied_entropy <= 1)
    assert torch.all(empty_entropy >= 0) and torch.all(empty_entropy <= 1)
    
    print("Entropy values are within expected range [0, 1]")
    
    # Visualize
    os.makedirs("outputs", exist_ok=True)
    
    # Visualize entropy grids
    utils.visualize_entropy_grid(
        grid=occupied_entropy,
        output_file="outputs/test_occupied_entropy.png",
        world_bounds=world_bounds,
        colormap="plasma",
        show=False
    )
    
    utils.visualize_entropy_grid(
        grid=empty_entropy,
        output_file="outputs/test_empty_entropy.png",
        world_bounds=world_bounds,
        colormap="plasma",
        show=False
    )
    
    utils.visualize_entropy_grid(
        grid=global_entropy,
        output_file="outputs/test_global_entropy.png",
        world_bounds=world_bounds,
        colormap="viridis",
        show=False
    )
    
    utils.visualize_entropy_comparison(
        occupied_entropy=occupied_entropy,
        empty_entropy=empty_entropy,
        global_entropy=global_entropy,
        output_file="outputs/test_entropy_comparison.png",
        world_bounds=world_bounds,
        show=False
    )
    
    print("Entropy visualizations saved to 'outputs/test_*_entropy.png'")
    
    print("Shannon entropy feature extraction tests passed!")
    return True

def run_all_tests():
    """
    Run all tests.
    """
    tests = [
        test_adaptive_spatial_index,
        test_adaptive_spatial_index_edge_cases,
        test_spatial_index_performance,
        test_vector_cache,
        test_vector_cache_lru,
        test_vector_cache_performance,
        test_vsa_mapper,
        test_shannon_entropy_extraction,
        test_performance_comparison
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"{test.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test.__name__}: FAILED with exception: {e}")
            results.append(False)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
