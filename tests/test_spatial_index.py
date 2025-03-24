"""
Unit tests for spatial indexing.

This module contains unit tests for the spatial indexing in the Sequential VSA-OGM system.
"""

import unittest
import numpy as np
import torch
import sys
import os
import time

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spatial_index import SpatialIndex


class TestSpatialIndex(unittest.TestCase):
    """Test cases for spatial indexing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        
        # Create a simple point cloud for testing
        self.num_points = 1000
        self.points_2d = torch.rand((self.num_points, 2), device=self.device) * 100
        
        # Create a more complex point cloud with clusters
        cluster_centers = torch.tensor([[25, 25], [75, 75], [25, 75], [75, 25]], device=self.device)
        cluster_points = []
        for center in cluster_centers:
            # Create cluster with normal distribution
            cluster = center + torch.randn((250, 2), device=self.device) * 5
            cluster_points.append(cluster)
        self.clustered_points = torch.cat(cluster_points, dim=0)
        
        # Create a single point for edge case testing
        self.single_point = torch.tensor([[50, 50]], device=self.device)
        
        # Create an empty point cloud for edge case testing
        self.empty_points = torch.zeros((0, 2), device=self.device)
        
    def test_index_creation(self):
        """Test creation of spatial index."""
        # Test with default cell size
        index = SpatialIndex(self.points_2d, device=self.device)
        
        # Check that points are stored correctly
        self.assertTrue(torch.allclose(index.points, self.points_2d))
        
        # Check dimensionality
        self.assertEqual(index.dimensionality, 2)
        
        # Test with custom cell size
        cell_size = 5.0
        index = SpatialIndex(self.points_2d, cell_size=cell_size, device=self.device)
        
        # Check cell size
        self.assertEqual(index.cell_size, cell_size)
        
        # Test with single point
        index = SpatialIndex(self.single_point, device=self.device)
        self.assertEqual(len(index.cell_to_points), 1)
        
        # Test with empty point cloud
        index = SpatialIndex(self.empty_points, device=self.device)
        self.assertEqual(len(index.cell_to_points), 0)
        
    def test_grid_calculation(self):
        """Test grid calculation."""
        cell_size = 10.0
        index = SpatialIndex(self.points_2d, cell_size=cell_size, device=self.device)
        
        # Check grid size calculation
        expected_grid_size = torch.ceil((index.max_bounds - index.min_bounds) / cell_size).long()
        self.assertTrue(torch.allclose(index.grid_size, expected_grid_size))
        
        # Check that all points are assigned to cells
        total_points = sum(len(points) for points in index.cell_to_points.values())
        self.assertEqual(total_points, self.num_points)
        
    def test_range_query(self):
        """Test range query."""
        cell_size = 10.0
        index = SpatialIndex(self.clustered_points, cell_size=cell_size, device=self.device)
        
        # Test query at cluster center
        center = torch.tensor([25, 25], device=self.device)
        radius = 5.0
        indices = index.range_query(center, radius)
        
        # Check that returned points are within radius
        points = index.points[indices]
        distances = torch.sqrt(torch.sum((points - center) ** 2, dim=1))
        self.assertTrue(torch.all(distances <= radius))
        
        # Test query with no points in range
        empty_center = torch.tensor([0, 0], device=self.device)
        empty_indices = index.range_query(empty_center, radius)
        self.assertEqual(len(empty_indices), 0)
        
        # Test query with all points in range
        large_radius = 150.0
        all_indices = index.range_query(center, large_radius)
        self.assertEqual(len(all_indices), len(self.clustered_points))
        
    def test_k_nearest(self):
        """Test k nearest neighbors query."""
        cell_size = 10.0
        index = SpatialIndex(self.clustered_points, cell_size=cell_size, device=self.device)
        
        # Test query at cluster center
        center = torch.tensor([25, 25], device=self.device)
        k = 10
        indices = index.k_nearest(center, k)
        
        # Check that k points are returned
        self.assertEqual(len(indices), k)
        
        # Check that points are sorted by distance
        points = index.points[indices]
        distances = torch.sqrt(torch.sum((points - center) ** 2, dim=1))
        sorted_distances, _ = torch.sort(distances)
        self.assertTrue(torch.allclose(distances, sorted_distances))
        
        # Test with k larger than number of points
        large_k = 2000
        indices = index.k_nearest(center, large_k)
        self.assertEqual(len(indices), len(self.clustered_points))
        
    def test_update_points(self):
        """Test updating points."""
        cell_size = 10.0
        index = SpatialIndex(self.points_2d, cell_size=cell_size, device=self.device)
        
        # Update with new points
        new_points = torch.rand((500, 2), device=self.device) * 100
        index.update_points(new_points)
        
        # Check that points are updated
        self.assertTrue(torch.allclose(index.points, new_points))
        
        # Check that grid is rebuilt
        total_points = sum(len(points) for points in index.cell_to_points.values())
        self.assertEqual(total_points, len(new_points))
        
    def test_get_points_in_quadrant(self):
        """Test getting points in a quadrant."""
        cell_size = 10.0
        index = SpatialIndex(self.clustered_points, cell_size=cell_size, device=self.device)
        
        # Define quadrant bounds
        min_bounds = torch.tensor([20, 20], device=self.device)
        max_bounds = torch.tensor([30, 30], device=self.device)
        
        # Get points in quadrant
        indices = index.get_points_in_quadrant(min_bounds, max_bounds)
        
        # Check that returned points are within quadrant
        points = index.points[indices]
        in_x_range = (points[:, 0] >= min_bounds[0]) & (points[:, 0] <= max_bounds[0])
        in_y_range = (points[:, 1] >= min_bounds[1]) & (points[:, 1] <= max_bounds[1])
        self.assertTrue(torch.all(in_x_range & in_y_range))
        
        # Test with quadrant outside point cloud
        outside_min = torch.tensor([-10, -10], device=self.device)
        outside_max = torch.tensor([-5, -5], device=self.device)
        outside_indices = index.get_points_in_quadrant(outside_min, outside_max)
        self.assertEqual(len(outside_indices), 0)
        
    def test_performance(self):
        """Test performance compared to brute force."""
        # Create a larger point cloud for performance testing
        num_points = 10000
        points = torch.rand((num_points, 2), device=self.device) * 1000
        
        # Create spatial index
        cell_size = 50.0
        index = SpatialIndex(points, cell_size=cell_size, device=self.device)
        
        # Test point and radius
        center = torch.tensor([500, 500], device=self.device)
        radius = 100.0
        
        # Measure time for spatial index query
        start_time = time.time()
        indices = index.range_query(center, radius)
        spatial_time = time.time() - start_time
        
        # Measure time for brute force query
        start_time = time.time()
        distances = torch.sqrt(torch.sum((points - center) ** 2, dim=1))
        brute_indices = torch.where(distances <= radius)[0]
        brute_time = time.time() - start_time
        
        # Check that results are the same
        self.assertEqual(len(indices), len(brute_indices))
        
        # Check that spatial index is faster (should be much faster for large point clouds)
        self.assertLess(spatial_time, brute_time)
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty point cloud
        index = SpatialIndex(self.empty_points, device=self.device)
        
        # Range query should return empty tensor
        center = torch.tensor([0, 0], device=self.device)
        radius = 10.0
        indices = index.range_query(center, radius)
        self.assertEqual(len(indices), 0)
        
        # K nearest should return empty tensor
        k = 5
        indices = index.k_nearest(center, k)
        self.assertEqual(len(indices), 0)
        
        # Test with single point
        index = SpatialIndex(self.single_point, device=self.device)
        
        # Range query should return the point if in range
        indices = index.range_query(self.single_point[0], radius)
        self.assertEqual(len(indices), 1)
        
        # Range query should return empty tensor if out of range
        far_center = torch.tensor([1000, 1000], device=self.device)
        indices = index.range_query(far_center, radius)
        self.assertEqual(len(indices), 0)
        
        # K nearest should return the point
        indices = index.k_nearest(self.single_point[0], k)
        self.assertEqual(len(indices), 1)
        
    def test_3d_points(self):
        """Test with 3D points."""
        # Create 3D point cloud
        num_points = 1000
        points_3d = torch.rand((num_points, 3), device=self.device) * 100
        
        # Create spatial index
        cell_size = 10.0
        index = SpatialIndex(points_3d, cell_size=cell_size, device=self.device)
        
        # Check dimensionality
        self.assertEqual(index.dimensionality, 3)
        
        # Test range query
        center = torch.tensor([50, 50, 50], device=self.device)
        radius = 20.0
        indices = index.range_query(center, radius)
        
        # Check that returned points are within radius
        points = index.points[indices]
        distances = torch.sqrt(torch.sum((points - center) ** 2, dim=1))
        self.assertTrue(torch.all(distances <= radius))


if __name__ == "__main__":
    unittest.main()
