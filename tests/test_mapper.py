"""
Unit tests for VSA mapper.

This module contains unit tests for the VSA mapper in the Sequential VSA-OGM system.
"""

import unittest
import numpy as np
import torch
import sys
import os
import time

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mapper import VSAMapper
from src.config import Config


class TestVSAMapper(unittest.TestCase):
    """Test cases for VSA mapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.world_bounds = (-50.0, 50.0, -50.0, 50.0)  # (xmin, xmax, ymin, ymax)
        
        # Create default configuration
        self.config = Config()
        
        # Create mapper
        self.mapper = VSAMapper(
            world_bounds=self.world_bounds,
            config=self.config
        )
        
        # Create test point cloud
        self.num_points = 1000
        self.points = torch.rand((self.num_points, 2), device=self.device) * 100 - 50  # Range [-50, 50]
        
        # Create labels (1=occupied, 0=empty)
        self.labels = torch.randint(0, 2, (self.num_points,), device=self.device)
        
    def test_initialization(self):
        """Test initialization of VSA mapper."""
        # Check world bounds
        self.assertEqual(self.mapper.world_bounds, self.world_bounds)
        
        # Check that components are initialized
        self.assertIsNotNone(self.mapper.memory)
        self.assertIsNotNone(self.mapper.processor)
        self.assertIsNotNone(self.mapper.entropy_extractor)
        self.assertIsNotNone(self.mapper.spatial_index)
        
        # Check that memory has correct dimensions
        self.assertEqual(self.mapper.memory.vector_dim, self.config.vsa.dimensions)
        
        # Check that processor has correct parameters
        self.assertEqual(self.mapper.processor.sample_resolution, self.config.sequential.sample_resolution)
        self.assertEqual(self.mapper.processor.sensor_range, self.config.sequential.sensor_range)
        
    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        # Create custom configuration
        config_dict = {
            "vsa": {
                "dimensions": 512,
                "length_scale": 2.0,
            },
            "quadrant": {
                "size": 16,
            },
            "sequential": {
                "sample_resolution": 5.0,
                "sensor_range": 30.0,
            },
            "entropy": {
                "disk_radius": 5,
                "occupied_threshold": 0.7,
                "empty_threshold": 0.2,
            },
            "system": {
                "device": "cpu",
                "show_progress": False,
            }
        }
        custom_config = Config(config_dict)
        
        # Create mapper with custom configuration
        mapper = VSAMapper(
            world_bounds=self.world_bounds,
            config=custom_config
        )
        
        # Check that configuration is applied
        self.assertEqual(mapper.memory.vector_dim, custom_config.vsa.dimensions)
        self.assertEqual(mapper.memory.quadrant_size, custom_config.quadrant.size)
        self.assertEqual(mapper.processor.sample_resolution, custom_config.sequential.sample_resolution)
        self.assertEqual(mapper.processor.sensor_range, custom_config.sequential.sensor_range)
        self.assertEqual(mapper.entropy_extractor.disk_radius, custom_config.entropy.disk_radius)
        self.assertEqual(mapper.entropy_extractor.occupied_threshold, custom_config.entropy.occupied_threshold)
        self.assertEqual(mapper.entropy_extractor.empty_threshold, custom_config.entropy.empty_threshold)
        
    def test_process_point_cloud(self):
        """Test processing point cloud."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Check that memory is updated
        occupied_memory = self.mapper.memory.occupied_memory
        empty_memory = self.mapper.memory.empty_memory
        
        # Check that at least some memory vectors are non-zero
        self.assertGreater(torch.sum(torch.abs(occupied_memory)).item(), 0.0)
        self.assertGreater(torch.sum(torch.abs(empty_memory)).item(), 0.0)
        
    def test_get_occupancy_grid(self):
        """Test getting occupancy grid."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Get occupancy grid
        occupancy_grid = self.mapper.get_occupancy_grid(resolution=10.0)
        
        # Check result structure
        self.assertIn('grid', occupancy_grid)
        self.assertIn('x_coords', occupancy_grid)
        self.assertIn('y_coords', occupancy_grid)
        
        # Check shapes
        expected_size = int(100.0 / 10.0)  # world_width / resolution
        self.assertEqual(occupancy_grid['grid'].shape, (expected_size, expected_size))
        self.assertEqual(occupancy_grid['x_coords'].shape[0], expected_size)
        self.assertEqual(occupancy_grid['y_coords'].shape[0], expected_size)
        
        # Check that grid contains binary values
        unique_values = torch.unique(occupancy_grid['grid'])
        self.assertTrue(torch.all(torch.isin(unique_values, torch.tensor([0, 1], device=self.device))))
        
    def test_get_entropy_grid(self):
        """Test getting entropy grid."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Get entropy grid
        entropy_grid = self.mapper.get_entropy_grid(resolution=10.0)
        
        # Check result structure
        self.assertIn('grid', entropy_grid)
        self.assertIn('x_coords', entropy_grid)
        self.assertIn('y_coords', entropy_grid)
        
        # Check shapes
        expected_size = int(100.0 / 10.0)  # world_width / resolution
        self.assertEqual(entropy_grid['grid'].shape, (expected_size, expected_size))
        self.assertEqual(entropy_grid['x_coords'].shape[0], expected_size)
        self.assertEqual(entropy_grid['y_coords'].shape[0], expected_size)
        
        # Check that grid contains values in range [-1, 1]
        self.assertTrue(torch.all(entropy_grid['grid'] >= -1.0))
        self.assertTrue(torch.all(entropy_grid['grid'] <= 1.0))
        
    def test_get_classification(self):
        """Test getting classification."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Get classification
        classification = self.mapper.get_classification(resolution=10.0)
        
        # Check result structure
        self.assertIn('grid', classification)
        self.assertIn('x_coords', classification)
        self.assertIn('y_coords', classification)
        
        # Check shapes
        expected_size = int(100.0 / 10.0)  # world_width / resolution
        self.assertEqual(classification['grid'].shape, (expected_size, expected_size))
        self.assertEqual(classification['x_coords'].shape[0], expected_size)
        self.assertEqual(classification['y_coords'].shape[0], expected_size)
        
        # Check that grid contains values in {-1, 0, 1}
        unique_values = torch.unique(classification['grid'])
        self.assertTrue(torch.all(torch.isin(unique_values, torch.tensor([-1, 0, 1], device=self.device))))
        
    def test_get_confidence_map(self):
        """Test getting confidence map."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Get confidence map
        confidence_map = self.mapper.get_confidence_map(resolution=10.0)
        
        # Check result structure
        self.assertIn('grid', confidence_map)
        self.assertIn('x_coords', confidence_map)
        self.assertIn('y_coords', confidence_map)
        
        # Check shapes
        expected_size = int(100.0 / 10.0)  # world_width / resolution
        self.assertEqual(confidence_map['grid'].shape, (expected_size, expected_size))
        self.assertEqual(confidence_map['x_coords'].shape[0], expected_size)
        self.assertEqual(confidence_map['y_coords'].shape[0], expected_size)
        
        # Check that grid contains values in range [0, 1]
        self.assertTrue(torch.all(confidence_map['grid'] >= 0.0))
        self.assertTrue(torch.all(confidence_map['grid'] <= 1.0))
        
    def test_get_quadrant_bounds(self):
        """Test getting quadrant bounds."""
        # Get quadrant bounds
        bounds = self.mapper.get_quadrant_bounds()
        
        # Check structure
        self.assertEqual(len(bounds), 2)
        
        # Check shapes
        self.assertEqual(bounds[0].shape[0], self.config.quadrant.size + 1)
        self.assertEqual(bounds[1].shape[0], self.config.quadrant.size + 1)
        
        # Check that bounds span the world
        self.assertEqual(bounds[0][0].item(), self.world_bounds[0])
        self.assertEqual(bounds[0][-1].item(), self.world_bounds[1])
        self.assertEqual(bounds[1][0].item(), self.world_bounds[2])
        self.assertEqual(bounds[1][-1].item(), self.world_bounds[3])
        
    def test_get_quadrant_centers(self):
        """Test getting quadrant centers."""
        # Get quadrant centers
        centers = self.mapper.get_quadrant_centers()
        
        # Check shape
        self.assertEqual(centers.shape, (self.config.quadrant.size**2, 2))
        
        # Check that centers are within world bounds
        self.assertTrue(torch.all(centers[:, 0] >= self.world_bounds[0]))
        self.assertTrue(torch.all(centers[:, 0] <= self.world_bounds[1]))
        self.assertTrue(torch.all(centers[:, 1] >= self.world_bounds[2]))
        self.assertTrue(torch.all(centers[:, 1] <= self.world_bounds[3]))
        
    def test_get_sample_positions(self):
        """Test getting sample positions."""
        # Get sample positions
        positions = self.mapper.get_sample_positions()
        
        # Check shape
        expected_num_x = int(100.0 / self.config.sequential.sample_resolution)
        expected_num_y = int(100.0 / self.config.sequential.sample_resolution)
        expected_num_positions = expected_num_x * expected_num_y
        self.assertEqual(positions.shape, (expected_num_positions, 2))
        
        # Check that positions are within world bounds
        self.assertTrue(torch.all(positions[:, 0] >= self.world_bounds[0]))
        self.assertTrue(torch.all(positions[:, 0] <= self.world_bounds[1]))
        self.assertTrue(torch.all(positions[:, 1] >= self.world_bounds[2]))
        self.assertTrue(torch.all(positions[:, 1] <= self.world_bounds[3]))
        
    def test_query_point(self):
        """Test querying a point."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Query a point
        point = torch.tensor([0.0, 0.0], device=self.device)
        result = self.mapper.query_point(point)
        
        # Check result structure
        self.assertIn('occupied', result)
        self.assertIn('empty', result)
        self.assertIn('quadrant_idx', result)
        
        # Check that similarities are in range [-1, 1]
        self.assertTrue(-1.0 <= result['occupied'].item() <= 1.0)
        self.assertTrue(-1.0 <= result['empty'].item() <= 1.0)
        
    def test_different_resolutions(self):
        """Test with different resolutions."""
        # Process point cloud
        self.mapper.process_point_cloud(self.points, self.labels, show_progress=False)
        
        # Get grids with different resolutions
        high_res = 5.0
        low_res = 20.0
        
        high_res_grid = self.mapper.get_occupancy_grid(resolution=high_res)
        low_res_grid = self.mapper.get_occupancy_grid(resolution=low_res)
        
        # Check shapes
        expected_high_res_size = int(100.0 / high_res)
        expected_low_res_size = int(100.0 / low_res)
        
        self.assertEqual(high_res_grid['grid'].shape, (expected_high_res_size, expected_high_res_size))
        self.assertEqual(low_res_grid['grid'].shape, (expected_low_res_size, expected_low_res_size))
        
        # Check that high resolution grid has more cells
        self.assertGreater(high_res_grid['grid'].numel(), low_res_grid['grid'].numel())
        
    def test_structured_point_cloud(self):
        """Test with structured point cloud."""
        # Create a structured point cloud with a circle
        num_points = 1000
        radius = 30.0
        center = torch.tensor([0.0, 0.0], device=self.device)
        
        # Generate random angles
        angles = torch.rand(num_points, device=self.device) * 2 * np.pi
        
        # Generate points on circle
        x = center[0] + radius * torch.cos(angles)
        y = center[1] + radius * torch.sin(angles)
        
        # Add some noise
        noise = torch.randn((num_points, 2), device=self.device) * 2.0
        
        # Create points
        circle_points = torch.stack([x, y], dim=1) + noise
        
        # Create labels (all occupied)
        circle_labels = torch.ones(num_points, device=self.device)
        
        # Process point cloud
        self.mapper.process_point_cloud(circle_points, circle_labels, show_progress=False)
        
        # Get occupancy grid
        occupancy_grid = self.mapper.get_occupancy_grid(resolution=5.0)
        
        # Check that grid has occupied cells
        self.assertGreater(torch.sum(occupancy_grid['grid']).item(), 0)
        
        # Check that occupied cells form a rough circle
        grid = occupancy_grid['grid']
        x_coords = occupancy_grid['x_coords']
        y_coords = occupancy_grid['y_coords']
        
        # Find center of grid
        center_x_idx = torch.argmin(torch.abs(x_coords)).item()
        center_y_idx = torch.argmin(torch.abs(y_coords)).item()
        
        # Check that center is not occupied
        self.assertEqual(grid[center_x_idx, center_y_idx].item(), 0)
        
        # Check that some cells at radius distance are occupied
        radius_idx = int(radius / 5.0)  # radius / resolution
        
        # Check right, left, up, down from center
        if center_x_idx + radius_idx < grid.shape[0]:
            self.assertEqual(grid[center_x_idx + radius_idx, center_y_idx].item(), 1)
        
        if center_x_idx - radius_idx >= 0:
            self.assertEqual(grid[center_x_idx - radius_idx, center_y_idx].item(), 1)
        
        if center_y_idx + radius_idx < grid.shape[1]:
            self.assertEqual(grid[center_x_idx, center_y_idx + radius_idx].item(), 1)
        
        if center_y_idx - radius_idx >= 0:
            self.assertEqual(grid[center_x_idx, center_y_idx - radius_idx].item(), 1)
        
    def test_performance(self):
        """Test performance of mapper."""
        # Create a larger point cloud
        num_points = 10000
        points = torch.rand((num_points, 2), device=self.device) * 100 - 50  # Range [-50, 50]
        labels = torch.randint(0, 2, (num_points,), device=self.device)
        
        # Measure time for processing
        start_time = time.time()
        self.mapper.process_point_cloud(points, labels, show_progress=False)
        processing_time = time.time() - start_time
        
        # Check that processing completes in reasonable time
        # This is a rough check and may need adjustment based on hardware
        self.assertLess(processing_time, 30.0)
        
        # Measure time for getting occupancy grid
        start_time = time.time()
        self.mapper.get_occupancy_grid(resolution=5.0)
        grid_time = time.time() - start_time
        
        # Check that grid generation completes in reasonable time
        self.assertLess(grid_time, 5.0)
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty point cloud
        empty_points = torch.zeros((0, 2), device=self.device)
        empty_labels = torch.zeros(0, device=self.device)
        
        # Process empty point cloud
        self.mapper.process_point_cloud(empty_points, empty_labels, show_progress=False)
        
        # Get occupancy grid
        occupancy_grid = self.mapper.get_occupancy_grid(resolution=10.0)
        
        # Check that grid is all zeros
        self.assertEqual(torch.sum(occupancy_grid['grid']).item(), 0)
        
        # Test with single point
        single_point = torch.tensor([[0.0, 0.0]], device=self.device)
        single_label = torch.tensor([1], device=self.device)
        
        # Process single point
        self.mapper.process_point_cloud(single_point, single_label, show_progress=False)
        
        # Get occupancy grid
        occupancy_grid = self.mapper.get_occupancy_grid(resolution=10.0)
        
        # Check that grid has at least one occupied cell
        self.assertGreater(torch.sum(occupancy_grid['grid']).item(), 0)
        
        # Test with out of bounds point
        out_of_bounds = torch.tensor([[1000.0, 1000.0]], device=self.device)
        out_of_bounds_label = torch.tensor([1], device=self.device)
        
        # Process out of bounds point (should be skipped)
        self.mapper.process_point_cloud(out_of_bounds, out_of_bounds_label, show_progress=False)
        
        # Get occupancy grid
        occupancy_grid = self.mapper.get_occupancy_grid(resolution=10.0)
        
        # Check that grid is unchanged
        self.assertGreater(torch.sum(occupancy_grid['grid']).item(), 0)


if __name__ == "__main__":
    unittest.main()
