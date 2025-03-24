"""
Unit tests for quadrant memory.

This module contains unit tests for the quadrant memory in the Sequential VSA-OGM system.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quadrant_memory import QuadrantMemory
from src.vector_ops import similarity


class TestQuadrantMemory(unittest.TestCase):
    """Test cases for quadrant memory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.vector_dim = 1024
        self.world_bounds = (-50.0, 50.0, -50.0, 50.0)  # (xmin, xmax, ymin, ymax)
        self.quadrant_size = 4  # 4x4 grid of quadrants
        
        # Create quadrant memory
        self.memory = QuadrantMemory(
            world_bounds=self.world_bounds,
            quadrant_size=self.quadrant_size,
            vector_dim=self.vector_dim,
            device=self.device
        )
        
        # Create test points
        self.test_points = torch.tensor([
            [0.0, 0.0],    # Center
            [25.0, 25.0],  # Upper right quadrant
            [-25.0, 25.0], # Upper left quadrant
            [-25.0, -25.0],# Lower left quadrant
            [25.0, -25.0]  # Lower right quadrant
        ], device=self.device)
        
        # Create labels (1=occupied, 0=empty)
        self.test_labels = torch.tensor([1, 1, 0, 0, 1], device=self.device)
        
    def test_initialization(self):
        """Test initialization of quadrant memory."""
        # Check world dimensions
        self.assertEqual(self.memory.world_width, 100.0)
        self.assertEqual(self.memory.world_height, 100.0)
        
        # Check quadrant dimensions
        self.assertEqual(self.memory.quadrant_width, 25.0)
        self.assertEqual(self.memory.quadrant_height, 25.0)
        
        # Check axis vectors
        self.assertEqual(self.memory.axis_vectors.shape, (2, self.vector_dim))
        
        # Check quadrant centers
        self.assertEqual(self.memory.quadrant_centers.shape, (self.quadrant_size**2, 2))
        
        # Check memory vectors
        self.assertEqual(self.memory.occupied_memory.shape, (self.quadrant_size**2, self.vector_dim))
        self.assertEqual(self.memory.empty_memory.shape, (self.quadrant_size**2, self.vector_dim))
        
    def test_get_quadrant_index(self):
        """Test getting quadrant index for points."""
        # Test center point
        idx = self.memory.get_quadrant_index([0.0, 0.0])
        self.assertEqual(idx, 5)  # Should be in the center-right quadrant
        
        # Test upper right corner
        idx = self.memory.get_quadrant_index([49.0, 49.0])
        self.assertEqual(idx, 15)  # Should be in the upper right quadrant
        
        # Test lower left corner
        idx = self.memory.get_quadrant_index([-49.0, -49.0])
        self.assertEqual(idx, 0)  # Should be in the lower left quadrant
        
        # Test edge case - exactly on boundary
        idx = self.memory.get_quadrant_index([0.0, -50.0])
        self.assertEqual(idx, 4)  # Should be in the bottom-center quadrant
        
        # Test edge case - exactly on corner
        idx = self.memory.get_quadrant_index([-50.0, -50.0])
        self.assertEqual(idx, 0)  # Should be in the lower left quadrant
        
        # Test edge case - exactly on upper bound
        idx = self.memory.get_quadrant_index([50.0, 50.0])
        self.assertEqual(idx, 15)  # Should be in the upper right quadrant
        
        # Test with tensor input
        idx = self.memory.get_quadrant_index(torch.tensor([25.0, 25.0], device=self.device))
        self.assertEqual(idx, 10)  # Should be in the upper right quadrant
        
        # Test with numpy input
        idx = self.memory.get_quadrant_index(np.array([25.0, 25.0]))
        self.assertEqual(idx, 10)  # Should be in the upper right quadrant
        
        # Test with out of bounds point (should raise ValueError)
        with self.assertRaises(ValueError):
            self.memory.get_quadrant_index([100.0, 100.0])
        
    def test_encode_point(self):
        """Test encoding points into vectors."""
        # Encode a point
        point = torch.tensor([25.0, 25.0], device=self.device)
        encoded = self.memory.encode_point(point)
        
        # Check shape
        self.assertEqual(encoded.shape, (self.vector_dim,))
        
        # Check that encoding is deterministic
        encoded2 = self.memory.encode_point(point)
        self.assertTrue(torch.allclose(encoded, encoded2))
        
        # Check that different points have different encodings
        point2 = torch.tensor([0.0, 0.0], device=self.device)
        encoded3 = self.memory.encode_point(point2)
        self.assertFalse(torch.allclose(encoded, encoded3))
        
        # Check with numpy input
        encoded4 = self.memory.encode_point(np.array([25.0, 25.0]))
        self.assertTrue(torch.allclose(encoded, encoded4))
        
    def test_update_with_point(self):
        """Test updating memory with a single point."""
        # Update with occupied point
        point = torch.tensor([25.0, 25.0], device=self.device)
        quadrant_idx = self.memory.get_quadrant_index(point)
        
        # Get initial memory
        initial_memory = self.memory.occupied_memory[quadrant_idx].clone()
        
        # Update memory
        self.memory.update_with_point(point, is_occupied=True)
        
        # Check that memory has changed
        self.assertFalse(torch.allclose(initial_memory, self.memory.occupied_memory[quadrant_idx]))
        
        # Update with empty point
        point = torch.tensor([-25.0, -25.0], device=self.device)
        quadrant_idx = self.memory.get_quadrant_index(point)
        
        # Get initial memory
        initial_memory = self.memory.empty_memory[quadrant_idx].clone()
        
        # Update memory
        self.memory.update_with_point(point, is_occupied=False)
        
        # Check that memory has changed
        self.assertFalse(torch.allclose(initial_memory, self.memory.empty_memory[quadrant_idx]))
        
    def test_update_with_points(self):
        """Test updating memory with multiple points."""
        # Get initial memories
        initial_occupied = self.memory.occupied_memory.clone()
        initial_empty = self.memory.empty_memory.clone()
        
        # Update with test points
        self.memory.update_with_points(self.test_points, self.test_labels)
        
        # Check that memories have changed
        self.assertFalse(torch.allclose(initial_occupied, self.memory.occupied_memory))
        self.assertFalse(torch.allclose(initial_empty, self.memory.empty_memory))
        
        # Check specific quadrants
        # Center point (occupied)
        center_idx = self.memory.get_quadrant_index([0.0, 0.0])
        self.assertFalse(torch.allclose(initial_occupied[center_idx], self.memory.occupied_memory[center_idx]))
        
        # Upper left point (empty)
        ul_idx = self.memory.get_quadrant_index([-25.0, 25.0])
        self.assertFalse(torch.allclose(initial_empty[ul_idx], self.memory.empty_memory[ul_idx]))
        
    def test_normalize_memories(self):
        """Test normalizing memory vectors."""
        # Update with test points
        self.memory.update_with_points(self.test_points, self.test_labels)
        
        # Normalize memories
        self.memory.normalize_memories()
        
        # Check that all vectors are normalized
        for i in range(len(self.memory.occupied_memory)):
            if torch.norm(self.memory.occupied_memory[i]) > 0:
                self.assertAlmostEqual(torch.norm(self.memory.occupied_memory[i]).item(), 1.0, places=5)
            
        for i in range(len(self.memory.empty_memory)):
            if torch.norm(self.memory.empty_memory[i]) > 0:
                self.assertAlmostEqual(torch.norm(self.memory.empty_memory[i]).item(), 1.0, places=5)
        
    def test_query_point(self):
        """Test querying memory for a point."""
        # Update with test points
        self.memory.update_with_points(self.test_points, self.test_labels)
        self.memory.normalize_memories()
        
        # Query a point that should be occupied
        result = self.memory.query_point([25.0, 25.0])
        
        # Check result structure
        self.assertIn('occupied', result)
        self.assertIn('empty', result)
        self.assertIn('quadrant_idx', result)
        
        # Check that occupied similarity is higher than empty similarity
        self.assertGreater(result['occupied'].item(), result['empty'].item())
        
        # Query a point that should be empty
        result = self.memory.query_point([-25.0, 25.0])
        
        # Check that empty similarity is higher than occupied similarity
        self.assertGreater(result['empty'].item(), result['occupied'].item())
        
        # Query a point that hasn't been seen
        result = self.memory.query_point([10.0, 10.0])
        
        # Check that similarities are reasonable
        self.assertTrue(-1.0 <= result['occupied'].item() <= 1.0)
        self.assertTrue(-1.0 <= result['empty'].item() <= 1.0)
        
    def test_query_grid(self):
        """Test querying memory for a grid of points."""
        # Update with test points
        self.memory.update_with_points(self.test_points, self.test_labels)
        self.memory.normalize_memories()
        
        # Query grid
        resolution = 10.0
        result = self.memory.query_grid(resolution)
        
        # Check result structure
        self.assertIn('x_coords', result)
        self.assertIn('y_coords', result)
        self.assertIn('occupied', result)
        self.assertIn('empty', result)
        
        # Check shapes
        expected_x_size = int(self.memory.world_width / resolution)
        expected_y_size = int(self.memory.world_height / resolution)
        self.assertEqual(result['x_coords'].shape[0], expected_x_size)
        self.assertEqual(result['y_coords'].shape[0], expected_y_size)
        self.assertEqual(result['occupied'].shape, (expected_x_size, expected_y_size))
        self.assertEqual(result['empty'].shape, (expected_x_size, expected_y_size))
        
        # Check that values are in reasonable range
        self.assertTrue(torch.all(result['occupied'] >= -1.0))
        self.assertTrue(torch.all(result['occupied'] <= 1.0))
        self.assertTrue(torch.all(result['empty'] >= -1.0))
        self.assertTrue(torch.all(result['empty'] <= 1.0))
        
    def test_get_quadrant_centers(self):
        """Test getting quadrant centers."""
        centers = self.memory.get_quadrant_centers()
        
        # Check shape
        self.assertEqual(centers.shape, (self.quadrant_size**2, 2))
        
        # Check that centers are within world bounds
        self.assertTrue(torch.all(centers[:, 0] >= self.world_bounds[0]))
        self.assertTrue(torch.all(centers[:, 0] <= self.world_bounds[1]))
        self.assertTrue(torch.all(centers[:, 1] >= self.world_bounds[2]))
        self.assertTrue(torch.all(centers[:, 1] <= self.world_bounds[3]))
        
    def test_get_quadrant_bounds(self):
        """Test getting quadrant bounds."""
        bounds = self.memory.get_quadrant_bounds()
        
        # Check structure
        self.assertEqual(len(bounds), 2)
        
        # Check shapes
        self.assertEqual(bounds[0].shape[0], self.quadrant_size + 1)
        self.assertEqual(bounds[1].shape[0], self.quadrant_size + 1)
        
        # Check that bounds span the world
        self.assertEqual(bounds[0][0].item(), self.world_bounds[0])
        self.assertEqual(bounds[0][-1].item(), self.world_bounds[1])
        self.assertEqual(bounds[1][0].item(), self.world_bounds[2])
        self.assertEqual(bounds[1][-1].item(), self.world_bounds[3])
        
    def test_get_memory_vectors(self):
        """Test getting memory vectors."""
        vectors = self.memory.get_memory_vectors()
        
        # Check structure
        self.assertIn('occupied', vectors)
        self.assertIn('empty', vectors)
        
        # Check shapes
        self.assertEqual(vectors['occupied'].shape, (self.quadrant_size**2, self.vector_dim))
        self.assertEqual(vectors['empty'].shape, (self.quadrant_size**2, self.vector_dim))
        
    def test_memory_organization(self):
        """Test memory organization with different quadrant sizes."""
        # Test with different quadrant sizes
        for size in [2, 8, 16]:
            memory = QuadrantMemory(
                world_bounds=self.world_bounds,
                quadrant_size=size,
                vector_dim=self.vector_dim,
                device=self.device
            )
            
            # Check quadrant dimensions
            self.assertEqual(memory.quadrant_width, self.memory.world_width / size)
            self.assertEqual(memory.quadrant_height, self.memory.world_height / size)
            
            # Check memory shapes
            self.assertEqual(memory.occupied_memory.shape, (size**2, self.vector_dim))
            self.assertEqual(memory.empty_memory.shape, (size**2, self.vector_dim))
            
            # Check quadrant centers
            self.assertEqual(memory.quadrant_centers.shape, (size**2, 2))
            
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very small quadrant size
        memory = QuadrantMemory(
            world_bounds=self.world_bounds,
            quadrant_size=1,  # Single quadrant
            vector_dim=self.vector_dim,
            device=self.device
        )
        
        # Check that there's only one quadrant
        self.assertEqual(memory.occupied_memory.shape[0], 1)
        self.assertEqual(memory.empty_memory.shape[0], 1)
        
        # Test with very large quadrant size
        memory = QuadrantMemory(
            world_bounds=self.world_bounds,
            quadrant_size=100,  # Many quadrants
            vector_dim=self.vector_dim,
            device=self.device
        )
        
        # Check that there are many quadrants
        self.assertEqual(memory.occupied_memory.shape[0], 100**2)
        self.assertEqual(memory.empty_memory.shape[0], 100**2)
        
        # Test with different world bounds
        memory = QuadrantMemory(
            world_bounds=(0.0, 100.0, 0.0, 100.0),  # Positive quadrant only
            quadrant_size=self.quadrant_size,
            vector_dim=self.vector_dim,
            device=self.device
        )
        
        # Check world dimensions
        self.assertEqual(memory.world_width, 100.0)
        self.assertEqual(memory.world_height, 100.0)
        
        # Check that quadrant centers are within bounds
        centers = memory.get_quadrant_centers()
        self.assertTrue(torch.all(centers[:, 0] >= 0.0))
        self.assertTrue(torch.all(centers[:, 0] <= 100.0))
        self.assertTrue(torch.all(centers[:, 1] >= 0.0))
        self.assertTrue(torch.all(centers[:, 1] <= 100.0))


if __name__ == "__main__":
    unittest.main()
