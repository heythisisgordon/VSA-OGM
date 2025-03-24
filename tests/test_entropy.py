"""
Unit tests for entropy extraction.

This module contains unit tests for the entropy extraction in the Sequential VSA-OGM system.
"""

import unittest
import numpy as np
import torch
import sys
import os
import time

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entropy import EntropyExtractor


class TestEntropyExtractor(unittest.TestCase):
    """Test cases for entropy extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.disk_radius = 3
        self.occupied_threshold = 0.6
        self.empty_threshold = 0.3
        
        # Create entropy extractor
        self.extractor = EntropyExtractor(
            disk_radius=self.disk_radius,
            occupied_threshold=self.occupied_threshold,
            empty_threshold=self.empty_threshold,
            device=self.device
        )
        
        # Create test probability fields
        self.grid_size = 50
        
        # Create uniform probability field
        self.uniform_probs = torch.ones((self.grid_size, self.grid_size), device=self.device) * 0.5
        
        # Create random probability field
        self.random_probs = torch.rand((self.grid_size, self.grid_size), device=self.device)
        
        # Create structured probability field with clear occupied and empty regions
        self.structured_probs = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        # Set center region as occupied (high probability)
        center = self.grid_size // 2
        radius = self.grid_size // 4
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius:
                    self.structured_probs[i, j] = 0.9  # High probability (occupied)
                else:
                    self.structured_probs[i, j] = 0.1  # Low probability (empty)
        
    def test_initialization(self):
        """Test initialization of entropy extractor."""
        # Check parameters
        self.assertEqual(self.extractor.disk_radius, self.disk_radius)
        self.assertEqual(self.extractor.occupied_threshold, self.occupied_threshold)
        self.assertEqual(self.extractor.empty_threshold, self.empty_threshold)
        
        # Check disk filter
        self.assertEqual(self.extractor.disk_filter.shape, (2 * self.disk_radius + 1, 2 * self.disk_radius + 1))
        self.assertAlmostEqual(torch.sum(self.extractor.disk_filter).item(), 1.0, places=5)
        
    def test_disk_filter_creation(self):
        """Test creation of disk filter."""
        # Create extractor with different disk radius
        radius = 5
        extractor = EntropyExtractor(disk_radius=radius, device=self.device)
        
        # Check filter shape
        self.assertEqual(extractor.disk_filter.shape, (2 * radius + 1, 2 * radius + 1))
        
        # Check that filter is normalized
        self.assertAlmostEqual(torch.sum(extractor.disk_filter).item(), 1.0, places=5)
        
        # Check that filter is a disk (values inside radius are non-zero, outside are zero)
        size = 2 * radius + 1
        center = radius
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= radius:
                    self.assertGreater(extractor.disk_filter[i, j].item(), 0.0)
                else:
                    self.assertEqual(extractor.disk_filter[i, j].item(), 0.0)
        
    def test_apply_disk_filter(self):
        """Test application of disk filter."""
        # Apply filter to uniform probability field
        filtered = self.extractor._apply_disk_filter(self.uniform_probs)
        
        # Check shape
        self.assertEqual(filtered.shape, self.uniform_probs.shape)
        
        # Check that uniform field remains uniform after filtering
        self.assertTrue(torch.allclose(filtered, self.uniform_probs, atol=1e-5))
        
        # Apply filter to structured probability field
        filtered = self.extractor._apply_disk_filter(self.structured_probs)
        
        # Check that filtering smooths the field (reduces max and increases min)
        self.assertLess(torch.max(filtered).item(), torch.max(self.structured_probs).item())
        self.assertGreater(torch.min(filtered).item(), torch.min(self.structured_probs).item())
        
    def test_calculate_entropy(self):
        """Test calculation of Shannon entropy."""
        # Calculate entropy for uniform probability field
        entropy = self.extractor.calculate_entropy(self.uniform_probs)
        
        # Check shape
        self.assertEqual(entropy.shape, self.uniform_probs.shape)
        
        # Check that uniform field has maximum entropy
        # Maximum entropy for binary distribution is 1.0 (in bits)
        self.assertTrue(torch.allclose(entropy, torch.ones_like(entropy), atol=1e-5))
        
        # Calculate entropy for structured probability field
        entropy = self.extractor.calculate_entropy(self.structured_probs)
        
        # Check that entropy is lower in regions with clear occupied/empty status
        # and higher in transition regions
        center = self.grid_size // 2
        radius = self.grid_size // 4
        
        # Check center (occupied) region
        center_entropy = entropy[center, center].item()
        self.assertLess(center_entropy, 0.5)  # Low entropy in clearly occupied region
        
        # Check outer (empty) region
        outer_entropy = entropy[0, 0].item()
        self.assertLess(outer_entropy, 0.5)  # Low entropy in clearly empty region
        
        # Check transition region
        transition_point = center + radius  # Point at the boundary
        if transition_point < self.grid_size:
            transition_entropy = entropy[transition_point, center].item()
            self.assertGreater(transition_entropy, center_entropy)  # Higher entropy in transition region
        
    def test_apply_born_rule(self):
        """Test application of Born rule."""
        # Create similarity scores in range [-1, 1]
        similarity = torch.linspace(-1.0, 1.0, 100, device=self.device)
        
        # Apply Born rule
        probabilities = self.extractor.apply_born_rule(similarity)
        
        # Check shape
        self.assertEqual(probabilities.shape, similarity.shape)
        
        # Check range [0, 1]
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.all(probabilities <= 1.0))
        
        # Check specific values
        # similarity = -1 -> probability = 0
        self.assertAlmostEqual(probabilities[0].item(), 0.0, places=5)
        
        # similarity = 0 -> probability = 0.25
        middle_idx = len(similarity) // 2
        self.assertAlmostEqual(probabilities[middle_idx].item(), 0.25, places=5)
        
        # similarity = 1 -> probability = 1
        self.assertAlmostEqual(probabilities[-1].item(), 1.0, places=5)
        
    def test_extract_features(self):
        """Test feature extraction."""
        # Create occupied and empty probability fields
        occupied_probs = self.structured_probs.clone()
        empty_probs = 1.0 - occupied_probs  # Inverse of occupied
        
        # Extract features
        features = self.extractor.extract_features(occupied_probs, empty_probs)
        
        # Check result structure
        self.assertIn('occupied_entropy', features)
        self.assertIn('empty_entropy', features)
        self.assertIn('global_entropy', features)
        self.assertIn('classification', features)
        
        # Check shapes
        self.assertEqual(features['occupied_entropy'].shape, occupied_probs.shape)
        self.assertEqual(features['empty_entropy'].shape, empty_probs.shape)
        self.assertEqual(features['global_entropy'].shape, occupied_probs.shape)
        self.assertEqual(features['classification'].shape, occupied_probs.shape)
        
        # Check classification values (-1, 0, 1)
        unique_values = torch.unique(features['classification'])
        self.assertTrue(torch.all(torch.isin(unique_values, torch.tensor([-1, 0, 1], device=self.device))))
        
        # Check that high occupied probability regions are classified as occupied
        center = self.grid_size // 2
        self.assertEqual(features['classification'][center, center].item(), 1)  # Occupied
        
        # Check that low occupied probability regions are classified as empty
        self.assertEqual(features['classification'][0, 0].item(), -1)  # Empty
        
    def test_classify_grid(self):
        """Test grid classification."""
        # Create global entropy field
        global_entropy = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        
        # Set different regions with different entropy values
        center = self.grid_size // 2
        radius = self.grid_size // 4
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius:
                    global_entropy[i, j] = 0.8  # High positive entropy (occupied)
                elif dist < 2 * radius:
                    global_entropy[i, j] = 0.0  # Zero entropy (unknown)
                else:
                    global_entropy[i, j] = -0.8  # High negative entropy (empty)
        
        # Classify grid
        classification = self.extractor.classify_grid(global_entropy)
        
        # Check shape
        self.assertEqual(classification.shape, global_entropy.shape)
        
        # Check classification values
        self.assertEqual(classification[center, center].item(), 1)  # Occupied
        self.assertEqual(classification[0, 0].item(), -1)  # Empty
        
        # Check transition region
        transition_point = center + int(1.5 * radius)
        if transition_point < self.grid_size:
            self.assertEqual(classification[transition_point, center].item(), 0)  # Unknown
        
    def test_get_occupancy_grid(self):
        """Test getting occupancy grid."""
        # Create classification grid
        classification = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int, device=self.device)
        
        # Set different regions with different classifications
        center = self.grid_size // 2
        radius = self.grid_size // 4
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius:
                    classification[i, j] = 1  # Occupied
                elif dist < 2 * radius:
                    classification[i, j] = 0  # Unknown
                else:
                    classification[i, j] = -1  # Empty
        
        # Get occupancy grid
        occupancy_grid = self.extractor.get_occupancy_grid(classification)
        
        # Check shape
        self.assertEqual(occupancy_grid.shape, classification.shape)
        
        # Check binary values (0, 1)
        unique_values = torch.unique(occupancy_grid)
        self.assertTrue(torch.all(torch.isin(unique_values, torch.tensor([0, 1], device=self.device))))
        
        # Check that occupied regions are 1, others are 0
        self.assertEqual(occupancy_grid[center, center].item(), 1)  # Occupied
        self.assertEqual(occupancy_grid[0, 0].item(), 0)  # Empty -> 0
        
        # Check transition region
        transition_point = center + int(1.5 * radius)
        if transition_point < self.grid_size:
            self.assertEqual(occupancy_grid[transition_point, center].item(), 0)  # Unknown -> 0
        
    def test_get_confidence_map(self):
        """Test getting confidence map."""
        # Create global entropy field
        global_entropy = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        
        # Set different regions with different entropy values
        center = self.grid_size // 2
        radius = self.grid_size // 4
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius:
                    global_entropy[i, j] = 0.8  # High positive entropy (occupied)
                elif dist < 2 * radius:
                    global_entropy[i, j] = 0.0  # Zero entropy (unknown)
                else:
                    global_entropy[i, j] = -0.8  # High negative entropy (empty)
        
        # Get confidence map
        confidence = self.extractor.get_confidence_map(global_entropy)
        
        # Check shape
        self.assertEqual(confidence.shape, global_entropy.shape)
        
        # Check range [0, 1]
        self.assertTrue(torch.all(confidence >= 0.0))
        self.assertTrue(torch.all(confidence <= 1.0))
        
        # Check that high absolute entropy regions have high confidence
        self.assertGreater(confidence[center, center].item(), 0.9)  # Occupied
        self.assertGreater(confidence[0, 0].item(), 0.9)  # Empty
        
        # Check that zero entropy regions have low confidence
        transition_point = center + int(1.5 * radius)
        if transition_point < self.grid_size:
            self.assertLess(confidence[transition_point, center].item(), 0.1)  # Unknown
        
    def test_vectorized_operations(self):
        """Test performance of vectorized operations."""
        # Create large probability field
        large_size = 500
        large_probs = torch.rand((large_size, large_size), device=self.device)
        
        # Measure time for vectorized entropy calculation
        start_time = time.time()
        entropy = self.extractor.calculate_entropy(large_probs)
        vectorized_time = time.time() - start_time
        
        # Check that calculation completes in reasonable time
        self.assertLess(vectorized_time, 5.0)  # Should be fast even for large grids
        
        # Check that result has correct shape
        self.assertEqual(entropy.shape, large_probs.shape)
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with extreme probability values
        extreme_probs = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        
        # Set some regions to 0, some to 1
        extreme_probs[:self.grid_size//2, :] = 0.0
        extreme_probs[self.grid_size//2:, :] = 1.0
        
        # Calculate entropy
        entropy = self.extractor.calculate_entropy(extreme_probs)
        
        # Check that entropy is low for extreme probabilities
        self.assertTrue(torch.all(entropy < 0.1))
        
        # Test with NaN values
        nan_probs = torch.ones((self.grid_size, self.grid_size), device=self.device)
        nan_probs[0, 0] = float('nan')
        
        # Calculate entropy (should handle NaN gracefully)
        entropy = self.extractor.calculate_entropy(nan_probs)
        
        # Check that result doesn't have NaN
        self.assertFalse(torch.isnan(entropy).any())


if __name__ == "__main__":
    unittest.main()
