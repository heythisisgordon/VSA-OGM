"""
Unit tests for vector operations.

This module contains unit tests for the vector operations in the Sequential VSA-OGM system.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_ops import (
    bind, 
    power, 
    invert, 
    normalize, 
    make_unitary, 
    batch_bind, 
    similarity
)


class TestVectorOps(unittest.TestCase):
    """Test cases for vector operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.vector_dim = 1024
        self.test_vector1 = make_unitary(self.vector_dim, self.device)
        self.test_vector2 = make_unitary(self.vector_dim, self.device)
        self.test_scalar = 0.5
        
    def test_make_unitary(self):
        """Test make_unitary function."""
        # Test vector creation
        vector = make_unitary(self.vector_dim, self.device)
        
        # Check shape
        self.assertEqual(vector.shape, (self.vector_dim,))
        
        # Check norm is approximately 1
        self.assertAlmostEqual(torch.linalg.norm(vector).item(), 1.0, places=5)
        
        # Check different calls produce different vectors
        vector2 = make_unitary(self.vector_dim, self.device)
        self.assertFalse(torch.allclose(vector, vector2))
        
    def test_bind(self):
        """Test bind function."""
        # Test binding two vectors
        bound = bind([self.test_vector1, self.test_vector2], self.device)
        
        # Check shape
        self.assertEqual(bound.shape, (self.vector_dim,))
        
        # Check binding is commutative
        bound2 = bind([self.test_vector2, self.test_vector1], self.device)
        self.assertTrue(torch.allclose(bound, bound2, atol=1e-5))
        
        # Check binding with list and tensor input
        tensor_input = torch.stack([self.test_vector1, self.test_vector2])
        bound_tensor = bind(tensor_input, self.device)
        self.assertTrue(torch.allclose(bound, bound_tensor, atol=1e-5))
        
        # Check binding preserves norm approximately
        self.assertAlmostEqual(torch.linalg.norm(bound).item(), 1.0, places=5)
        
    def test_power(self):
        """Test power function."""
        # Test fractional binding
        powered = power(self.test_vector1, self.test_scalar)
        
        # Check shape
        self.assertEqual(powered.shape, (self.vector_dim,))
        
        # Check power with scalar 0 returns original vector
        powered_zero = power(self.test_vector1, 0.0)
        self.assertTrue(torch.allclose(powered_zero, torch.ones_like(powered_zero), atol=1e-5))
        
        # Check power with scalar 1 returns vector
        powered_one = power(self.test_vector1, 1.0)
        self.assertTrue(torch.allclose(powered_one, self.test_vector1, atol=1e-5))
        
        # Check power with negative scalar is inverse of positive scalar
        powered_pos = power(self.test_vector1, self.test_scalar)
        powered_neg = power(self.test_vector1, -self.test_scalar)
        dot_product = torch.sum(powered_pos * powered_neg).item()
        self.assertAlmostEqual(dot_product, 1.0, places=5)
        
    def test_invert(self):
        """Test invert function."""
        # Test inversion
        inverted = invert(self.test_vector1)
        
        # Check shape
        self.assertEqual(inverted.shape, (self.vector_dim,))
        
        # Check inversion is its own inverse
        inverted_twice = invert(inverted)
        self.assertTrue(torch.allclose(inverted_twice, self.test_vector1))
        
        # Check binding with inverse gives identity-like behavior
        bound = bind([self.test_vector1, inverted], self.device)
        similarity_val = similarity(bound, torch.ones_like(bound))
        self.assertGreater(similarity_val.item(), 0.9)
        
    def test_normalize(self):
        """Test normalize function."""
        # Create non-normalized vector
        non_normalized = torch.randn(self.vector_dim, device=self.device)
        
        # Test normalization
        normalized = normalize(non_normalized)
        
        # Check shape
        self.assertEqual(normalized.shape, (self.vector_dim,))
        
        # Check norm is 1
        self.assertAlmostEqual(torch.linalg.norm(normalized).item(), 1.0, places=5)
        
        # Check normalization of zero vector
        zero_vector = torch.zeros(self.vector_dim, device=self.device)
        normalized_zero = normalize(zero_vector)
        self.assertTrue(torch.allclose(normalized_zero, zero_vector))
        
    def test_batch_bind(self):
        """Test batch_bind function."""
        # Create batch of vectors
        batch_size = 10
        vectors = torch.zeros((batch_size, self.vector_dim), device=self.device)
        for i in range(batch_size):
            vectors[i] = make_unitary(self.vector_dim, self.device)
            
        # Create axis vectors
        num_axes = 2
        axis_vectors = torch.zeros((num_axes, self.vector_dim), device=self.device)
        for i in range(num_axes):
            axis_vectors[i] = make_unitary(self.vector_dim, self.device)
            
        # Create values
        values = torch.rand((batch_size, num_axes), device=self.device)
        
        # Test batch binding
        batch_bound = batch_bind(vectors, axis_vectors, values)
        
        # Check shape
        self.assertEqual(batch_bound.shape, (batch_size, self.vector_dim))
        
        # Check individual binding matches batch binding
        for i in range(batch_size):
            # Compute individual binding
            powered_axes = []
            for j in range(num_axes):
                powered_axes.append(power(axis_vectors[j], values[i, j].item()))
            individual_bound = bind([vectors[i]] + powered_axes, self.device)
            
            # Check similarity
            sim = similarity(batch_bound[i], individual_bound)
            self.assertGreater(sim.item(), 0.99)
            
    def test_similarity(self):
        """Test similarity function."""
        # Test similarity of identical vectors
        sim = similarity(self.test_vector1, self.test_vector1)
        self.assertAlmostEqual(sim.item(), 1.0, places=5)
        
        # Test similarity of orthogonal vectors
        # Orthogonal vectors should have similarity close to 0
        sim = similarity(self.test_vector1, self.test_vector2)
        self.assertLess(abs(sim.item()), 0.1)
        
        # Test similarity is symmetric
        sim1 = similarity(self.test_vector1, self.test_vector2)
        sim2 = similarity(self.test_vector2, self.test_vector1)
        self.assertAlmostEqual(sim1.item(), sim2.item(), places=5)
        
        # Test batch similarity
        batch_size = 5
        vectors1 = torch.zeros((batch_size, self.vector_dim), device=self.device)
        vectors2 = torch.zeros((batch_size, self.vector_dim), device=self.device)
        for i in range(batch_size):
            vectors1[i] = make_unitary(self.vector_dim, self.device)
            vectors2[i] = make_unitary(self.vector_dim, self.device)
            
        batch_sim = similarity(vectors1, vectors2)
        self.assertEqual(batch_sim.shape, (batch_size,))
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with zero vector
        zero_vector = torch.zeros(self.vector_dim, device=self.device)
        
        # Binding with zero vector
        bound_zero = bind([self.test_vector1, zero_vector], self.device)
        self.assertTrue(torch.allclose(bound_zero, zero_vector))
        
        # Power with very large scalar
        large_scalar = 1000.0
        powered_large = power(self.test_vector1, large_scalar)
        self.assertFalse(torch.isnan(powered_large).any())
        
        # Power with very small scalar
        small_scalar = 1e-6
        powered_small = power(self.test_vector1, small_scalar)
        self.assertFalse(torch.isnan(powered_small).any())
        
        # Similarity with zero vector
        sim_zero = similarity(self.test_vector1, zero_vector)
        self.assertEqual(sim_zero.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
