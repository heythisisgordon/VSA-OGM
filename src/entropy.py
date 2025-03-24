"""
Shannon entropy extraction for feature classification.

This module implements Shannon entropy calculation for feature extraction from
probability fields. It applies optimized disk filters to probability fields and
provides classification mechanisms for occupied/empty determination.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union, Dict, Any


class EntropyExtractor:
    """
    Implements Shannon entropy extraction for feature classification.
    
    This class provides methods to calculate Shannon entropy from probability
    fields and classify regions as occupied, empty, or unknown based on
    entropy thresholds.
    """
    
    def __init__(self, 
                 disk_radius: int = 3,
                 occupied_threshold: float = 0.6,
                 empty_threshold: float = 0.3,
                 device: str = "cpu") -> None:
        """
        Initialize the entropy extractor.
        
        Args:
            disk_radius: Radius of the disk filter for local entropy calculation
            occupied_threshold: Threshold above which a cell is classified as occupied
            empty_threshold: Threshold below which a cell is classified as empty
            device: Device to perform calculations on ("cpu" or "cuda")
        """
        self.disk_radius = disk_radius
        self.occupied_threshold = occupied_threshold
        self.empty_threshold = empty_threshold
        self.device = device
        
        # Create disk filter
        self._create_disk_filter()
        
    def _create_disk_filter(self) -> None:
        """Create a disk filter for local entropy calculation."""
        # Create a square grid
        size = 2 * self.disk_radius + 1
        x = torch.linspace(-self.disk_radius, self.disk_radius, size)
        y = torch.linspace(-self.disk_radius, self.disk_radius, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Create disk mask (1 inside disk, 0 outside)
        disk = (xx**2 + yy**2 <= self.disk_radius**2).float()
        
        # Normalize the disk filter
        self.disk_filter = disk / torch.sum(disk)
        self.disk_filter = self.disk_filter.to(self.device)
        
    def _apply_disk_filter(self, 
                          probability_field: torch.Tensor) -> torch.Tensor:
        """
        Apply the disk filter to a probability field using convolution.
        
        Args:
            probability_field: 2D tensor of probabilities
            
        Returns:
            Filtered probability field
        """
        # Ensure probability field is on the correct device
        probability_field = probability_field.to(self.device)
        
        # Reshape filter for 2D convolution (1, 1, height, width)
        filter_reshaped = self.disk_filter.unsqueeze(0).unsqueeze(0)
        
        # Reshape probability field for 2D convolution (1, 1, height, width)
        prob_reshaped = probability_field.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        filtered = torch.nn.functional.conv2d(
            prob_reshaped, 
            filter_reshaped, 
            padding=self.disk_radius
        )
        
        # Return to original shape
        return filtered.squeeze()
        
    def calculate_entropy(self, 
                         probability_field: Union[np.ndarray, torch.Tensor],
                         eps: float = 1e-10) -> torch.Tensor:
        """
        Calculate Shannon entropy for a probability field.
        
        Args:
            probability_field: 2D array/tensor of probabilities in range [0, 1]
            eps: Small value to avoid log(0)
            
        Returns:
            2D tensor of entropy values
        """
        # Convert to tensor if needed
        if isinstance(probability_field, np.ndarray):
            probability_field = torch.from_numpy(probability_field).float().to(self.device)
        else:
            probability_field = probability_field.float().to(self.device)
            
        # Ensure probabilities are in [0, 1]
        probability_field = torch.clamp(probability_field, eps, 1.0 - eps)
        
        # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -probability_field * torch.log2(probability_field) - \
                 (1 - probability_field) * torch.log2(1 - probability_field)
        
        # Apply disk filter to get local entropy
        local_entropy = self._apply_disk_filter(entropy)
        
        return local_entropy
    
    def extract_features(self, 
                        occupied_probs: Union[np.ndarray, torch.Tensor],
                        empty_probs: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract features from occupied and empty probability fields.
        
        Args:
            occupied_probs: 2D array/tensor of occupancy probabilities
            empty_probs: 2D array/tensor of emptiness probabilities
            
        Returns:
            Dictionary containing:
                - occupied_entropy: Entropy of occupied probabilities
                - empty_entropy: Entropy of empty probabilities
                - global_entropy: Combined entropy (occupied - empty)
                - classification: Cell classification (-1=empty, 0=unknown, 1=occupied)
        """
        # Calculate entropy for occupied and empty probabilities
        occupied_entropy = self.calculate_entropy(occupied_probs)
        empty_entropy = self.calculate_entropy(empty_probs)
        
        # Calculate global entropy (occupied - empty)
        global_entropy = occupied_entropy - empty_entropy
        
        # Classify cells based on global entropy
        classification = torch.zeros_like(global_entropy, dtype=torch.int)
        classification[global_entropy > self.occupied_threshold] = 1  # Occupied
        classification[global_entropy < -self.empty_threshold] = -1   # Empty
        
        return {
            'occupied_entropy': occupied_entropy,
            'empty_entropy': empty_entropy,
            'global_entropy': global_entropy,
            'classification': classification
        }
    
    def apply_born_rule(self, 
                       similarity_scores: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply Born rule to convert similarity scores to probabilities.
        
        The Born rule squares the similarity scores to convert them to probabilities.
        
        Args:
            similarity_scores: Tensor of similarity scores in range [-1, 1]
            
        Returns:
            Tensor of probabilities in range [0, 1]
        """
        # Convert to tensor if needed
        if isinstance(similarity_scores, np.ndarray):
            similarity_scores = torch.from_numpy(similarity_scores).float().to(self.device)
        else:
            similarity_scores = similarity_scores.float().to(self.device)
            
        # Normalize to [0, 1] range and apply Born rule (square)
        normalized = (similarity_scores + 1) / 2
        probabilities = normalized ** 2
        
        return probabilities
    
    def classify_grid(self, 
                     global_entropy: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Classify a grid based on global entropy values.
        
        Args:
            global_entropy: 2D array/tensor of global entropy values
            
        Returns:
            2D tensor of classifications (-1=empty, 0=unknown, 1=occupied)
        """
        # Convert to tensor if needed
        if isinstance(global_entropy, np.ndarray):
            global_entropy = torch.from_numpy(global_entropy).float().to(self.device)
        else:
            global_entropy = global_entropy.float().to(self.device)
            
        # Classify cells based on global entropy
        classification = torch.zeros_like(global_entropy, dtype=torch.int)
        classification[global_entropy > self.occupied_threshold] = 1  # Occupied
        classification[global_entropy < -self.empty_threshold] = -1   # Empty
        
        return classification
    
    def get_occupancy_grid(self, 
                          classification: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert classification to binary occupancy grid.
        
        Args:
            classification: 2D array/tensor of classifications (-1=empty, 0=unknown, 1=occupied)
            
        Returns:
            Binary occupancy grid (0=free/unknown, 1=occupied)
        """
        # Convert to tensor if needed
        if isinstance(classification, np.ndarray):
            classification = torch.from_numpy(classification).to(self.device)
        else:
            classification = classification.to(self.device)
            
        # Create binary occupancy grid (occupied = 1, empty/unknown = 0)
        occupancy_grid = torch.zeros_like(classification, dtype=torch.int)
        occupancy_grid[classification == 1] = 1
        
        return occupancy_grid
    
    def get_confidence_map(self, 
                          global_entropy: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Generate a confidence map from global entropy.
        
        Args:
            global_entropy: 2D array/tensor of global entropy values
            
        Returns:
            Confidence map in range [0, 1]
        """
        # Convert to tensor if needed
        if isinstance(global_entropy, np.ndarray):
            global_entropy = torch.from_numpy(global_entropy).float().to(self.device)
        else:
            global_entropy = global_entropy.float().to(self.device)
            
        # Calculate confidence as absolute value of global entropy
        # scaled to [0, 1] range
        confidence = torch.abs(global_entropy)
        max_confidence = torch.max(confidence)
        if max_confidence > 0:
            confidence = confidence / max_confidence
            
        return confidence
