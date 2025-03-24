"""
Sequential VSA-OGM: Vector Symbolic Architecture for Occupancy Grid Mapping.

This package provides a pythonic implementation for converting static 2D point clouds
into probabilistic occupancy grid maps using Vector Symbolic Architecture (VSA) and
Shannon entropy for feature extraction.
"""

from src.mapper import VSAMapper
from src.quadrant_memory import QuadrantMemory
from src.sequential_processor import SequentialProcessor
from src.spatial_index import SpatialIndex
from src.entropy import EntropyExtractor
from src.config import Config

__all__ = [
    'VSAMapper',
    'QuadrantMemory',
    'SequentialProcessor',
    'SpatialIndex',
    'EntropyExtractor',
    'Config'
]

__version__ = '0.1.0'
