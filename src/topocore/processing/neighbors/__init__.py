"""
topocore.processing.neighbors
=============================

Spatial neighbor search for point clouds.

This package provides spatial neighbor search capabilities using
KD-tree acceleration. It supports:
- KNN (k-nearest neighbors)
- Radius search
- Batched queries
- Adaptive KNN/radius based on local density

The neighbor search is the foundation for many processing operations:
- Outlier removal
- Normal estimation
- Classification
- Segmentation
- Registration

Public API
----------
- NeighborSearch: abstract interface for neighbor search
- KDTreeNeighborSearch: KD-tree implementation
- NeighborhoodManager: high-level manager with adaptive strategies

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import NeighborSearch
from .kdtree import KDTreeNeighborSearch
from .manager import NeighborhoodManager

__all__ = [
    "NeighborSearch",
    "KDTreeNeighborSearch",
    "NeighborhoodManager",
]
