"""
topocore.processing.features
============================

Feature computation for point clouds.

This package provides infrastructure for computing per-point features
(attributes) from a point cloud. Features describe local geometry,
topology, and radiometry of the point cloud.

Features available:
- PCA-based: eigenvalues, eigenvectors, omnivariance, anisotropy,
  linearity, planarity, sphericity, surface_variation, verticality,
  eigenentropy
- Geometric: height, relative_height, density, distance_to_neighbor
- Normals: normal, curvature (computed separately in processing.normals)

Public API
----------
- FeatureComputer: abstract base class for feature computers
- ScalarFeatureComputer: base class for scalar features
- VectorFeatureComputer: base class for vector features
- PCAFeatures: PCA-based feature computation
- PCAFeatureComputer: wrapper for PCA features
- FeatureManager: high-level manager for feature computation and caching

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import FeatureComputer, ScalarFeatureComputer, VectorFeatureComputer
from .geometric import (
    DensityFeatureComputer,
    DistanceToNeighborFeatureComputer,
    HeightFeatureComputer,
    RelativeHeightFeatureComputer,
)
from .manager import FeatureManager
from .pca import PCAFeatureComputer, PCAFeatures

__all__ = [
    "FeatureComputer",
    "ScalarFeatureComputer",
    "VectorFeatureComputer",
    "PCAFeatures",
    "PCAFeatureComputer",
    "HeightFeatureComputer",
    "RelativeHeightFeatureComputer",
    "DensityFeatureComputer",
    "DistanceToNeighborFeatureComputer",
    "FeatureManager",
]
