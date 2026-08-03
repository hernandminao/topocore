"""
topocore.processing.normals
===========================

Surface normal and curvature estimation.

This package provides various methods for estimating surface normals
and curvature from point clouds:

- PCA: Standard Principal Component Analysis
- Weighted PCA: Gaussian-weighted PCA for improved accuracy
- Robust PCA: Reweighted PCA for outlier robustness (future)
- MLS: Moving Least Squares (future)

The normals are oriented consistently (upward or toward a viewpoint).

Public API
----------
- NormalEstimator: abstract base class for normal estimators
- CurvatureEstimator: abstract base class for curvature estimators
- NormalAndCurvatureEstimator: combined interface
- PCANormalEstimator: PCA-based normal estimation
- PCACurvatureEstimator: PCA-based curvature estimation
- WeightedPCANormalEstimator: weighted PCA normal estimation
- NormalManager: high-level manager with method selection

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import (
    CurvatureEstimator,
    NormalAndCurvatureEstimator,
    NormalEstimator,
)
from .manager import NormalManager
from .pca import PCACurvatureEstimator, PCANormalEstimator
from .weighted_pca import WeightedPCANormalEstimator

__all__ = [
    "NormalEstimator",
    "CurvatureEstimator",
    "NormalAndCurvatureEstimator",
    "PCANormalEstimator",
    "PCACurvatureEstimator",
    "WeightedPCANormalEstimator",
    "NormalManager",
]
