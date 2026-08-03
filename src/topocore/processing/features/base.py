"""
topocore.processing.features.base
=================================

Abstract base class for feature computation.

This module defines the interface for computing per-point features
(attributes) from a point cloud. Features are derived quantities
that describe the local geometry, topology, or radiometry of the
point cloud.

The FeatureComputer interface supports:
- Single feature computation (one feature at a time)
- Batch feature computation (multiple features at once)
- Caching of computed features

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.types import FloatArray1D, FloatArray2D


class FeatureComputer(ABC):
    """
    Abstract interface for computing point cloud features.

    Features are per-point attributes that describe local properties
    such as geometry (normals, curvature), topology (density), or
    radiometry (intensity, RGB).

    Implementations must be thread-safe and efficient for large
    point clouds. Features should be computed on-demand and cached
    when possible.

    Examples
    --------
    >>> computer = NormalFeatureComputer(k=10)
    >>> normals = computer.compute(cloud)
    >>> normals.shape  # (N, 3)

    >>> computer = CurvatureFeatureComputer(k=10)
    >>> curvature = computer.compute(cloud)
    >>> curvature.shape  # (N,)
    """

    @abstractmethod
    def compute(
        self,
        cloud: PointCloud,
    ) -> Any:
        """
        Compute the feature for a point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        Any
            The computed feature(s). Shape depends on the feature type.
        """
        ...

    @abstractmethod
    def name(
        self,
    ) -> str:
        """
        Return the name of the feature.

        Returns
        -------
        str
            Feature name (e.g., "normal", "curvature", "density").
        """
        ...

    @abstractmethod
    def dimension(
        self,
    ) -> int:
        """
        Return the dimension of the feature.

        Returns
        -------
        int
            Feature dimension (1 for scalar, 3 for vector, etc.).
        """
        ...

    @abstractmethod
    def requires_neighbors(
        self,
    ) -> bool:
        """
        Return whether this feature requires neighbor search.

        Returns
        -------
        bool
            True if the feature requires neighbor queries.
        """
        ...

    @abstractmethod
    def default_k(
        self,
    ) -> int | None:
        """
        Return the default number of neighbors for this feature.

        Returns
        -------
        int | None
            Default K value, or None if not applicable.
        """
        ...

    @abstractmethod
    def default_radius(
        self,
    ) -> float | None:
        """
        Return the default radius for this feature.

        Returns
        -------
        float | None
            Default radius, or None if not applicable.
        """
        ...


class ScalarFeatureComputer(FeatureComputer):
    """
    Abstract base class for scalar feature computers.

    Scalar features produce a single value per point (e.g., curvature,
    density, height percentile).
    """

    @abstractmethod
    def compute(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """
        Compute the scalar feature.

        Returns
        -------
        FloatArray1D
            Array of shape (N,) containing the feature values.
        """
        ...

    def dimension(
        self,
    ) -> int:
        return 1


class VectorFeatureComputer(FeatureComputer):
    """
    Abstract base class for vector feature computers.

    Vector features produce multiple values per point (e.g., normal,
    eigenvalues, eigenvectors).
    """

    @abstractmethod
    def compute(
        self,
        cloud: PointCloud,
    ) -> FloatArray2D:
        """
        Compute the vector feature.

        Returns
        -------
        FloatArray2D
            Array of shape (N, D) containing the feature values.
        """
        ...


__all__ = [
    "FeatureComputer",
    "ScalarFeatureComputer",
    "VectorFeatureComputer",
]
