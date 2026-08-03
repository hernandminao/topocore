"""
topocore.processing.normals.base
================================

Abstract base classes for normal and curvature estimation.

This module defines the interfaces for:
- Normal estimation (computing surface normals)
- Curvature estimation (computing curvature values)

Normal and curvature estimation are fundamental operations for point
cloud analysis, used in classification, segmentation, registration,
and visualization.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
)


class NormalEstimator(ABC):
    """
    Abstract interface for normal estimation.

    Normal estimators compute a unit normal vector for each point
    in the point cloud. The normals are oriented consistently
    (typically upward or toward a viewpoint).

    Examples
    --------
    >>> estimator = PCANormalEstimator(k=10)
    >>> normals = estimator.estimate(cloud)
    >>> normals.shape  # (N, 3)
    """

    @abstractmethod
    def estimate(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray2D:
        """
        Estimate normals for all points.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        FloatArray2D
            Array of shape (N, 3) containing unit normal vectors.
        """
        ...

    @abstractmethod
    def estimate_at(
        self,
        cloud: PointCloud,
        indices: IntArray1D | None = None,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray2D:
        """
        Estimate normals for specific points.

        Parameters
        ----------
        cloud
            Input point cloud.
        indices
            Indices of points to estimate. If None, estimates all.

        Returns
        -------
        FloatArray2D
            Array of shape (M, 3) containing unit normal vectors.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the estimator name."""
        ...

    @abstractmethod
    def requires_k(self) -> int | None:
        """Return the required number of neighbors."""
        ...

    @abstractmethod
    def supports_weighted(self) -> bool:
        """Return whether weighted estimation is supported."""
        ...


class CurvatureEstimator(ABC):
    """
    Abstract interface for curvature estimation.

    Curvature estimators compute a scalar curvature value for each
    point, measuring how much the surface deviates from planarity.

    Examples
    --------
    >>> estimator = CurvatureEstimator(k=10)
    >>> curvature = estimator.estimate(cloud)
    >>> curvature.shape  # (N,)
    """

    @abstractmethod
    def estimate(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray1D:
        """
        Estimate curvature for all points.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        FloatArray1D
            Array of shape (N,) containing curvature values.
        """
        ...

    @abstractmethod
    def estimate_at(
        self,
        cloud: PointCloud,
        indices: IntArray1D | None = None,
    ) -> FloatArray1D:
        """
        Estimate curvature for specific points.

        Parameters
        ----------
        cloud
            Input point cloud.
        indices
            Indices of points to estimate. If None, estimates all.

        Returns
        -------
        FloatArray1D
            Array of shape (M,) containing curvature values.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the estimator name."""
        ...

    @abstractmethod
    def requires_k(self) -> int | None:
        """Return the required number of neighbors."""
        ...


class NormalAndCurvatureEstimator(ABC):
    """
    Combined interface for normal and curvature estimation.

    This interface allows estimating both normals and curvature
    in a single pass, which is more efficient than computing them
    separately.

    Examples
    --------
    >>> estimator = PCANormalEstimator(k=10)
    >>> normals, curvature = estimator.estimate_both(cloud)
    """

    @abstractmethod
    def estimate_both(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> tuple[FloatArray2D, FloatArray1D]:
        """
        Estimate both normals and curvature.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        tuple
            (normals, curvature) where normals has shape (N, 3) and
            curvature has shape (N,).
        """
        ...


__all__ = [
    "NormalEstimator",
    "CurvatureEstimator",
    "NormalAndCurvatureEstimator",
]
