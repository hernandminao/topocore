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

import numpy as np

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import NormalError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    Vector3D,
)


def validate_viewpoint(value: Vector3D | None) -> Vector3D | None:
    """
    Validate a `viewpoint` constructor argument shared by
    PCANormalEstimator and WeightedPCANormalEstimator.

    PR21 remediation (PCA-VIEWPOINT-001): previously, `viewpoint` was
    stored as-is with no validation -- a caller passing a plain tuple
    or list (a reasonable mistake given the type hint alone doesn't
    make the ndarray requirement obvious at a glance) would not fail
    here, but later inside `_orient_normals()` with a confusing
    `TypeError`, not the NormalError this class's other invalid
    arguments raise. This validates the same three properties
    NormalManager's own viewpoint setter already checked (is it
    array-like, does it have shape (3,)) plus a numeric-dtype check,
    consistently raising NormalError with a clear message for all of
    them, matching this class's existing error-handling contract.

    Parameters
    ----------
    value
        The viewpoint argument as received by the constructor.

    Returns
    -------
    Vector3D | None
        The validated viewpoint, unchanged, or None.

    Raises
    ------
    NormalError
        If `value` is not None and is not a numeric NumPy array of
        shape (3,).
    """
    if value is None:
        return None

    if not isinstance(value, np.ndarray):
        raise NormalError(f"viewpoint must be a numpy array, got {type(value).__name__}.")

    if value.shape != (3,):
        raise NormalError(f"viewpoint must have shape (3,), got {value.shape}.")

    if not np.issubdtype(value.dtype, np.number):
        raise NormalError(f"viewpoint must have a numeric dtype, got {value.dtype}.")

    return value


def select_at_indices(
    array: FloatArray1D | FloatArray2D,
    indices: IntArray1D | None,
) -> FloatArray1D | FloatArray2D:
    """
    Return `array` unchanged if `indices` is None, otherwise the
    selected rows, cast to float64.

    PR21 remediation (architectural bypass, Phase 4): this exact
    4-line snippet -- select rows if indices is given, cast to
    float64 -- was independently repeated, identically, in
    PCANormalEstimator.estimate_at() (for normals),
    PCACurvatureEstimator.estimate_at() (for curvature),
    WeightedPCANormalEstimator.estimate_at() (for normals), and
    NormalManager.estimate_at() (for normals). Unlike the
    surrounding call each of these makes to obtain the full
    (normals, curvature) result -- which differ for a legitimate
    architectural reason (NormalManager's own caching/dispatch
    responsibility genuinely requires it to work with the shared,
    possibly-cached full array rather than delegating this whole
    method to a specific estimator's own estimate_at(), which would
    recompute from scratch and defeat that cache -- confirmed
    directly before this change) -- this final indexing step has no
    such justification for being duplicated four times.

    Parameters
    ----------
    array
        The full result array (e.g. normals of shape (N, 3), or
        curvature of shape (N,)).
    indices
        If given, the row indices to select. If None, `array` is
        returned as-is (still cast to float64 if not already).

    Returns
    -------
    FloatArray1D | FloatArray2D
        `array[indices]` if `indices` is not None, else `array`.
    """
    if indices is not None:
        return array[indices].astype(np.float64, copy=False)

    return array


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
    "CurvatureEstimator",
    "NormalAndCurvatureEstimator",
    "NormalEstimator",
    "select_at_indices",
    "validate_viewpoint",
]
