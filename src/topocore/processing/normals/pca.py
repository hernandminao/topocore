"""
topocore.processing.normals.pca
===============================

PCA-based normal and curvature estimation.

This module implements standard PCA-based normal estimation:
- Compute the covariance matrix of the k nearest neighbors
- The eigenvector corresponding to the smallest eigenvalue is the normal
- The curvature is the ratio of the smallest eigenvalue to the sum

The normals are oriented upward by default (z >= 0).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

import numpy as np

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import compute_pca
from topocore.processing.exceptions import NormalError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    Vector3D,
)

from .base import NormalAndCurvatureEstimator, NormalEstimator


class PCANormalEstimator(NormalEstimator, NormalAndCurvatureEstimator):
    """
    PCA-based normal and curvature estimation.

    This estimator uses Principal Component Analysis (PCA) on the
    local neighborhood of each point to estimate the surface normal
    and curvature.

    Parameters
    ----------
    k
        Number of neighbors for PCA (must be at least 3).
    orient_upward
        If True, flip normals so z >= 0 (default: True).
    viewpoint
        If provided, orient normals toward this point instead of upward.
    """

    __slots__ = (
        "_k",
        "_orient_upward",
        "_viewpoint",
    )

    def __init__(
        self,
        k: int = 10,
        orient_upward: bool = True,
        viewpoint: Vector3D | None = None,
    ) -> None:
        if k < 3:
            raise NormalError(f"k must be at least 3 for PCA, got {k}.")

        self._k = k
        self._orient_upward = orient_upward
        self._viewpoint = viewpoint

    @override
    def estimate(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray2D:
        """Estimate normals for all points."""

        normals, _ = self.estimate_both(
            cloud,
            manager=manager,
        )

        return normals

    @override
    def estimate_at(
        self,
        cloud: PointCloud,
        indices: IntArray1D | None = None,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray2D:
        """Estimate normals for specific points."""

        normals, _ = self.estimate_both(
            cloud,
            manager=manager,
        )

        if indices is not None:
            return normals[indices].astype(
                np.float64,
                copy=False,
            )

        return normals

    @override
    def estimate_both(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> tuple[FloatArray2D, FloatArray1D]:
        """Estimate both normals and curvature."""

        if cloud.is_empty:
            raise NormalError("Cannot estimate normals on an empty point cloud.")

        if cloud.point_count < self._k:
            raise NormalError(f"Point cloud has {cloud.point_count} points, but PCA requires at least {self._k}.")

        if manager is None:
            manager = NeighborhoodManager.from_point_cloud(
                cloud,
            )

        pca = compute_pca(
            manager,
            k=self._k,
        )

        eigenvalues = pca.eigenvalues
        eigenvectors = pca.eigenvectors

        if eigenvalues.shape[0] != cloud.point_count:
            raise NormalError("Invalid PCA eigenvalues size.")

        if eigenvectors.shape[0] != cloud.point_count:
            raise NormalError("Invalid PCA eigenvectors size.")

        # eigh returns ascending eigenvalues.
        # Smallest eigenvector = normal.
        normals = eigenvectors[:, :, 0]

        sum_l = eigenvalues.sum(
            axis=1,
        )

        curvature = np.where(
            sum_l > 0,
            eigenvalues[:, 0] / sum_l,
            0.0,
        )

        normals = self._orient_normals(
            normals,
            pca.points,
        )

        if not np.isfinite(normals).all():
            raise NormalError("Normal estimation produced invalid values.")

        if not np.isfinite(curvature).all():
            raise NormalError("Curvature estimation produced invalid values.")
        return normals, curvature

    @override
    def name(self) -> str:
        return "pca"

    @override
    def requires_k(self) -> int | None:
        return self._k

    @override
    def supports_weighted(self) -> bool:
        return False

    def _orient_normals(
        self,
        normals: FloatArray2D,
        points: FloatArray2D,
    ) -> FloatArray2D:
        """
        Orient normals upward or toward the viewpoint.
        """
        if self._viewpoint is not None:
            # Orient toward viewpoint
            to_viewpoint = self._viewpoint[None, :] - points
            alignment = np.einsum("ni,ni->n", normals, to_viewpoint)
            flip = alignment < 0.0
            normals = normals.copy()
            normals[flip] *= -1.0
            return normals

        if self._orient_upward:
            # Orient upward (z >= 0)
            normals = normals.copy()
            flip = normals[:, 2] < 0.0
            normals[flip] *= -1.0

        return normals


class PCACurvatureEstimator:
    """
    PCA-based curvature estimation only (without normals).

    This is a convenience wrapper that computes only curvature
    using PCA.

    Parameters
    ----------
    k
        Number of neighbors for PCA (must be at least 3).
    """

    __slots__ = ("_estimator",)

    def __init__(
        self,
        k: int = 10,
    ) -> None:
        self._estimator = PCANormalEstimator(k)

    def estimate(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray1D:
        """Estimate curvature for all points."""
        _, curvature = self._estimator.estimate_both(
            cloud,
            manager=manager,
        )
        return curvature

    def estimate_at(
        self,
        cloud: PointCloud,
        indices: IntArray1D | None = None,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> FloatArray1D:
        """Estimate curvature for specific points."""
        _, curvature = self._estimator.estimate_both(
            cloud,
            manager=manager,
        )

        if indices is not None:
            return curvature[indices].astype(
                np.float64,
                copy=False,
            )

        return curvature

    def name(self) -> str:
        return "pca_curvature"

    def requires_k(self) -> int | None:
        return self._estimator.requires_k()


__all__ = [
    "PCANormalEstimator",
    "PCACurvatureEstimator",
]
