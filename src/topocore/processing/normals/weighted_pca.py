"""
topocore.processing.normals.weighted_pca
========================================

Weighted PCA-based normal estimation.

This module implements weighted PCA normal estimation, where points
closer to the query point have higher influence on the covariance
matrix. This can produce more accurate normals, especially near
edges and corners.

The weighting function is:
    w_i = exp(-distance_i^2 / sigma^2)

where sigma controls the falloff of the Gaussian weight.

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

from .base import (
    NormalAndCurvatureEstimator,
    NormalEstimator,
    select_at_indices,
    validate_viewpoint,
)


class WeightedPCANormalEstimator(
    NormalEstimator,
    NormalAndCurvatureEstimator,
):
    """
    Weighted PCA-based normal estimation.

    This estimator uses weighted PCA where points closer to the
    query point have higher influence. The weighting uses a Gaussian
    kernel.

    Parameters
    ----------
    k
        Number of neighbors (must be at least 3).
    sigma
        Sigma for Gaussian weighting. If None, uses the average
        distance to neighbors.
    orient_upward
        If True, flip normals so z >= 0 (default: True).
    viewpoint
        If provided, orient normals toward this point instead of upward.
    """

    __slots__ = (
        "_k",
        "_sigma",
        "_orient_upward",
        "_viewpoint",
    )

    def __init__(
        self,
        k: int = 10,
        sigma: float | None = None,
        orient_upward: bool = True,
        viewpoint: Vector3D | None = None,
    ) -> None:
        if k < 3:
            raise NormalError(f"k must be at least 3 for weighted PCA, got {k}.")

        self._k = k
        self._sigma = sigma
        self._orient_upward = orient_upward
        self._viewpoint = validate_viewpoint(viewpoint)

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
        """Estimate normals for selected points."""
        normals, _ = self.estimate_both(
            cloud,
            manager=manager,
        )

        return select_at_indices(normals, indices)

    @override
    def estimate_both(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> tuple[FloatArray2D, FloatArray1D]:
        """Estimate both normals and curvature using weighted PCA."""

        # PR21 remediation (NORMALS-SIBLING-001): PCANormalEstimator's
        # own estimate_both() validates these two conditions before
        # ever touching NeighborhoodManager/compute_pca(), giving a
        # clear, domain-specific NormalError. This class previously
        # had no equivalent checks, letting ProcessingError/
        # NeighborError from those lower-level dependencies leak
        # through unwrapped -- a real inconsistency between sibling
        # estimator classes' public exception contracts for the
        # exact same input conditions. Mirrored here, matching this
        # file's own "weighted PCA" wording convention (see
        # __init__'s k < 3 check just above).
        if cloud.is_empty:
            raise NormalError("Cannot estimate normals on an empty point cloud.")

        if cloud.point_count < self._k:
            raise NormalError(
                f"Point cloud has {cloud.point_count} points, but weighted PCA requires at least {self._k}."
            )

        if manager is None:
            manager = NeighborhoodManager.from_point_cloud(
                cloud,
            )

        pca = compute_pca(
            manager,
            k=self._k,
        )

        points = pca.points
        neighbor_points = pca.neighbor_points
        distances = pca.distances

        # Compute centroids (weighted)
        weights = self._compute_weights(distances)

        weight_sums = weights.sum(axis=1, keepdims=True)
        centroids = np.einsum("nki,nk->ni", neighbor_points, weights) / weight_sums

        # Center the points
        centered = neighbor_points - centroids[:, None, :]
        weighted_centered = centered * weights[:, :, None]

        # Compute weighted covariance matrices
        n = self._k
        covariances = np.einsum("nki,nkj->nij", weighted_centered, centered) / n

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)

        # Normal = smallest eigenvector (index 0)
        normals = eigenvectors[:, :, 0]

        # Curvature = smallest eigenvalue / sum of eigenvalues
        sum_l = eigenvalues.sum(axis=1)
        curvature = np.where(sum_l > 0, eigenvalues[:, 0] / sum_l, 0.0)

        # Orient normals
        normals = self._orient_normals(normals, points)

        return normals, curvature

    def _compute_weights(
        self,
        distances: FloatArray2D,
    ) -> FloatArray2D:
        """
        Compute Gaussian weights from distances.

        If sigma is not specified, it is estimated as the mean
        distance to neighbors.
        """
        if self._sigma is not None:
            sigma2 = self._sigma**2
        else:
            sigma2 = distances.mean() ** 2

        if sigma2 == 0:
            # All points are coincident
            return np.ones_like(distances)

        weights = np.exp(-(distances**2) / sigma2)

        # Ensure the query point itself has weight 1
        weights[:, 0] = 1.0

        return weights

    def _orient_normals(
        self,
        normals: FloatArray2D,
        points: FloatArray2D,
    ) -> FloatArray2D:
        """Orient normals upward or toward the viewpoint."""
        if self._viewpoint is not None:
            to_viewpoint = self._viewpoint[None, :] - points
            alignment = np.einsum("ni,ni->n", normals, to_viewpoint)
            flip = alignment < 0.0
            normals = normals.copy()
            normals[flip] *= -1.0
            return normals

        if self._orient_upward:
            normals = normals.copy()
            flip = normals[:, 2] < 0.0
            normals[flip] *= -1.0

        return normals

    @override
    def name(self) -> str:
        return "weighted_pca"

    @override
    def requires_k(self) -> int | None:
        return self._k

    @override
    def supports_weighted(self) -> bool:
        return True


__all__ = [
    "WeightedPCANormalEstimator",
]
