"""
topocore.processing.features.pca
================================

PCA-based feature computation.

This module implements PCA (Principal Component Analysis) based
features: eigenvalues, eigenvectors, and derived quantities such as
omnivariance, anisotropy, linearity, planarity, sphericity, and
verticality.

These features are fundamental for point cloud analysis and are used
in classification, segmentation, and object detection.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import compute_pca
from topocore.processing.exceptions import PointDescriptorError, ProcessingError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import FloatArray1D, FloatArray2D

from .base import FeatureComputer


class PCAFeatureResult(TypedDict):
    eigenvalues: FloatArray2D
    eigenvectors: NDArray[np.float64]
    omnivariance: FloatArray1D
    anisotropy: FloatArray1D
    linearity: FloatArray1D
    planarity: FloatArray1D
    sphericity: FloatArray1D
    surface_variation: FloatArray1D
    verticality: FloatArray1D
    eigenentropy: FloatArray1D


class PCAFeatures:
    """
    Compute PCA-based features from a point cloud.

    This class computes eigenvalues and eigenvectors from the local
    covariance matrix of each point's neighborhood, and derives
    geometric features from the eigenvalues.

    Features computed:
    - eigenvalues: (?1, ?2, ?3) with ?1 ? ?2 ? ?3
    - eigenvectors: principal directions
    - omnivariance: (?1 * ?2 * ?3)^(1/3)
    - anisotropy: (?1 - ?3) / ?1
    - linearity: (?1 - ?2) / ?1
    - planarity: (?2 - ?3) / ?1
    - sphericity: ?3 / ?1
    - verticality: 1 - |eigenvector_z|
    - surface_variation: ?3 / (?1 + ?2 + ?3)
    - eigenentropy: -sum(p_i * log(p_i)), where p_i = ?_i / sum(?)

    Notes
    -----
    The eigenvalues are clamped to 0.0 after eigendecomposition to
    avoid propagating small negative values caused by numerical error.

    Parameters
    ----------
    k
        Number of neighbors for PCA.
    """

    __slots__ = ("_k",)

    def __init__(
        self,
        k: int = 10,
    ) -> None:
        if k < 3:
            raise PointDescriptorError(f"k must be at least 3 for PCA, got {k}.")
        self._k = k

    @property
    def k(self) -> int:
        return self._k

    def compute_all(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> PCAFeatureResult:
        """
        Compute all PCA-based features for the input cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        PCAFeatureResult
            Dictionary with all PCA-derived features.

        Raises
        ------
        PointDescriptorError
            If the cloud is empty, contains invalid coordinates,
            does not have enough points, neighborhood search returns
            an invalid shape, or eigendecomposition fails.
        """
        if cloud.is_empty:
            raise PointDescriptorError("Cannot compute PCA features on an empty point cloud.")

        if cloud.point_count < self._k:
            raise PointDescriptorError(
                f"Point cloud has {cloud.point_count} points, but PCA requires at least {self._k}."
            )

        # PR21 remediation (PCA-FEATURES-001): previously,
        # `NeighborhoodManager.from_point_cloud(cloud)` sat OUTSIDE
        # this try block -- only the later `compute_pca(...)` call
        # was covered. Confirmed directly that this construction step
        # raises a raw, undocumented `ValueError` (from scipy's own
        # cKDTree rejecting non-finite coordinates) for exactly the
        # "invalid coordinates" case this method's own docstring
        # already promises `PointDescriptorError` for -- the
        # ValueError has no relationship to TopoCore's own exception
        # hierarchy (confirmed via its MRO) and was escaping
        # uncaught. Widening the try to also cover manager
        # construction, and catching that specific, already-
        # identified ValueError alongside the existing
        # ProcessingError, closes the gap without introducing a new,
        # separate finite-coordinate validation (which would
        # duplicate a check scipy's own KD-tree construction already
        # performs) and without changing the manager-construction
        # logic itself.
        try:
            if manager is None:
                manager = NeighborhoodManager.from_point_cloud(
                    cloud,
                )

            pca = compute_pca(
                manager,
                k=self._k,
            )

        except (ProcessingError, ValueError) as exc:
            raise PointDescriptorError(
                str(exc),
            ) from exc

        eigenvalues = pca.eigenvalues
        eigenvectors = pca.eigenvectors

        # Sort eigenvalues in descending order and clamp tiny negative
        # values caused by floating-point error.
        eigvals = np.maximum(eigenvalues[:, ::-1], 0.0)
        eigvecs = eigenvectors[:, :, ::-1]

        # Extract components
        l1 = eigvals[:, 0]
        l2 = eigvals[:, 1]
        l3 = eigvals[:, 2]

        sum_l = l1 + l2 + l3
        denom = np.where(sum_l > 0.0, sum_l, 1.0)
        l1_safe = np.where(l1 > 0.0, l1, 1.0)

        # Eigenentropy is computed from normalized eigenvalues:
        # p_i = ?_i / (?1 + ?2 + ?3)
        eps = np.finfo(np.float64).eps
        normalized = np.clip(eigvals / denom[:, None], eps, None)
        eigenentropy = -(normalized * np.log(normalized)).sum(axis=1)

        return {
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "omnivariance": np.cbrt(np.maximum(l1 * l2 * l3, 0.0)),
            "anisotropy": (l1 - l3) / l1_safe,
            "linearity": (l1 - l2) / l1_safe,
            "planarity": (l2 - l3) / l1_safe,
            "sphericity": l3 / l1_safe,
            "surface_variation": l3 / denom,
            "verticality": np.clip(1.0 - np.abs(eigvecs[:, 2, 2]), 0.0, 1.0),
            "eigenentropy": eigenentropy,
        }

    def eigenvalues(
        self,
        cloud: PointCloud,
    ) -> FloatArray2D:
        """
        Compute eigenvalues (?1, ?2, ?3) for each point.

        Returns
        -------
        FloatArray2D
            Array of shape (N, 3) with ?1 ? ?2 ? ?3.
        """
        return self.compute_all(cloud)["eigenvalues"]

    def eigenvectors(
        self,
        cloud: PointCloud,
    ) -> NDArray[np.float64]:
        """
        Compute eigenvectors for each point.

        Returns
        -------
        NDArray
            Array of shape (N, 3, 3) where each 3x3 matrix contains
            the eigenvectors in columns, ordered to match the sorted
            eigenvalues (?1, ?2, ?3).
        """
        return self.compute_all(cloud)["eigenvectors"]

    def omnivariance(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute omnivariance: (?1 * ?2 * ?3)^(1/3)."""
        return self.compute_all(cloud)["omnivariance"]

    def anisotropy(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute anisotropy: (?1 - ?3) / ?1."""
        return self.compute_all(cloud)["anisotropy"]

    def linearity(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute linearity: (?1 - ?2) / ?1."""
        return self.compute_all(cloud)["linearity"]

    def planarity(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute planarity: (?2 - ?3) / ?1."""
        return self.compute_all(cloud)["planarity"]

    def sphericity(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute sphericity: ?3 / ?1."""
        return self.compute_all(cloud)["sphericity"]

    def surface_variation(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute surface variation: ?3 / (?1 + ?2 + ?3)."""
        return self.compute_all(cloud)["surface_variation"]

    def verticality(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute verticality: 1 - |eigenvector_z|."""
        return self.compute_all(cloud)["verticality"]

    def eigenentropy(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """
        Compute eigenentropy: -sum(p_i * log(p_i)),
        where p_i = ?_i / (?1 + ?2 + ?3).
        """
        return self.compute_all(cloud)["eigenentropy"]

    def compute(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> PCAFeatureResult:
        """
        Compute all PCA-based geometric features.

        This method computes the neighborhood search and PCA
        decomposition once and returns all derived features.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        PCAFeatureResult
            All PCA-derived geometric features.
        """
        return self.compute_all(
            cloud,
            manager=manager,
        )


class PCAFeatureComputer(FeatureComputer):
    """
    FeatureComputer wrapper for PCA features.

    This class provides a unified interface for computing specific
    PCA-based features.

    Parameters
    ----------
    feature_name
        Name of the feature to compute. One of:
        "eigenvalues", "eigenvectors", "omnivariance", "anisotropy",
        "linearity", "planarity", "sphericity", "surface_variation",
        "verticality", "eigenentropy".
    k
        Number of neighbors for PCA.
    """

    __slots__ = (
        "_feature_name",
        "_pca",
    )

    _DIMENSIONS = {
        "eigenvalues": 3,
        "eigenvectors": 9,  # 3x3 matrix flattened in row-major (C-order)
        "omnivariance": 1,
        "anisotropy": 1,
        "linearity": 1,
        "planarity": 1,
        "sphericity": 1,
        "surface_variation": 1,
        "verticality": 1,
        "eigenentropy": 1,
    }

    def __init__(
        self,
        feature_name: str,
        k: int = 10,
    ) -> None:
        if feature_name not in self._DIMENSIONS:
            raise PointDescriptorError(f"Unknown PCA feature: {feature_name}")

        self._feature_name = feature_name
        self._pca = PCAFeatures(k)

    def compute(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute the selected PCA feature.

        Notes
        -----
        For ``feature_name="eigenvectors"``, the returned shape is
        ``(N, 9)`` and each 3x3 eigenvector matrix is flattened using
        NumPy's default row-major order (C-order). The original
        unflattened representation stores eigenvectors in columns.
        """

        features = self._pca.compute_all(
            cloud,
            manager=manager,
        )

        match self._feature_name:
            case "eigenvalues":
                return features["eigenvalues"]

            case "eigenvectors":
                return features["eigenvectors"].reshape(-1, 9)

            case "omnivariance":
                return features["omnivariance"]

            case "anisotropy":
                return features["anisotropy"]

            case "linearity":
                return features["linearity"]

            case "planarity":
                return features["planarity"]

            case "sphericity":
                return features["sphericity"]

            case "surface_variation":
                return features["surface_variation"]

            case "verticality":
                return features["verticality"]

            case "eigenentropy":
                return features["eigenentropy"]

        raise PointDescriptorError(f"Unknown PCA feature: {self._feature_name}")

    def name(self) -> str:
        return f"pca_{self._feature_name}"

    def dimension(self) -> int:
        return self._DIMENSIONS[self._feature_name]

    def requires_neighbors(self) -> bool:
        return True

    def default_k(self) -> int | None:
        return self._pca.k

    def default_radius(self) -> float | None:
        return None


__all__ = [
    "PCAFeatures",
    "PCAFeatureComputer",
]
