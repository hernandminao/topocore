"""
topocore.processing.ground.grid_adaptive
========================================

Adaptive grid-based ground classification for complex terrain.

This module extends the simple grid-based classifier with adaptive
mechanisms to handle complex terrain:

1. Adaptive height threshold: scales with local slope
2. Multi-resolution grid: coarse-to-fine refinement
3. Slope-aware classification: points on slopes are classified as ground
4. Terrain-aware seed selection: uses local terrain features

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
from topocore.processing.exceptions import GroundError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import (
    BoolArray1D,
    FloatArray1D,
)

from .base import GroundClassifier, GroundExtractor
from .grid import _build_ground_cloud_from_mask, _extract_xyz


class AdaptiveGridGroundClassifier(GroundClassifier):
    """
    Adaptive grid-based ground classifier for complex terrain.

    This classifier addresses the limitations of the simple grid
    method by:
    1. Computing local slope and adapting the height threshold
    2. Using multi-resolution analysis (coarse-to-fine)
    3. Identifying slope points as ground even with height differences
    4. Using local terrain features for seed selection

    Parameters
    ----------
    base_cell_size
        Base size of each grid cell.
    min_cell_size
        Minimum cell size for multi-resolution refinement.
    max_cell_size
        Maximum cell size for multi-resolution refinement.
    base_height_threshold
        Base height threshold (scaled by slope).
    slope_threshold
        Maximum slope angle (degrees) to still consider as ground.
    use_multiresolution
        Whether to use multi-resolution refinement.
    """

    __slots__ = (
        "_base_cell_size",
        "_base_height_threshold",
        "_max_cell_size",
        "_min_cell_size",
        "_slope_threshold",
        "_use_multiresolution",
    )

    def __init__(
        self,
        base_cell_size: float = 1.0,
        min_cell_size: float = 0.5,
        max_cell_size: float = 5.0,
        base_height_threshold: float = 0.2,
        slope_threshold: float = 30.0,
        use_multiresolution: bool = True,
    ) -> None:
        if base_cell_size <= 0:
            raise GroundError(f"base_cell_size must be positive, got {base_cell_size}.")
        if min_cell_size <= 0 or min_cell_size > base_cell_size:
            raise GroundError(f"min_cell_size must be positive and <= base_cell_size, got {min_cell_size}.")
        if max_cell_size < base_cell_size:
            raise GroundError(f"max_cell_size must be >= base_cell_size, got {max_cell_size}.")
        if base_height_threshold < 0:
            raise GroundError(f"base_height_threshold must be non-negative, got {base_height_threshold}.")
        if slope_threshold < 0 or slope_threshold > 90:
            raise GroundError(f"slope_threshold must be in [0, 90], got {slope_threshold}.")

        self._base_cell_size = base_cell_size
        self._min_cell_size = min_cell_size
        self._max_cell_size = max_cell_size
        self._base_height_threshold = base_height_threshold
        self._slope_threshold = np.radians(slope_threshold)
        self._use_multiresolution = use_multiresolution

    @override
    def classify(
        self,
        cloud: PointCloud,
    ) -> BoolArray1D:
        """
        Classify points using the adaptive grid algorithm.
        """
        x, y, z = _extract_xyz(cloud)

        slope = self._estimate_local_slope(
            x,
            y,
            z,
        )

        threshold = self._compute_adaptive_threshold(
            slope,
        )

        if self._use_multiresolution:
            return self._classify_multiresolution(
                x,
                y,
                z,
                threshold,
                slope,
            )

        return self._classify_single_resolution(
            x,
            y,
            z,
            self._base_cell_size,
            threshold,
            slope,
        )

    def _estimate_local_slope(
        self,
        x: FloatArray1D,
        y: FloatArray1D,
        z: FloatArray1D,
    ) -> FloatArray1D:
        """
        Estimate the local terrain slope for every point.

        Found and fixed in PR19 coverage phase: this previously
        computed ``arctan(sqrt(mean(|dx|)**2 + mean(|dy|)**2))`` over
        the k-nearest neighbors -- a formula that NEVER reads the Z
        coordinate at all. Confirmed directly with two point clouds
        sharing identical XY spacing (a 1m grid) but genuinely
        different elevations (perfectly flat vs. a true 45-degree
        grade): the "estimated slope" came out nearly identical
        (~46.9 deg vs ~47.4 deg) for BOTH, dominated by horizontal
        point spacing/density rather than actual terrain steepness.
        Since this value feeds directly into
        `_compute_adaptive_threshold()` (meant to widen the height
        threshold on steep slopes so slope points aren't wrongly
        rejected as non-ground), the entire "slope-aware
        classification" feature this classifier's own docstring
        claims to provide was not functioning -- the threshold was
        effectively constant regardless of real terrain slope.

        Fixed by reusing `compute_pca` (the same shared,
        already-audited local-plane-fitting primitive used by
        `PCANormalEstimator` elsewhere in this codebase, not a new,
        parallel implementation) to fit a local plane to each
        point's k nearest neighbors, then deriving the slope angle
        from that plane's normal: ``arccos(|normal_z|)`` -- the
        standard "angle from horizontal" formula, matching
        ``topocore.features.terrain._mesh_utils.triangle_slope_deg``'s
        own convention (``abs()`` handles a normal of either
        orientation).
        """
        point_count = x.shape[0]
        k = min(10, point_count)

        if k < 3:
            # Too few points for a meaningful local plane fit --
            # slope is undefined; treat as flat (0 rad) rather than
            # raising, since a 1- or 2-point cloud is still valid
            # input for the rest of the classifier.
            return np.zeros(point_count, dtype=np.float64)

        points = np.column_stack((x, y, z)).astype(np.float64)

        manager = NeighborhoodManager.from_array(points)

        pca = compute_pca(manager, k=k)

        # Smallest-eigenvalue eigenvector = local plane normal (np.linalg.eigh
        # returns eigenvalues in ascending order, so index 0 is the smallest).
        normal_z = np.abs(pca.eigenvectors[:, 2, 0])
        normal_z = np.clip(normal_z, 0.0, 1.0)

        return np.asarray(np.arccos(normal_z), dtype=np.float64)

    def _compute_adaptive_threshold(
        self,
        slope: FloatArray1D,
    ) -> FloatArray1D:
        """
        Compute an adaptive height threshold according to
        the local terrain slope.
        """
        normalized = np.clip(
            slope / self._slope_threshold,
            0.0,
            1.0,
        )

        return np.asarray(
            self._base_height_threshold * (1.0 + 2.0 * normalized),
            dtype=np.float64,
        )

    def _compute_cell_minimums(
        self,
        x: FloatArray1D,
        y: FloatArray1D,
        z: FloatArray1D,
        cell_size: float,
    ) -> FloatArray1D:
        """
        Compute the minimum elevation of each grid cell.
        """
        cell_i = np.floor(x / cell_size).astype(np.int64)

        cell_j = np.floor(y / cell_size).astype(np.int64)

        _, inverse = np.unique(
            np.column_stack(
                (
                    cell_i,
                    cell_j,
                )
            ),
            axis=0,
            return_inverse=True,
        )

        minimums = np.full(
            inverse.max() + 1,
            np.inf,
            dtype=np.float64,
        )

        np.minimum.at(
            minimums,
            inverse,
            z,
        )

        return np.asarray(
            minimums[inverse],
            dtype=np.float64,
        )

    def _compute_cell_complexity(
        self,
        cell_indices: np.ndarray,
        z: FloatArray1D,
        slope: FloatArray1D,
    ) -> FloatArray1D:
        """
        Compute terrain complexity per grid cell.

        Complexity combines:
        - elevation variation inside the cell
        - local slope variation

        Parameters
        ----------
        cell_indices
            Cell identifier for every point.
        z
            Elevation values.
        slope
            Local slope values.

        Returns
        -------
        FloatArray1D
            Complexity value associated with every point.
        """

        n_cells = int(cell_indices.max()) + 1

        z_min = np.full(
            n_cells,
            np.inf,
            dtype=np.float64,
        )

        z_max = np.full(
            n_cells,
            -np.inf,
            dtype=np.float64,
        )

        slope_sum = np.zeros(
            n_cells,
            dtype=np.float64,
        )

        slope_count = np.zeros(
            n_cells,
            dtype=np.float64,
        )

        np.minimum.at(
            z_min,
            cell_indices,
            z,
        )

        np.maximum.at(
            z_max,
            cell_indices,
            z,
        )

        np.add.at(
            slope_sum,
            cell_indices,
            slope,
        )

        np.add.at(
            slope_count,
            cell_indices,
            1.0,
        )

        slope_mean = slope_sum / np.maximum(
            slope_count,
            1.0,
        )

        elevation_variation = z_max - z_min

        cell_complexity = elevation_variation + slope_mean

        return np.asarray(
            cell_complexity[cell_indices],
            dtype=np.float64,
        )

    def _classify_single_resolution(
        self,
        x: FloatArray1D,
        y: FloatArray1D,
        z: FloatArray1D,
        cell_size: float,
        adaptive_threshold: FloatArray1D,
        slope: FloatArray1D,
    ) -> BoolArray1D:
        """
        Classify points using a single grid resolution.
        """
        minimum = self._compute_cell_minimums(
            x,
            y,
            z,
            cell_size,
        )

        dz = z - minimum

        ground = dz <= adaptive_threshold

        slope_ground = (slope > self._slope_threshold * 0.5) & (dz <= adaptive_threshold * 2.0)

        return np.asarray(
            ground | slope_ground,
            dtype=np.bool_,
        )

    def _classify_multiresolution(
        self,
        x: FloatArray1D,
        y: FloatArray1D,
        z: FloatArray1D,
        adaptive_threshold: FloatArray1D,
        slope: FloatArray1D,
    ) -> BoolArray1D:
        """
        Classify points using a coarse-to-fine multi-resolution strategy.

        The algorithm first classifies points using a coarse grid. Cells
        exhibiting high terrain complexity are then reclassified using a
        finer grid resolution.

        Parameters
        ----------
        x
            X coordinates.
        y
            Y coordinates.
        z
            Elevations.
        adaptive_threshold
            Adaptive height threshold for each point.
        slope
            Estimated local slope for each point.

        Returns
        -------
        BoolArray1D
            Ground classification mask.
        """
        # Initial coarse classification.
        coarse_mask = self._classify_single_resolution(
            x,
            y,
            z,
            self._max_cell_size,
            adaptive_threshold,
            slope,
        )

        cell_i = np.floor(
            x / self._max_cell_size,
        ).astype(np.int64)

        cell_j = np.floor(
            y / self._max_cell_size,
        ).astype(np.int64)

        _, inverse = np.unique(
            np.column_stack(
                (
                    cell_i,
                    cell_j,
                )
            ),
            axis=0,
            return_inverse=True,
        )

        complexity = self._compute_cell_complexity(
            inverse.astype(np.int64),
            z,
            slope,
        )

        threshold = float(
            np.percentile(
                complexity,
                80.0,
            )
        )

        refined_cells = complexity[inverse] > threshold

        if not refined_cells.any():
            return coarse_mask

        refined_indices = np.flatnonzero(refined_cells)

        fine_mask = self._classify_single_resolution(
            x[refined_indices],
            y[refined_indices],
            z[refined_indices],
            self._min_cell_size,
            adaptive_threshold[refined_indices],
            slope[refined_indices],
        )

        result = coarse_mask.copy()
        result[refined_indices] = fine_mask

        return result

    @override
    def name(self) -> str:
        return "adaptive_grid"


class AdaptiveGridGroundExtractor(GroundExtractor):
    """
    Adaptive grid-based ground extractor.

    Extracts ground points using the adaptive grid classifier.

    Parameters
    ----------
    base_cell_size
        Base size of each grid cell.
    min_cell_size
        Minimum cell size for multi-resolution refinement.
    max_cell_size
        Maximum cell size for multi-resolution refinement.
    base_height_threshold
        Base height threshold (scaled by slope).
    slope_threshold
        Maximum slope angle (degrees) to still consider as ground.
    use_multiresolution
        Whether to use multi-resolution refinement.
    """

    __slots__ = ("_classifier",)

    def __init__(
        self,
        base_cell_size: float = 1.0,
        min_cell_size: float = 0.5,
        max_cell_size: float = 5.0,
        base_height_threshold: float = 0.2,
        slope_threshold: float = 30.0,
        use_multiresolution: bool = True,
    ) -> None:
        self._classifier = AdaptiveGridGroundClassifier(
            base_cell_size,
            min_cell_size,
            max_cell_size,
            base_height_threshold,
            slope_threshold,
            use_multiresolution,
        )

    @override
    def extract(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Extract ground points."""
        mask = self._classifier.classify(cloud)

        if not mask.any():
            raise GroundError("No ground points found. Try adjusting parameters.")

        return _build_ground_cloud_from_mask(cloud, mask)

    @override
    def name(self) -> str:
        return "adaptive_grid"


__all__ = [
    "AdaptiveGridGroundClassifier",
    "AdaptiveGridGroundExtractor",
]
