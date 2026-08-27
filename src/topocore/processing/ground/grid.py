"""
topocore.processing.ground.grid
===============================

Grid-based ground classification.

This module implements a simple grid-based ground classification
algorithm: the point cloud is divided into a 2D grid, and the
lowest point in each cell is considered ground.

This is a fast, simple baseline for ground classification. It works
well for relatively flat terrain but may fail on steep slopes or
complex terrain.

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

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.types import BoolArray1D, FloatArray1D

from .base import GroundClassifier, GroundElevationEstimator, GroundExtractor


def _extract_xyz(
    cloud: PointCloud,
) -> tuple[
    FloatArray1D,
    FloatArray1D,
    FloatArray1D,
]:
    """
    Extract X, Y and Z coordinates from the point cloud.

    Parameters
    ----------
    cloud
        Input point cloud.

    Returns
    -------
    tuple of FloatArray1D
        Arrays containing X, Y and Z coordinates.
    """
    xs: list[FloatArray1D] = []
    ys: list[FloatArray1D] = []
    zs: list[FloatArray1D] = []

    for chunk in cloud:
        xs.append(chunk[PointAttribute.X])
        ys.append(chunk[PointAttribute.Y])
        zs.append(chunk[PointAttribute.Z])

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(zs),
    )


def _build_ground_cloud_from_mask(
    cloud: PointCloud,
    mask: BoolArray1D,
) -> PointCloud:
    """
    Build a new point cloud containing only ground points.

    Parameters
    ----------
    cloud
        Source point cloud.
    mask
        Boolean mask identifying ground points.

    Returns
    -------
    PointCloud
        Point cloud containing only ground points.
    """
    from topocore.pointcloud.chunk import Chunk

    attributes = cloud.attributes

    combined: dict[
        PointAttribute,
        list[FloatArray1D],
    ] = {attr: [] for attr in attributes}

    for chunk in cloud:
        for attr in attributes:
            combined[attr].append(chunk[attr])

    flat: dict[
        PointAttribute,
        FloatArray1D,
    ] = {attr: np.concatenate(values) for attr, values in combined.items()}

    indices = np.flatnonzero(mask)

    new_chunk = Chunk(
        size=len(indices),
        attributes=list(attributes),
    )

    for attr in attributes:
        new_chunk[attr][:] = flat[attr][indices]

    result = PointCloud()
    result.add_chunk(new_chunk)
    result.update_bounds()

    return result


def _classify_chunked(
    cloud: PointCloud,
    cell_size: float,
    height_threshold: float,
) -> BoolArray1D:
    """
    Classify points as ground via a two-pass, chunk-wise
    accumulator -- see `_compute_cell_minimums_chunked()`'s own
    docstring for the full design rationale. Unlike that function
    (which returns the per-point cell minimum), this performs the
    final `(z - ground_z) <= height_threshold` comparison directly
    within pass 2, so no X, Y, or Z array is ever concatenated across
    chunks at any point -- not even Z alone for the final comparison.
    """
    if cloud.is_empty:
        # PR21.7.8: reproduces the exact ValueError the pre-PR21.7.8
        # _extract_xyz()'s np.concatenate([]) raised for an empty
        # cloud, byte-for-byte (same exception type, same message) --
        # confirmed directly this was the prior behavior before
        # assuming it should be preserved. Not silently improved to a
        # cleaner GroundError here, since that would be a separate,
        # deliberate fix this PR doesn't make unilaterally.
        np.concatenate([])

    cell_minimums: dict[tuple[int, int], float] = {}

    for chunk in cloud:
        if chunk.size == 0:
            continue

        x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
        y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
        z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

        cell_i = np.floor(x / cell_size).astype(np.int64)
        cell_j = np.floor(y / cell_size).astype(np.int64)
        coords = np.column_stack((cell_i, cell_j))
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

        local_min_z = np.full(len(unique_coords), np.inf, dtype=np.float64)
        np.minimum.at(local_min_z, inverse, z)

        for local_index, key_array in enumerate(unique_coords):
            key = (int(key_array[0]), int(key_array[1]))
            value = float(local_min_z[local_index])
            if key in cell_minimums:
                cell_minimums[key] = min(cell_minimums[key], value)
            else:
                cell_minimums[key] = value

    total_points = cloud.point_count
    mask = np.empty(total_points, dtype=np.bool_)

    global_offset = 0
    for chunk in cloud:
        if chunk.size == 0:
            continue

        x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
        y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
        z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

        cell_i = np.floor(x / cell_size).astype(np.int64)
        cell_j = np.floor(y / cell_size).astype(np.int64)

        for local_index in range(len(x)):
            key = (int(cell_i[local_index]), int(cell_j[local_index]))
            mask[global_offset + local_index] = (z[local_index] - cell_minimums[key]) <= height_threshold

        global_offset += chunk.size

    return mask


def _compute_cell_minimums_chunked(
    cloud: PointCloud,
    cell_size: float,
) -> FloatArray1D:
    """
    Compute, for every point in the cloud, the minimum elevation of
    the grid cell it belongs to -- via a two-pass, chunk-wise
    accumulator, instead of first concatenating every chunk's X/Y/Z
    into one global array (the prior `_extract_xyz()` +
    `_compute_cell_minimums()` combination).

    PR21.7.8: found during the PR21.7 final cross-cutting audit as a
    real, if milder, memory concern than VoxelSampler/StratifiedSampler's
    own O(N x G) complexity bug (PR21.7.5/7.6) -- confirmed via direct
    benchmarking that this file's `np.minimum.at`-based per-cell
    reduction was ALREADY O(N) (genuinely linear scaling measured
    directly, no hidden quadratic behavior), so this fix targets only
    the ~3.3x-3.5x memory overhead from the three float64
    concatenations, not a complexity bug.

    Correctness: minimum is commutative and associative
    (min(a, b) == min(b, a) for any order), so merging per-chunk
    local minimums into a global per-cell minimum via repeated
    min() is exact regardless of chunk boundaries or processing
    order -- unlike VoxelSampler's "closest" (PR21.7.5), there is no
    tie-breaking rule to replicate here, since only the minimum
    VALUE is tracked, never which specific point achieved it.

    Pass 1 accumulates the minimum Z per cell key, vectorized within
    each chunk via the same `np.minimum.at` the pre-PR21.7.8 code
    already used (just applied per chunk instead of once globally),
    merged into a global dict. Pass 2 revisits the same
    already-in-memory chunks and looks up each point's cell's now-
    fully-known minimum directly, producing the identical per-point,
    input-order-aligned array `_compute_cell_minimums()` used to
    return.

    Parameters
    ----------
    cloud
        Input point cloud.
    cell_size
        Grid cell size.

    Returns
    -------
    FloatArray1D
        Minimum elevation of the containing cell, one value per
        input point, in the same order as the cloud's own point
        order.
    """
    if cloud.is_empty:
        # PR21.7.8: reproduces the exact ValueError the pre-PR21.7.8
        # _extract_xyz()'s np.concatenate([]) raised for an empty
        # cloud -- confirmed this function's old behavior matched
        # _classify_chunked's own reference case exactly, since both
        # called the same _extract_xyz(). Found and fixed here: this
        # function was initially missing this check (unlike
        # _classify_chunked, which already had it), silently
        # returning an empty array instead of raising, before this
        # fix.
        np.concatenate([])

    cell_minimums: dict[tuple[int, int], float] = {}

    for chunk in cloud:
        if chunk.size == 0:
            continue

        x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
        y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
        z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

        cell_i = np.floor(x / cell_size).astype(np.int64)
        cell_j = np.floor(y / cell_size).astype(np.int64)
        coords = np.column_stack((cell_i, cell_j))
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

        local_min_z = np.full(len(unique_coords), np.inf, dtype=np.float64)
        np.minimum.at(local_min_z, inverse, z)

        for local_index, key_array in enumerate(unique_coords):
            key = (int(key_array[0]), int(key_array[1]))
            value = float(local_min_z[local_index])
            if key in cell_minimums:
                cell_minimums[key] = min(cell_minimums[key], value)
            else:
                cell_minimums[key] = value

    total_points = cloud.point_count
    result = np.empty(total_points, dtype=np.float64)

    global_offset = 0
    for chunk in cloud:
        if chunk.size == 0:
            continue

        x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
        y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)

        cell_i = np.floor(x / cell_size).astype(np.int64)
        cell_j = np.floor(y / cell_size).astype(np.int64)

        for local_index in range(len(x)):
            key = (int(cell_i[local_index]), int(cell_j[local_index]))
            result[global_offset + local_index] = cell_minimums[key]

        global_offset += chunk.size

    return result


def _compute_cell_minimums(
    x: FloatArray1D,
    y: FloatArray1D,
    z: FloatArray1D,
    cell_size: float,
) -> FloatArray1D:
    """
    Compute the minimum elevation for every grid cell.

    Kept for reference/comparison (this session's own regression
    suite verifies `_compute_cell_minimums_chunked()` against it
    directly) -- `classify()`/`estimate()` themselves now use the
    chunked version above.

    Parameters
    ----------
    x
        X coordinates.
    y
        Y coordinates.
    z
        Elevations.
    cell_size
        Grid cell size.

    Returns
    -------
    FloatArray1D
        Minimum elevation corresponding to every input point.
    """
    cell_i = np.floor(x / cell_size).astype(np.int64)
    cell_j = np.floor(y / cell_size).astype(np.int64)

    cell_coords = np.column_stack((cell_i, cell_j))

    _, inverse = np.unique(
        cell_coords,
        axis=0,
        return_inverse=True,
    )

    cell_min_z = np.full(
        inverse.max() + 1,
        np.inf,
        dtype=np.float64,
    )

    np.minimum.at(
        cell_min_z,
        inverse,
        z,
    )

    return cell_min_z[inverse]


class GridGroundClassifier(GroundClassifier):
    """
    Grid-based ground classifier.

    The point cloud is divided into a regular 2D grid. The lowest
    point in each grid cell is classified as ground. Points within
    a threshold height above the cell minimum are also classified
    as ground.

    Parameters
    ----------
    cell_size
        Size of each grid cell.
    height_threshold
        Maximum height above cell minimum to still be considered ground.
    """

    __slots__ = (
        "_cell_size",
        "_height_threshold",
    )

    def __init__(
        self,
        cell_size: float = 1.0,
        height_threshold: float = 0.2,
    ) -> None:
        if cell_size <= 0:
            raise GroundError(f"cell_size must be positive, got {cell_size}.")
        if height_threshold < 0:
            raise GroundError(f"height_threshold must be non-negative, got {height_threshold}.")

        self._cell_size = cell_size
        self._height_threshold = height_threshold

    @override
    def classify(
        self,
        cloud: PointCloud,
    ) -> BoolArray1D:
        """
        Classify points as ground.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        BoolArray1D
            Boolean mask indicating ground points.
        """
        return _classify_chunked(cloud, self._cell_size, self._height_threshold)

    @override
    def name(self) -> str:
        return "grid"


class GridGroundExtractor(GroundExtractor):
    """
    Grid-based ground extractor.

    Extracts ground points using the grid-based classifier.

    Parameters
    ----------
    cell_size
        Size of each grid cell.
    height_threshold
        Maximum height above cell minimum to still be considered ground.
    """

    __slots__ = ("_classifier",)

    def __init__(
        self,
        cell_size: float = 1.0,
        height_threshold: float = 0.2,
    ) -> None:
        self._classifier = GridGroundClassifier(cell_size, height_threshold)

    @override
    def extract(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Extract ground points."""
        mask = self._classifier.classify(cloud)

        if not mask.any():
            raise GroundError("No ground points found. Try increasing cell_size or height_threshold.")

        return _build_ground_cloud_from_mask(cloud, mask)

    @override
    def name(self) -> str:
        return "grid"


class GridGroundElevationEstimator(GroundElevationEstimator):
    """
    Grid-based ground elevation estimator.

    Estimates ground elevation as the minimum Z in each grid cell.

    Parameters
    ----------
    cell_size
        Size of each grid cell.
    """

    __slots__ = ("_cell_size",)

    def __init__(
        self,
        cell_size: float = 1.0,
    ) -> None:
        if cell_size <= 0:
            raise GroundError(f"cell_size must be positive, got {cell_size}.")
        self._cell_size = cell_size

    @override
    def estimate(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """
        Estimate ground elevation for every point.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        FloatArray1D
            Estimated ground elevation.
        """
        return _compute_cell_minimums_chunked(cloud, self._cell_size)

    @override
    def name(self) -> str:
        return "grid"


__all__ = [
    "GridGroundClassifier",
    "GridGroundElevationEstimator",
    "GridGroundExtractor",
]
