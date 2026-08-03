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


def _compute_cell_minimums(
    x: FloatArray1D,
    y: FloatArray1D,
    z: FloatArray1D,
    cell_size: float,
) -> FloatArray1D:
    """
    Compute the minimum elevation for every grid cell.

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
        x, y, z = _extract_xyz(cloud)

        ground_z = _compute_cell_minimums(
            x,
            y,
            z,
            self._cell_size,
        )

        return (z - ground_z) <= self._height_threshold

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
        x, y, z = _extract_xyz(cloud)

        return _compute_cell_minimums(
            x,
            y,
            z,
            self._cell_size,
        )

    @override
    def name(self) -> str:
        return "grid"


__all__ = [
    "GridGroundClassifier",
    "GridGroundExtractor",
    "GridGroundElevationEstimator",
]
