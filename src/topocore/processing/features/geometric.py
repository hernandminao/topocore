"""
topocore.processing.features.geometric
======================================

Geometric feature computation.

This module implements geometric features that describe the local
shape and topology of the point cloud, including:

- Height (absolute and relative)
- Slope
- Roughness
- Planarity (alternative to PCA-based)
- Density (local point density)
- Distance to nearest neighbor

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import cast

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import PointDescriptorError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import FloatArray1D, FloatArray2D, IntArray1D

from .base import FeatureComputer


class HeightFeatureComputer(FeatureComputer):
    """
    Compute absolute height (Z coordinate).

    This is a trivial feature that simply returns the Z coordinate
    of each point.
    """

    def compute(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        if PointAttribute.Z not in cloud.attributes:
            raise PointDescriptorError("Point cloud has no Z coordinate.")

        z_values = []
        for chunk in cloud:
            z_values.append(chunk[PointAttribute.Z])
        return np.concatenate(z_values)

    def name(self) -> str:
        return "height"

    def dimension(self) -> int:
        return 1

    def requires_neighbors(self) -> bool:
        return False

    def default_k(self) -> int | None:
        return None

    def default_radius(self) -> float | None:
        return None


class RelativeHeightFeatureComputer(FeatureComputer):
    """
    Compute relative height (height above ground).

    This feature requires the point cloud to have a ground
    classification. The relative height is computed as:
    Z - ground_Z, where ground_Z is the elevation of the
    nearest ground point.

    Parameters
    ----------
    ground_class
        The classification code for ground points (default: 2).
    k
        Number of ground neighbors to consider.
    """

    __slots__ = ("_ground_class", "_k")

    def __init__(
        self,
        ground_class: int = 2,
        k: int = 1,
    ) -> None:
        self._ground_class = ground_class
        self._k = k

    def compute(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        if PointAttribute.Z not in cloud.attributes:
            raise PointDescriptorError("Point cloud has no Z coordinate.")

        if PointAttribute.CLASSIFICATION not in cloud.attributes:
            raise PointDescriptorError("Point cloud has no classification attribute.")

        # Extract all points
        z_values: list[FloatArray1D] = []
        classifications: list[IntArray1D] = []
        for chunk in cloud:
            z_values.append(cast(FloatArray1D, chunk[PointAttribute.Z]))
            classifications.append(cast(IntArray1D, chunk[PointAttribute.CLASSIFICATION]))

        z: FloatArray1D = np.concatenate(z_values)
        cls: IntArray1D = np.concatenate(classifications)

        # Find ground points
        ground_mask = cls == self._ground_class

        if not ground_mask.any():
            raise PointDescriptorError("No ground points found for relative height computation.")

        # Build index from ground points
        ground_points: FloatArray2D = np.stack(
            [
                z[ground_mask],
                np.zeros_like(z[ground_mask]),
                np.zeros_like(z[ground_mask]),
            ],
            axis=1,
        )

        # For each point, find the nearest ground point
        manager = NeighborhoodManager.from_array(ground_points)
        indices, _ = manager.query_point(0.0, 0.0, z[0], k=self._k)

        ground_z: FloatArray1D = ground_points[indices, 0]

        result: FloatArray1D = z - ground_z
        return result

    def name(self) -> str:
        return "relative_height"

    def dimension(self) -> int:
        return 1

    def requires_neighbors(self) -> bool:
        return True

    def default_k(self) -> int | None:
        return self._k

    def default_radius(self) -> float | None:
        return None


class DensityFeatureComputer(FeatureComputer):
    """
    Compute local point density.

    Density is computed as the number of points within a sphere of
    radius `radius` divided by the sphere volume.

    Parameters
    ----------
    radius
        Search radius for density computation.
    """

    __slots__ = ("_radius",)

    def __init__(
        self,
        radius: float = 1.0,
    ) -> None:
        if radius <= 0:
            raise PointDescriptorError(f"radius must be positive, got {radius}.")
        self._radius = radius

    def compute(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        manager = NeighborhoodManager.from_point_cloud(cloud)

        # Query radius for all points
        results = manager.radius_many(radius=self._radius, include_self=True)

        # Compute densities
        volume = (4.0 / 3.0) * np.pi * (self._radius**3)
        densities = np.array([len(r) / volume for r in results])

        return densities

    def name(self) -> str:
        return "density"

    def dimension(self) -> int:
        return 1

    def requires_neighbors(self) -> bool:
        return True

    def default_k(self) -> int | None:
        return None

    def default_radius(self) -> float | None:
        return self._radius


class DistanceToNeighborFeatureComputer(FeatureComputer):
    """
    Compute distance to the nearest neighbor.

    This feature computes the Euclidean distance to the k-th
    nearest neighbor (default: 1st neighbor).

    Parameters
    ----------
    k
        Number of neighbors. The distance to the k-th neighbor is returned.
    """

    __slots__ = ("_k",)

    def __init__(
        self,
        k: int = 1,
    ) -> None:
        if k < 1:
            raise PointDescriptorError(f"k must be at least 1, got {k}.")
        self._k = k

    def compute(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        manager = NeighborhoodManager.from_point_cloud(cloud)

        # Query KNN for all points
        _, distances = manager.knn_many(k=self._k + 1, include_self=False)

        # Return distance to the k-th neighbor (0-indexed)
        return distances[:, self._k - 1]

    def name(self) -> str:
        return f"distance_to_neighbor_{self._k}"

    def dimension(self) -> int:
        return 1

    def requires_neighbors(self) -> bool:
        return True

    def default_k(self) -> int | None:
        return self._k

    def default_radius(self) -> float | None:
        return None


__all__ = [
    "HeightFeatureComputer",
    "RelativeHeightFeatureComputer",
    "DensityFeatureComputer",
    "DistanceToNeighborFeatureComputer",
]
