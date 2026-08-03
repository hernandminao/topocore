"""
topocore.processing.filters.radius
==================================

Radius Outlier Removal (ROR) filter.

This filter removes points that have fewer than `min_neighbors` points
within a sphere of radius `radius`. This is effective for removing
isolated points and noise.

This filter is complementary to the Statistical Outlier Removal filter:
- SOR works on the distribution of distances (global)
- ROR works on local density (local)

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
from topocore.processing.exceptions import FilterError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import BoolArray1D

from .base import BinaryFilter


class RadiusOutlierFilter(BinaryFilter):
    """
    Radius Outlier Removal (ROR) filter.

    This filter removes points that have fewer than `min_neighbors`
    points within a sphere of radius `radius`.

    Parameters
    ----------
    radius
        Search radius.
    min_neighbors
        Minimum number of neighbors within the radius.
    include_self
        Whether to count the point itself as a neighbor.
    min_points
        Minimum number of points that must remain after filtering.

    Examples
    --------
    >>> filter = RadiusOutlierFilter(radius=1.0, min_neighbors=4)
    >>> filtered_cloud = filter.apply(cloud)
    """

    __slots__ = (
        "_radius",
        "_min_neighbors",
        "_include_self",
        "_min_points",
        "_strict",
    )

    def __init__(
        self,
        radius: float = 1.0,
        min_neighbors: int = 4,
        include_self: bool = True,
        min_points: int = 3,
        strict: bool = False,
    ) -> None:
        if radius <= 0:
            raise FilterError(f"radius must be positive, got {radius}.")
        if min_neighbors < 0:
            raise FilterError(f"min_neighbors cannot be negative, got {min_neighbors}.")
        if min_points < 0:
            raise FilterError(f"min_points cannot be negative, got {min_points}.")

        self._radius = radius
        self._min_neighbors = min_neighbors
        self._include_self = include_self
        self._min_points = min_points
        self._strict = strict

    def mask(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> BoolArray1D:
        """Compute the keep mask using radius outlier detection."""

        if cloud.is_empty:
            raise FilterError("Cannot filter an empty point cloud.")

        # Build neighbor index
        if manager is None:
            manager = NeighborhoodManager.from_point_cloud(cloud)

        # For each point, count neighbors within radius
        neighbor_lists = manager.radius_many(
            radius=self._radius,
            include_self=self._include_self,
        )

        # Count neighbors
        neighbor_counts = np.fromiter(
            (len(neighbors) for neighbors in neighbor_lists),
            dtype=np.int64,
            count=cloud.point_count,
        )

        # Keep points with sufficient neighbors
        mask = neighbor_counts >= self._min_neighbors

        # Ensure minimum points are kept
        if mask.sum() < self._min_points:
            if self._strict:
                raise FilterError(f"Filter would keep only {mask.sum()} points, below minimum {self._min_points}.")
            # Return all points (no filtering)
            mask = np.ones(len(cloud), dtype=bool)

        return mask

    @override
    def name(self) -> str:
        return f"radius_outlier(radius={self._radius}, min_neighbors={self._min_neighbors})"


__all__ = [
    "RadiusOutlierFilter",
]
