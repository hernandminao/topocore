"""
topocore.processing.filters.statistical
=======================================

Statistical Outlier Removal (SOR) filter.

This filter removes points whose average distance to their k nearest
neighbors is beyond a threshold (mean + std_ratio * standard deviation)
of the global distribution of average distances.

This is one of the most commonly used outlier removal filters in
point cloud processing.

References
----------
- Rusu et al. (2008). Towards 3D Point Cloud Based Object Maps
  for Household Environments. Robotics and Autonomous Systems.

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


class StatisticalOutlierFilter(BinaryFilter):
    """
    Statistical Outlier Removal (SOR) filter.

    This filter removes points whose average distance to their
    k nearest neighbors is statistically anomalous.

    Parameters
    ----------
    k
        Number of neighbors for distance computation.
    std_ratio
        Number of standard deviations from the mean to use as threshold.
    min_points
        Minimum number of points that must remain after filtering.
        If filtering would remove too many points, the original cloud
        is returned (with a warning, or error if min_points is strict).

    Examples
    --------
    >>> filter = StatisticalOutlierFilter(k=8, std_ratio=1.0)
    >>> filtered_cloud = filter.apply(cloud)
    """

    __slots__ = (
        "_k",
        "_std_ratio",
        "_min_points",
        "_strict",
    )

    def __init__(
        self,
        k: int = 8,
        std_ratio: float = 1.0,
        min_points: int = 3,
        strict: bool = False,
    ) -> None:
        if k < 1:
            raise FilterError(f"k must be at least 1, got {k}.")
        if std_ratio <= 0:
            raise FilterError(f"std_ratio must be positive, got {std_ratio}.")
        if min_points < 0:
            raise FilterError(f"min_points cannot be negative, got {min_points}.")

        self._k = k
        self._std_ratio = std_ratio
        self._min_points = min_points
        self._strict = strict

    @override
    def mask(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> BoolArray1D:
        """Compute the keep mask using statistical outlier detection."""
        if cloud.is_empty:
            raise FilterError("Cannot filter an empty point cloud.")

        # Build neighbor index
        if manager is None:
            manager = NeighborhoodManager.from_point_cloud(
                cloud,
            )

        # For each point, compute average distance to k neighbors
        _, distances = manager.knn_many(k=self._k, include_self=False)

        # Average distance for each point
        mean_distances = distances.mean(axis=1)

        # Compute global mean and standard deviation
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()

        # Compute threshold
        threshold = global_mean + self._std_ratio * global_std

        # Keep points with average distance <= threshold
        mask = mean_distances <= threshold

        # Ensure minimum points are kept
        if mask.sum() < self._min_points:
            if self._strict:
                raise FilterError(f"Filter would keep only {mask.sum()} points, below minimum {self._min_points}.")
            # Return all points (no filtering)
            mask = np.ones(len(cloud), dtype=bool)

        return mask

    @override
    def name(self) -> str:
        return f"statistical_outlier(k={self._k}, std_ratio={self._std_ratio})"


__all__ = [
    "StatisticalOutlierFilter",
]
