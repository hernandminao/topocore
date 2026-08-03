"""
topocore.processing.filters.base
================================

Abstract base classes for point cloud filters.

This module defines the interfaces for point cloud filters:
- Filters that produce a boolean mask (keep/discard)
- Filters that transform the point cloud (modify points)
- Filters that extract a subset of points

All filters follow a consistent pattern and can be chained together
in processing pipelines.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import (
    build_cloud_from_mask,
    flatten_attributes,
)
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.types import BoolArray1D


class Filter(ABC):
    """
    Abstract base class for point cloud filters.

    Filters process a point cloud and produce either:
    - a boolean classification mask
    - a transformed point cloud
    """

    @abstractmethod
    def mask(
        self,
        cloud: PointCloud,
    ) -> BoolArray1D:
        """
        Compute a boolean mask.

        True values indicate points that are kept.
        """

    @abstractmethod
    def apply(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Apply the filter and return a new point cloud.
        """

    @abstractmethod
    def name(self) -> str:
        """
        Return the filter identifier.
        """

    def __call__(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Callable interface.

        Equivalent to calling ``apply``.
        """
        return self.apply(cloud)


class BinaryFilter(Filter):
    """
    Abstract base class for filters based on boolean masks.

    Subclasses only need to implement ``mask``.
    The point cloud reconstruction is handled automatically.
    """

    @override
    @abstractmethod
    def mask(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> BoolArray1D:
        """
        Compute the binary keep mask.
        """

    @override
    def apply(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Apply the binary filter.

        Returns
        -------
        PointCloud
            Point cloud containing only selected points.
        """

        mask = self.mask(cloud)

        flattened = flatten_attributes(cloud)

        return build_cloud_from_mask(
            flattened,
            mask,
        )


__all__ = [
    "Filter",
    "BinaryFilter",
]
