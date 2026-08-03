"""
topocore.processing.filters.crop_box
====================================

Crop Box filter.

This filter keeps points that lie inside an axis-aligned 3D bounding
box (AABB). The box is defined by its minimum and maximum coordinates
along X, Y, and Z axes.

This is a generalization of the pass-through filter applied to all
three axes simultaneously.

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

from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.types import BoolArray1D

from .base import BinaryFilter


class CropBoxFilter(BinaryFilter):
    """
    Crop Box filter.

    Keeps points that lie inside an axis-aligned 3D bounding box.

    Parameters
    ----------
    box
        Bounding box defining the region to keep.

    Examples
    --------
    >>> box = BBox3D(min_x=0, min_y=0, min_z=0, max_x=100, max_y=100, max_z=50)
    >>> filter = CropBoxFilter(box)
    >>> filtered_cloud = filter.apply(cloud)
    """

    __slots__ = ("_box",)

    def __init__(
        self,
        box: BBox3D,
    ) -> None:
        self._box = box

    @override
    def mask(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> BoolArray1D:
        """Compute the keep mask for the crop box filter."""

        del manager

        if cloud.is_empty:
            raise FilterError("Cannot filter an empty point cloud.")

        required = {PointAttribute.X, PointAttribute.Y, PointAttribute.Z}
        if not required.issubset(cloud.attributes):
            raise FilterError("Point cloud has no X/Y/Z coordinates.")

        # Extract coordinates
        xs, ys, zs = [], [], []
        for chunk in cloud:
            xs.append(chunk[PointAttribute.X])
            ys.append(chunk[PointAttribute.Y])
            zs.append(chunk[PointAttribute.Z])

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)

        # Create mask
        return (
            (x >= self._box.min_x)
            & (x <= self._box.max_x)
            & (y >= self._box.min_y)
            & (y <= self._box.max_y)
            & (z >= self._box.min_z)
            & (z <= self._box.max_z)
        )

    @override
    def name(self) -> str:
        return f"crop_box({self._box.min_x}, {self._box.min_y}, {self._box.min_z}, {self._box.max_x}, {self._box.max_y}, {self._box.max_z})"

    @property
    def box(self) -> BBox3D:
        return self._box


__all__ = [
    "CropBoxFilter",
]
