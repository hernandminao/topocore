"""
topocore.processing.filters.clip_polygon
========================================

Clip Polygon filter.

This filter keeps points whose XY projection falls inside a 2D polygon.
The polygon is defined by a list of vertices (x, y). This is useful for
extracting points within an irregular region of interest (ROI).

The implementation uses a vectorized ray-casting algorithm for the
point-in-polygon test.

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
from numpy.typing import NDArray

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import flatten_attributes
from topocore.processing.exceptions import FilterError
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.types import BoolArray1D

from .base import BinaryFilter

type FloatArray1D = NDArray[np.float64]


def point_in_polygon(
    x: FloatArray1D,
    y: FloatArray1D,
    polygon_x: FloatArray1D,
    polygon_y: FloatArray1D,
) -> BoolArray1D:
    """
    Vectorized point-in-polygon test using ray casting.
    """

    vertex_count = polygon_x.size

    if vertex_count < 3:
        raise FilterError("Polygon must contain at least three vertices.")

    inside = np.zeros(
        x.shape[0],
        dtype=np.bool_,
    )

    previous = vertex_count - 1

    for current in range(vertex_count):
        xi = polygon_x[current]
        yi = polygon_y[current]

        xj = polygon_x[previous]
        yj = polygon_y[previous]

        crosses = (yi > y) != (yj > y)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi

        inside ^= crosses & (x < x_at_y)

        previous = current

    return inside


class ClipPolygonFilter(BinaryFilter):
    """
    Clip Polygon filter.

    Keeps points whose XY projection falls inside a 2D polygon.
    """

    __slots__ = (
        "_polygon_x",
        "_polygon_y",
    )

    def __init__(
        self,
        polygon: FloatArray1D | NDArray[np.float64],
    ) -> None:

        polygon_array = np.asarray(
            polygon,
            dtype=np.float64,
        )

        if polygon_array.ndim == 1:
            polygon_array = polygon_array.reshape(
                -1,
                2,
            )

        if polygon_array.ndim != 2 or polygon_array.shape[1] != 2:
            raise FilterError(f"polygon must have shape (M, 2), got {polygon_array.shape}.")

        if polygon_array.shape[0] < 3:
            raise FilterError("polygon must have at least 3 vertices.")

        self._polygon_x = np.asarray(
            polygon_array[:, 0],
            dtype=np.float64,
        )

        self._polygon_y = np.asarray(
            polygon_array[:, 1],
            dtype=np.float64,
        )

    @override
    def mask(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> BoolArray1D:
        """
        Compute the keep mask for the clip polygon filter.
        """
        del manager

        if cloud.is_empty:
            raise FilterError("Cannot filter an empty point cloud.")

        required = {
            PointAttribute.X,
            PointAttribute.Y,
        }

        if not required.issubset(cloud.attributes):
            raise FilterError("Point cloud must contain X/Y coordinates.")

        try:
            flattened = flatten_attributes(cloud)

        except Exception as exc:
            raise FilterError("Unable to flatten point cloud attributes.") from exc

        x = flattened[PointAttribute.X]

        y = flattened[PointAttribute.Y]

        return point_in_polygon(
            x,
            y,
            self._polygon_x,
            self._polygon_y,
        )

    @override
    def name(
        self,
    ) -> str:
        return f"clip_polygon({self._polygon_x.size} vertices)"

    @property
    def polygon(
        self,
    ) -> NDArray[np.float64]:
        """
        Return polygon vertices.
        """

        return np.stack(
            [
                self._polygon_x,
                self._polygon_y,
            ],
            axis=1,
        )


__all__ = [
    "ClipPolygonFilter",
    "point_in_polygon",
]
