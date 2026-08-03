"""
topocore.processing.filters.pass_through
========================================

Pass-through filter.

This filter keeps points whose coordinate along a specified axis falls
within a given range. It is commonly used to crop a point cloud to
a specific region of interest (ROI) before further processing.

The filter supports all three axes: X, Y, Z.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import Enum
from typing import override

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import flatten_attributes
from topocore.processing.exceptions import FilterError, ProcessingError
from topocore.processing.neighbors.manager import NeighborhoodManager
from topocore.processing.types import BoolArray1D

from .base import BinaryFilter


class Axis(Enum):
    """Coordinate axis for pass-through filtering."""

    X = "x"
    Y = "y"
    Z = "z"


_AXIS_TO_ATTRIBUTE = {
    Axis.X: PointAttribute.X,
    Axis.Y: PointAttribute.Y,
    Axis.Z: PointAttribute.Z,
}


class PassThroughFilter(BinaryFilter):
    """
    Pass-through filter.

    Keeps points whose coordinate along the specified axis is within
    the inclusive range [min_value, max_value].

    Parameters
    ----------
    axis
        Coordinate axis (X, Y, or Z).
    min_value
        Minimum coordinate value (inclusive).
    max_value
        Maximum coordinate value (inclusive).

    Examples
    --------
    >>> filter = PassThroughFilter(Axis.Z, min_value=0.0, max_value=100.0)
    >>> filtered_cloud = filter.apply(cloud)
    """

    __slots__ = (
        "_axis",
        "_min_value",
        "_max_value",
        "_attribute",
    )

    def __init__(
        self,
        axis: Axis,
        min_value: float,
        max_value: float,
    ) -> None:
        if min_value > max_value:
            raise FilterError(f"min_value ({min_value}) must be <= max_value ({max_value}).")

        self._axis = axis
        self._min_value = min_value
        self._max_value = max_value
        self._attribute = _AXIS_TO_ATTRIBUTE[axis]

    @override
    def mask(
        self,
        cloud: PointCloud,
        *,
        manager: NeighborhoodManager | None = None,
    ) -> BoolArray1D:
        """
        Compute the keep mask for the pass-through filter.

        Parameters
        ----------
        cloud:
            Input point cloud.

        Returns
        -------
        BoolArray1D
            Boolean mask indicating points inside the selected range.

        Raises
        ------
        FilterError
            If the cloud is empty or the required coordinate is missing.
        """
        del manager

        if cloud.is_empty:
            raise FilterError("Cannot filter an empty point cloud.")

        if self._attribute not in cloud.attributes:
            raise FilterError(f"Point cloud has no {self._attribute.value} attribute.")

        try:
            flattened = flatten_attributes(
                cloud,
            )

        except ProcessingError as exc:
            raise FilterError("Unable to flatten point cloud attributes.") from exc

        values = flattened[self._attribute]

        if values.shape[0] != cloud.point_count:
            raise FilterError(
                f"Attribute '{self._attribute.value}' size mismatch: {values.shape[0]} != {cloud.point_count}."
            )

        if not np.isfinite(values).all():
            raise FilterError(f"Attribute '{self._attribute.value}' contains NaN or Inf values.")

        return np.asarray(
            (values >= self._min_value) & (values <= self._max_value),
            dtype=np.bool_,
        )

    @override
    def name(self) -> str:
        return f"pass_through({self._axis.value}, {self._min_value}, {self._max_value})"

    @property
    def axis(self) -> Axis:
        return self._axis

    @property
    def min_value(self) -> float:
        return self._min_value

    @property
    def max_value(self) -> float:
        return self._max_value


__all__ = [
    "Axis",
    "PassThroughFilter",
]
