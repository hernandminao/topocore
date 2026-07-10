"""
topocore.terrain.interpolation
==============================

Terrain interpolation facade.

Provides a unified API for all interpolation methods.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import Enum

from topocore.geometry.point3d import Point3D
from topocore.terrain.barycentric import BarycentricInterpolator
from topocore.terrain.idw import IDWInterpolator
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.nearest import NearestInterpolator
from topocore.terrain.tin import TIN


class InterpolationMethod(str, Enum):
    """
    Supported interpolation methods.
    """

    LINEAR = "linear"
    BARYCENTRIC = "barycentric"
    IDW = "idw"
    NEAREST = "nearest"


class TerrainInterpolator:
    """
    High-level interpolation interface.

    Examples
    --------
    >>> interpolator = TerrainInterpolator(tin)
    >>> z = interpolator.interpolate(100.0, 200.0)

    >>> interpolator.method = InterpolationMethod.IDW
    >>> z = interpolator.interpolate(100.0, 200.0)
    """

    __slots__ = (
        "_linear",
        "_barycentric",
        "_idw",
        "_nearest",
        "_method",
    )

    def __init__(
        self,
        tin: TIN,
        *,
        method: InterpolationMethod = InterpolationMethod.LINEAR,
        power: float = 2.0,
    ) -> None:

        self._linear = LinearInterpolator(tin)
        self._barycentric = BarycentricInterpolator(tin)
        self._idw = IDWInterpolator(
            tin,
            power=power,
        )
        self._nearest = NearestInterpolator(tin)

        self._method = method

    @property
    def method(
        self,
    ) -> InterpolationMethod:
        return self._method

    @method.setter
    def method(
        self,
        value: InterpolationMethod,
    ) -> None:
        self._method = value

    @property
    def tin(
        self,
    ) -> TIN:
        return self._linear.tin

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate elevation.
        """
        if self._method is InterpolationMethod.LINEAR:
            return self._linear.interpolate(x, y)

        if self._method is InterpolationMethod.BARYCENTRIC:
            return self._barycentric.interpolate(x, y)

        if self._method is InterpolationMethod.IDW:
            return self._idw.interpolate(x, y)

        return self._nearest.interpolate(x, y)

    def interpolate_point(
        self,
        point: Point3D,
    ) -> float:
        return self.interpolate(
            point.x,
            point.y,
        )

    def __call__(
        self,
        x: float,
        y: float,
    ) -> float:
        return self.interpolate(
            x,
            y,
        )


__all__ = [
    "InterpolationMethod",
    "TerrainInterpolator",
]