"""
topocore.terrain.nearest
========================

Nearest-neighbor interpolation.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


class NearestInterpolator:
    """
    Nearest-neighbor interpolation.
    """

    __slots__ = ("_tin",)

    def __init__(
        self,
        tin: TIN,
    ) -> None:
        self._tin = tin

    @property
    def tin(
        self,
    ) -> TIN:
        return self._tin

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Return the elevation of the nearest vertex.
        """
        nearest = min(
            self._tin.vertices,
            key=lambda point: math.hypot(
                point.x - x,
                point.y - y,
            ),
        )

        return nearest.z

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
    "NearestInterpolator",
]
