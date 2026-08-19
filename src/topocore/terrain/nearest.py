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

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.base import BaseInterpolator
from topocore.terrain.tin import TIN


class NearestInterpolator(BaseInterpolator):
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

    def interpolate_many(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Vectorized nearest-neighbor interpolation.

        Ties resolve to the first vertex encountered in
        ``tin.vertices`` order, matching ``interpolate()``'s
        ``min()``-based scalar behavior exactly (Python's ``min()``
        and NumPy's ``argmin()`` both return the first occurrence of
        the minimum, not an arbitrary one).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        vertex_array = self._tin.vertex_array()
        vx, vy, vz = vertex_array[:, 0], vertex_array[:, 1], vertex_array[:, 2]

        dx = x_arr[:, None] - vx[None, :]
        dy = y_arr[:, None] - vy[None, :]
        distances = np.hypot(dx, dy)

        nearest_index = np.argmin(distances, axis=1)

        return vz[nearest_index]

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
