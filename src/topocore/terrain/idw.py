"""
topocore.terrain.idw
====================

Inverse Distance Weighting (IDW) interpolation.

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
from topocore.terrain.exceptions import InterpolationError
from topocore.terrain.tin import TIN


class IDWInterpolator:
    """
    Inverse Distance Weighting interpolator.

    Notes
    -----
    Uses all terrain vertices. Later PRs may introduce
    KD-Tree acceleration without changing the public API.
    """

    __slots__ = (
        "_tin",
        "_power",
        "_epsilon",
    )

    def __init__(
        self,
        tin: TIN,
        *,
        power: float = 2.0,
        epsilon: float = 1e-12,
    ) -> None:
        if power <= 0.0:
            raise ValueError("power must be greater than zero.")

        self._tin = tin
        self._power = float(power)
        self._epsilon = float(epsilon)

    @property
    def tin(
        self,
    ) -> TIN:
        return self._tin

    @property
    def power(
        self,
    ) -> float:
        return self._power

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate elevation using inverse-distance weighting.
        """
        weighted_sum = 0.0
        weight_sum = 0.0

        for point in self._tin.vertices:
            dx = point.x - x
            dy = point.y - y

            distance = math.hypot(
                dx,
                dy,
            )

            if distance <= self._epsilon:
                return point.z

            weight = 1.0 / (distance**self._power)

            weighted_sum += weight * point.z
            weight_sum += weight

        if math.isclose(weight_sum, 0.0):
            raise InterpolationError("Interpolation failed.")

        return weighted_sum / weight_sum

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
    "IDWInterpolator",
]
