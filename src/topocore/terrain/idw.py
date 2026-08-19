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

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.base import BaseInterpolator
from topocore.terrain.exceptions import InterpolationError
from topocore.terrain.tin import TIN


class IDWInterpolator(BaseInterpolator):
    """
    Inverse Distance Weighting interpolator.

    Notes
    -----
    Uses all terrain vertices. Later PRs may introduce
    KD-Tree acceleration without changing the public API.
    """

    __slots__ = (
        "_epsilon",
        "_power",
        "_tin",
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

    def interpolate_many(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Vectorized inverse-distance-weighted interpolation.

        Matches ``interpolate()``'s scalar behavior exactly,
        including the exact-vertex-match short-circuit (returns that
        vertex's ``z`` directly, first match in ``tin.vertices``
        order, instead of a weighted average with an infinite
        weight) and raising ``InterpolationError`` when the total
        weight is zero (only possible with zero vertices).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        vertex_array = self._tin.vertex_array()
        vx, vy, vz = vertex_array[:, 0], vertex_array[:, 1], vertex_array[:, 2]

        dx = x_arr[:, None] - vx[None, :]
        dy = y_arr[:, None] - vy[None, :]
        distances = np.hypot(dx, dy)

        result = np.empty(x_arr.shape[0], dtype=np.float64)

        exact_mask = distances <= self._epsilon
        has_exact = exact_mask.any(axis=1)

        for row_index in np.nonzero(has_exact)[0]:
            first_match = int(np.argmax(exact_mask[row_index]))
            result[row_index] = vz[first_match]

        normal_rows = ~has_exact

        if np.any(normal_rows):
            weights = 1.0 / (distances[normal_rows] ** self._power)
            weighted_sum = (weights * vz[None, :]).sum(axis=1)
            weight_sum = weights.sum(axis=1)

            if np.any(np.isclose(weight_sum, 0.0)):
                raise InterpolationError("Interpolation failed.")

            result[normal_rows] = weighted_sum / weight_sum

        return result

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
