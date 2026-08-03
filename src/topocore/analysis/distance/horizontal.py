"""
topocore.analysis.distance.horizontal
======================================

Horizontal (planar XY) distance computations.

Computes 2D Cartesian distances ignoring elevation.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.distance.euclidean import EuclideanDistance
from topocore.analysis.exceptions import DistanceError
from topocore.analysis.types import DistanceResult


class HorizontalDistance:
    """
    Computes horizontal (planar XY) distances between points.

    The Z coordinate is ignored.
    """

    __slots__ = ()

    _ENGINE: Final[EuclideanDistance] = EuclideanDistance(dimension="2d")

    @staticmethod
    def compute(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> DistanceResult:
        """
        Compute horizontal distance between two points.
        """

        result = HorizontalDistance._ENGINE.compute(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        return DistanceResult(
            value=result.value,
            method="horizontal",
            dimension="2d",
        )

    @staticmethod
    def compute_many(
        points_a: NDArray[np.float64],
        points_b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute horizontal distances between point arrays.

        Parameters
        ----------
        points_a
            Array ``(N,2)`` or ``(N,3)``.
        points_b
            Array ``(N,2)`` or ``(N,3)``.

        Returns
        -------
        NDArray[np.float64]
            Distance values.
        """

        if points_a.shape != points_b.shape:
            raise DistanceError("Point arrays must have identical shapes.")

        if points_a.ndim != 2:
            raise DistanceError("Point arrays must be two-dimensional.")

        if points_a.shape[1] < 2:
            raise DistanceError("Points require at least X and Y coordinates.")

        if not (np.isfinite(points_a).all() and np.isfinite(points_b).all()):
            raise DistanceError("Point arrays contain NaN or infinite values.")

        delta = points_b[:, :2] - points_a[:, :2]

        return np.sqrt(np.sum(delta * delta, axis=1))

    @staticmethod
    def distance_between_points(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> DistanceResult:
        """
        Compute horizontal distance between XY tuples.
        """

        if len(p1) != 2 or len(p2) != 2:
            raise DistanceError("Horizontal distance requires 2D points.")

        return HorizontalDistance.compute(
            p1[0],
            p1[1],
            p2[0],
            p2[1],
        )

    def __call__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> DistanceResult:
        """Compute horizontal distance."""
        return self.compute(
            x1,
            y1,
            x2,
            y2,
        )


__all__ = [
    "HorizontalDistance",
]
