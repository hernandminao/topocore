"""
topocore.analysis.distance.euclidean
====================================

Euclidean distance computations in 2D and 3D.

Provides optimized Cartesian distance calculations for individual points
and vectorized point arrays.

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

from topocore.analysis.exceptions import DistanceError
from topocore.analysis.types import DistanceResult


class EuclideanDistance:
    """
    Computes Euclidean distances between points.

    Supports planar (2D) and spatial (3D) Cartesian distances.
    """

    __slots__ = ("_dimension",)

    def __init__(self, *, dimension: str = "3d") -> None:
        if dimension not in {"2d", "3d"}:
            raise DistanceError(f"Invalid dimension '{dimension}'. Expected '2d' or '3d'.")

        self._dimension = dimension

    @property
    def dimension(self) -> str:
        """Distance dimensionality."""
        return self._dimension

    @property
    def method(self) -> str:
        """Distance calculation method."""
        return "euclidean"

    def compute(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        z1: float = 0.0,
        z2: float = 0.0,
    ) -> DistanceResult:
        """Compute Euclidean distance between two points."""

        self._validate_coordinates(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

        dx = x2 - x1
        dy = y2 - y1

        if self._dimension == "2d":
            value = math.hypot(dx, dy)
        else:
            dz = z2 - z1
            value = math.sqrt(dx * dx + dy * dy + dz * dz)

        return DistanceResult(
            value=float(value),
            method=self.method,
            dimension=self._dimension,
        )

    def compute_many(
        self,
        points_a: NDArray[np.float64],
        points_b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute distances between two point arrays.

        Parameters
        ----------
        points_a
            Array with shape ``(N,2)`` or ``(N,3)``.
        points_b
            Array with shape ``(N,2)`` or ``(N,3)``.

        Returns
        -------
        NDArray[np.float64]
            Distance array.

        Raises
        ------
        DistanceError
            If arrays have invalid shape.
        """

        if points_a.shape != points_b.shape:
            raise DistanceError("Point arrays must have identical shapes.")

        if points_a.ndim != 2:
            raise DistanceError("Point arrays must be two-dimensional.")

        if points_a.shape[1] not in {2, 3}:
            raise DistanceError("Expected point arrays with 2 or 3 dimensions.")

        if not (np.isfinite(points_a).all() and np.isfinite(points_b).all()):
            raise DistanceError("Point arrays contain NaN or infinite values.")

        delta = points_b - points_a

        if self._dimension == "2d":
            return np.sqrt(np.sum(delta[:, :2] ** 2, axis=1))

        return np.sqrt(np.sum(delta[:, :3] ** 2, axis=1))

    def distance_between_points(
        self,
        p1: tuple[float, ...],
        p2: tuple[float, ...],
    ) -> DistanceResult:
        """Compute distance between point tuples."""

        if len(p1) != len(p2):
            raise DistanceError("Point dimensionality mismatch.")

        if len(p1) == 2:
            return self.compute(
                p1[0],
                p1[1],
                p2[0],
                p2[1],
            )

        if len(p1) == 3:
            return self.compute(
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                p1[2],
                p2[2],
            )

        raise DistanceError(f"Unsupported point dimension: {len(p1)}.")

    def __call__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        z1: float = 0.0,
        z2: float = 0.0,
    ) -> DistanceResult:
        """Compute distance."""
        return self.compute(
            x1,
            y1,
            x2,
            y2,
            z1,
            z2,
        )

    @staticmethod
    def _validate_coordinates(
        *values: float,
    ) -> None:
        """Validate coordinate values."""

        for value in values:
            if not isinstance(value, (int, float)):
                raise DistanceError(f"Coordinate must be numeric; got {type(value)}.")

            if not math.isfinite(float(value)):
                raise DistanceError(f"Coordinate must be finite; got {value}.")


__all__ = [
    "EuclideanDistance",
]
