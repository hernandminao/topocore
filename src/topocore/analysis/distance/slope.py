"""
topocore.analysis.distance.slope
=================================

Slope distance computations.

Computes spatial distance, slope gradients and slope angles.

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

from topocore.analysis.distance.euclidean import EuclideanDistance
from topocore.analysis.exceptions import DistanceError
from topocore.analysis.types import DistanceResult


class SlopeDistance:
    """
    Computes slope distances between points.

    Represents the 3D distance considering horizontal displacement
    and elevation difference.
    """

    __slots__ = ()

    _ENGINE = EuclideanDistance(dimension="3d")

    @property
    def method(self) -> str:
        """Distance calculation method."""
        return "slope"

    @staticmethod
    def compute(
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> DistanceResult:
        """
        Compute slope distance between two 3D points.
        """

        SlopeDistance._validate(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

        result = SlopeDistance._ENGINE.compute(
            x1,
            y1,
            x2,
            y2,
            z1,
            z2,
        )

        return DistanceResult(
            value=result.value,
            method="slope",
            dimension="3d",
        )

    @staticmethod
    def compute_many(
        points_a: NDArray[np.float64],
        points_b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute slope distances between point arrays.
        """

        if points_a.shape != points_b.shape:
            raise DistanceError("Point arrays must have identical shapes.")

        if points_a.ndim != 2 or points_a.shape[1] != 3:
            raise DistanceError("Slope distance requires arrays with shape (N,3).")

        if not (np.isfinite(points_a).all() and np.isfinite(points_b).all()):
            raise DistanceError("Point arrays contain invalid values.")

        delta = points_b - points_a

        return np.sqrt(np.sum(delta * delta, axis=1))

    @staticmethod
    def from_horizontal_and_vertical(
        horizontal: float,
        vertical: float,
    ) -> DistanceResult:
        """
        Compute slope distance from horizontal and vertical components.
        """

        SlopeDistance._validate_components(
            horizontal,
            vertical,
        )

        return DistanceResult(
            value=float(
                math.hypot(
                    horizontal,
                    vertical,
                )
            ),
            method="slope",
            dimension="3d",
        )

    @staticmethod
    def slope_gradient(
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> float:
        """
        Compute slope gradient percentage.
        """

        SlopeDistance._validate(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

        horizontal = math.hypot(
            x2 - x1,
            y2 - y1,
        )

        if math.isclose(horizontal, 0.0, abs_tol=1e-12):
            raise DistanceError("Horizontal distance is zero; slope gradient is undefined.")

        return float(abs(z2 - z1) / horizontal * 100.0)

    @staticmethod
    def slope_angle(
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> float:
        """
        Compute slope angle in degrees.
        """

        SlopeDistance._validate(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

        horizontal = math.hypot(
            x2 - x1,
            y2 - y1,
        )

        vertical = abs(z2 - z1)

        return float(
            math.degrees(
                math.atan2(
                    vertical,
                    horizontal,
                )
            )
        )

    def __call__(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> DistanceResult:
        """Compute slope distance."""
        return self.compute(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

    @staticmethod
    def _validate(
        *values: float,
    ) -> None:
        """Validate coordinates."""

        for value in values:
            if not isinstance(value, (int, float)):
                raise DistanceError(f"Coordinate must be numeric; got {type(value)}.")

            if not math.isfinite(float(value)):
                raise DistanceError(f"Coordinate must be finite; got {value}.")

    @staticmethod
    def _validate_components(
        horizontal: float,
        vertical: float,
    ) -> None:
        """Validate distance components."""

        for value in (horizontal, vertical):
            if not isinstance(value, (int, float)):
                raise DistanceError("Distance components must be numeric.")

            if not math.isfinite(float(value)):
                raise DistanceError("Distance components must be finite.")

            if value < 0:
                raise DistanceError("Distance components cannot be negative.")


__all__ = [
    "SlopeDistance",
]
