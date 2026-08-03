"""
topocore.analysis.distance.vertical
====================================

Vertical distance computations.

Computes elevation differences between points.

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


class VerticalDistance:
    """
    Computes vertical distances between elevations.

    Returns absolute elevation separation.
    """

    __slots__ = ()

    @property
    def method(self) -> str:
        """Distance calculation method."""
        return "vertical"

    @staticmethod
    def compute(
        z1: float,
        z2: float,
    ) -> DistanceResult:
        """
        Compute absolute elevation difference.
        """

        VerticalDistance._validate(z1, z2)

        return DistanceResult(
            value=float(abs(z2 - z1)),
            method="vertical",
            dimension="1d",
        )

    @staticmethod
    def compute_many(
        elevations_a: NDArray[np.float64],
        elevations_b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute vertical distances between elevation arrays.

        Parameters
        ----------
        elevations_a
            First elevation array.
        elevations_b
            Second elevation array.

        Returns
        -------
        NDArray[np.float64]
            Absolute elevation differences.
        """

        if elevations_a.shape != elevations_b.shape:
            raise DistanceError("Elevation arrays must have identical shapes.")

        if not (np.isfinite(elevations_a).all() and np.isfinite(elevations_b).all()):
            raise DistanceError("Elevation arrays contain NaN or infinite values.")

        return np.abs(elevations_b - elevations_a)

    @staticmethod
    def elevation_difference(
        z1: float,
        z2: float,
    ) -> float:
        """
        Compute signed elevation difference.

        Positive values indicate the second point is higher.
        """

        VerticalDistance._validate(z1, z2)

        return float(z2 - z1)

    def __call__(
        self,
        z1: float,
        z2: float,
    ) -> DistanceResult:
        """Compute vertical distance."""
        return self.compute(z1, z2)

    @staticmethod
    def _validate(
        *values: float,
    ) -> None:
        """Validate elevation values."""

        for value in values:
            if not isinstance(value, (int, float)):
                raise DistanceError(f"Elevation must be numeric; got {type(value)}.")

            if not math.isfinite(float(value)):
                raise DistanceError(f"Elevation must be finite; got {value}.")


__all__ = [
    "VerticalDistance",
]
