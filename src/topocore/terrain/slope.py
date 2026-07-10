"""
topocore.terrain.slope
=======================

Slope computation over a Triangulated Irregular Network (TIN).

Slope is computed per triangle, directly from the plane defined by
its three vertices, rather than from a resampled raster. This gives
an exact value for each terrain facet instead of the finite-difference
approximation a grid-based (Horn's method) computation would
introduce -- the same approach used by professional TIN-based
earthwork engines.

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

from topocore.terrain._geometry import oriented_normal, slope_radians
from topocore.terrain.models import Triangle
from topocore.terrain.tin import TIN
from topocore.terrain.types import Slope


def triangle_slope(triangle: Triangle) -> Slope:
    """
    Compute the slope of a single triangle.

    Returns
    -------
    float
        Slope angle in degrees, in the range [0, 90], where 0 is
        horizontal and 90 is vertical.
    """
    normal = oriented_normal(triangle)

    return math.degrees(slope_radians(normal))


class SlopeCalculator:
    """
    Computes per-triangle slope over a TIN.
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

    def _triangle_at(
        self,
        index: int,
    ) -> Triangle:
        simplex = self._tin.simplices[index]
        vertices = self._tin.vertices

        return Triangle(
            vertices[int(simplex[0])],
            vertices[int(simplex[1])],
            vertices[int(simplex[2])],
        )

    def at(
        self,
        index: int,
    ) -> Slope:
        """
        Slope of a single TIN triangle, by index into ``tin.simplices``.
        """
        return triangle_slope(self._triangle_at(index))

    def compute(
        self,
    ) -> NDArray[np.float64]:
        """
        Slope for every triangle in the TIN.

        Returns
        -------
        ndarray
            One slope value per triangle, in degrees, in the same
            order as ``tin.simplices``.
        """
        simplices = self._tin.simplices

        values = np.empty(
            simplices.shape[0],
            dtype=np.float64,
        )

        for i in range(simplices.shape[0]):
            values[i] = triangle_slope(self._triangle_at(i))

        return values

    def __call__(
        self,
    ) -> NDArray[np.float64]:
        return self.compute()


__all__ = [
    "SlopeCalculator",
    "triangle_slope",
]
