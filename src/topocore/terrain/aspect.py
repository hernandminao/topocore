"""
topocore.terrain.aspect
=========================

Aspect computation over a Triangulated Irregular Network (TIN).

Aspect is the compass direction a terrain facet faces: the
horizontal direction of steepest descent. As with slope, it is
computed per triangle directly from the facet's plane, at the same
resolution as the terrain model, instead of from a resampled raster.

Convention
----------
Aspect is measured in degrees, clockwise from North:

* 0    -> North
* 90   -> East
* 180  -> South
* 270  -> West

Flat triangles have no defined downslope direction and return
``nan``, consistent with the NoData convention used across the
Terrain module (see ``DEFAULT_NO_DATA`` in ``terrain.constants``).

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

from topocore.terrain._geometry import aspect_radians, oriented_normal
from topocore.terrain.models import Triangle
from topocore.terrain.tin import TIN
from topocore.terrain.types import Aspect


def triangle_aspect(triangle: Triangle) -> Aspect:
    """
    Compute the aspect of a single triangle.

    Returns
    -------
    float
        Compass bearing in degrees, in the range [0, 360), or
        ``nan`` if the triangle is flat.
    """
    normal = oriented_normal(triangle)
    bearing = aspect_radians(normal)

    if bearing is None:
        return float("nan")

    return math.degrees(bearing)


class AspectCalculator:
    """
    Computes per-triangle aspect over a TIN.
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
    ) -> Aspect:
        """
        Aspect of a single TIN triangle, by index into ``tin.simplices``.
        """
        return triangle_aspect(self._triangle_at(index))

    def compute(
        self,
    ) -> NDArray[np.float64]:
        """
        Aspect for every triangle in the TIN.

        Returns
        -------
        ndarray
            One aspect value per triangle, in degrees, in the same
            order as ``tin.simplices`` (``nan`` for flat facets).
        """
        simplices = self._tin.simplices

        values = np.empty(
            simplices.shape[0],
            dtype=np.float64,
        )

        for i in range(simplices.shape[0]):
            values[i] = triangle_aspect(self._triangle_at(i))

        return values

    def __call__(
        self,
    ) -> NDArray[np.float64]:
        return self.compute()


__all__ = [
    "AspectCalculator",
    "triangle_aspect",
]
