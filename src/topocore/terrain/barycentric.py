"""
topocore.terrain.barycentric
============================

Barycentric interpolation over a TIN.

This interpolator performs exact linear interpolation inside the
triangle containing the query point.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.geometry.point3d import Point3D
from topocore.terrain.base import BaseInterpolator
from topocore.terrain.exceptions import InterpolationError
from topocore.terrain.tin import TIN
from topocore.terrain.weights import (
    barycentric_weights,
    inside_triangle,
)


class BarycentricInterpolator(BaseInterpolator):
    """
    Exact barycentric interpolation over a TIN.
    """

    __slots__ = ("_tin",)

    def __init__(
        self,
        tin: TIN,
    ) -> None:
        """
        Parameters
        ----------
        tin
            Terrain model.
        """
        self._tin = tin

    @property
    def tin(
        self,
    ) -> TIN:
        """
        Return the terrain model.
        """
        return self._tin

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate an elevation.

        Parameters
        ----------
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        float

        Raises
        ------
        InterpolationError
            If the point lies outside the TIN.
        """
        triangle_index = self._tin.find_triangle(
            x,
            y,
        )

        if triangle_index < 0:
            raise InterpolationError("Point lies outside the TIN.")

        return self.interpolate_triangle(
            triangle_index,
            x,
            y,
        )

    def interpolate_point(
        self,
        point: Point3D,
    ) -> float:
        """
        Interpolate using a Point3D.
        """
        return self.interpolate(
            point.x,
            point.y,
        )

    def interpolate_triangle(
        self,
        triangle_index: int,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate inside a known triangle.

        Parameters
        ----------
        triangle_index
            Triangle index.
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        float

        Raises
        ------
        InterpolationError
            If barycentric coordinates are invalid.
        """
        p1, p2, p3 = self._tin.triangle_vertices(
            triangle_index,
        )

        weights = barycentric_weights(
            x,
            y,
            p1,
            p2,
            p3,
        )

        if not inside_triangle(weights):
            raise InterpolationError("Point lies outside triangle.")

        w1, w2, w3 = weights

        return w1 * p1.z + w2 * p2.z + w3 * p3.z

    def interpolate_many(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Interpolate elevation for many query points.

        Unlike ``NearestInterpolator``/``IDWInterpolator``, this is
        a plain Python loop over ``interpolate()``, not a vectorized
        NumPy computation -- a deliberate, documented decision, not
        an oversight. ``TIN.find_triangle()`` is a brute-force
        O(triangle_count) scan with no spatial index (see its own
        docstring); vectorizing barycentric evaluation would still
        need to point-locate every query point one at a time against
        every triangle first, so there is no broadcasting shortcut
        available here the way there is for Nearest/IDW (which only
        ever need vertex coordinates, not triangle containment).
        Accelerating ``find_triangle()`` itself with a spatial index
        is explicitly deferred to PR20 (Optimization); this loop
        will benefit automatically once that lands, without any
        change needed here.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        result = np.empty(x_arr.shape[0], dtype=np.float64)

        for i in range(x_arr.shape[0]):
            result[i] = self.interpolate(float(x_arr[i]), float(y_arr[i]))

        return result

    def __call__(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Callable interface.
        """
        return self.interpolate(
            x,
            y,
        )


__all__ = [
    "BarycentricInterpolator",
]
