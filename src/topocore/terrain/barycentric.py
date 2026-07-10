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

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import InterpolationError
from topocore.terrain.tin import TIN
from topocore.terrain.weights import (
    barycentric_weights,
    inside_triangle,
)


class BarycentricInterpolator:
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
            raise InterpolationError(
                "Point lies outside the TIN."
            )

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
            raise InterpolationError(
                "Point lies outside triangle."
            )

        w1, w2, w3 = weights

        return (
            w1 * p1.z
            + w2 * p2.z
            + w3 * p3.z
        )

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