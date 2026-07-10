"""
topocore.terrain.linear
=======================

Linear interpolation over a TIN.

Linear interpolation on a TIN is mathematically equivalent to
barycentric interpolation. This class provides a semantic interface
for users who explicitly require linear interpolation while internally
reusing the barycentric implementation.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geometry.point3d import Point3D
from topocore.terrain.barycentric import BarycentricInterpolator
from topocore.terrain.tin import TIN


class LinearInterpolator:
    """
    Linear interpolator over a TIN.

    Notes
    -----
    Since each TIN triangle defines a plane, linear interpolation is
    exactly equivalent to barycentric interpolation.
    """

    __slots__ = ("_interpolator",)

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
        self._interpolator = BarycentricInterpolator(
            tin,
        )

    @property
    def tin(
        self,
    ) -> TIN:
        """
        Return the terrain model.
        """
        return self._interpolator.tin

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate the elevation at a coordinate.

        Parameters
        ----------
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        float
            Interpolated elevation.
        """
        return self._interpolator.interpolate(
            x,
            y,
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
            Interpolated elevation.
        """
        return self._interpolator.interpolate_triangle(
            triangle_index,
            x,
            y,
        )

    def interpolate_point(
        self,
        point: Point3D,
    ) -> float:
        """
        Interpolate the elevation of a point.

        Parameters
        ----------
        point
            Query point.

        Returns
        -------
        float
            Interpolated elevation.
        """
        return self._interpolator.interpolate_point(
            point,
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
    "LinearInterpolator",
]
