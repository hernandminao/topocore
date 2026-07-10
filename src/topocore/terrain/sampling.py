"""
topocore.terrain.sampling
=========================

Raster sampling utilities.

This module provides common sampling algorithms used by terrain
processing modules.

Sampling is independent from interpolation. Interpolation computes
unknown elevations from a TIN, while sampling retrieves values from an
already generated raster.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.terrain.cell import Cell
from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster

_COORD_OUTSIDE_RASTER_MSG = "Coordinate outside raster."


class RasterSampler:
    """
    Raster sampling utilities.
    """

    @staticmethod
    def nearest(
        raster: Raster,
        x: float,
        y: float,
    ) -> float:
        """
        Sample the nearest raster cell.

        Parameters
        ----------
        raster
            Source raster.
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the coordinate lies outside the raster.
        """
        if not raster.contains(x, y):
            raise ValueError(_COORD_OUTSIDE_RASTER_MSG)

        row = raster.grid.row(y)
        column = raster.grid.column(x)

        return raster.value(
            row,
            column,
        )

    @staticmethod
    def cell(
        raster: Raster,
        x: float,
        y: float,
    ) -> Cell:
        """
        Return the nearest raster cell.
        """
        if not raster.contains(x, y):
            raise ValueError(_COORD_OUTSIDE_RASTER_MSG)

        row = raster.grid.row(y)
        column = raster.grid.column(x)

        return raster.cell(
            row,
            column,
        )

    @staticmethod
    def index(
        grid: Grid,
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """
        Convert coordinates to raster indices.

        Parameters
        ----------
        grid
            Grid definition.
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        tuple[int, int]
        """
        if not grid.contains(x, y):
            raise ValueError("Coordinate outside grid.")

        return (
            grid.row(y),
            grid.column(x),
        )

    @staticmethod
    def coordinate(
        grid: Grid,
        row: int,
        column: int,
    ) -> tuple[float, float]:
        """
        Convert indices into coordinates.
        """
        return grid.coordinate(
            row,
            column,
        )

    @staticmethod
    def bilinear(
        raster: Raster,
        x: float,
        y: float,
    ) -> float:
        """
        Bilinear raster sampling.

        Parameters
        ----------
        raster
            Source raster.
        x
            X coordinate.
        y
            Y coordinate.

        Returns
        -------
        float

        Notes
        -----
        Returns NaN whenever one of the four neighbouring cells
        contains NoData.
        """
        grid = raster.grid

        if not grid.contains(x, y):
            raise ValueError(_COORD_OUTSIDE_RASTER_MSG)

        fx = (x - grid.min_x) / grid.resolution

        fy = (y - grid.min_y) / grid.resolution

        c0 = int(math.floor(fx))
        r0 = int(math.floor(fy))

        c1 = min(
            c0 + 1,
            grid.columns - 1,
        )

        r1 = min(
            r0 + 1,
            grid.rows - 1,
        )

        z00 = raster.value(r0, c0)
        z10 = raster.value(r0, c1)
        z01 = raster.value(r1, c0)
        z11 = raster.value(r1, c1)

        if not all(
            math.isfinite(v)
            for v in (
                z00,
                z10,
                z01,
                z11,
            )
        ):
            return float("nan")

        dx = fx - c0
        dy = fy - r0

        z0 = z00 * (1.0 - dx) + z10 * dx

        z1 = z01 * (1.0 - dx) + z11 * dx

        return z0 * (1.0 - dy) + z1 * dy


__all__ = [
    "RasterSampler",
]
