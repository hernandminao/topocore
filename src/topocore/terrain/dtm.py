"""
topocore.terrain.dtm
====================

Digital Terrain Model (DTM).

This module provides the public raster terrain model generated from a
TIN through one of the available interpolation methods.

The DTM is immutable and separates:

- Grid geometry
- Raster elevations
- Interpolation algorithm

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topocore.terrain.base import BaseDTM
from topocore.terrain.base import BaseInterpolator
from topocore.terrain.cell import Cell
from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster
from topocore.terrain.tin import TIN


@dataclass(
    frozen=True,
    slots=True,
)
class DTM(BaseDTM):
    """
    Digital Terrain Model.

    Parameters
    ----------
    tin
        Source terrain.
    grid
        Raster grid.
    raster
        Raster elevations.
    """

    tin: TIN

    grid: Grid

    raster: Raster

    @classmethod
    def from_tin(
        cls,
        tin: TIN,
        grid: Grid,
        interpolator: BaseInterpolator,
    ) -> DTM:
        """
        Generate a DTM from a TIN.

        Parameters
        ----------
        tin
            Source TIN.
        grid
            Raster grid.
        interpolator
            Elevation interpolator, already bound to ``tin``
            (see e.g. ``IDWInterpolator(tin)``).

        Returns
        -------
        DTM
        """
        values = np.full(
            grid.shape,
            np.nan,
            dtype=np.float64,
        )

        for row in range(grid.rows):
            for column in range(grid.columns):
                x, y = grid.coordinate(
                    row,
                    column,
                )

                values[
                    row,
                    column,
                ] = interpolator.interpolate(
                    x,
                    y,
                )

        raster = Raster(
            grid=grid,
            values=values,
        )

        return cls(
            tin=tin,
            grid=grid,
            raster=raster,
        )

    @property
    def rows(
        self,
    ) -> int:
        """
        Number of raster rows.
        """
        return self.grid.rows

    @property
    def columns(
        self,
    ) -> int:
        """
        Number of raster columns.
        """
        return self.grid.columns

    @property
    def elevations(
        self,
    ) -> np.ndarray:
        """
        Elevation matrix.
        """
        return self.raster.values

    @property
    def width(
        self,
    ) -> int:
        """
        Grid width, in columns.
        """
        return self.grid.columns

    @property
    def height(
        self,
    ) -> int:
        """
        Grid height, in rows.
        """
        return self.grid.rows

    @property
    def resolution(
        self,
    ) -> float:
        """
        Grid resolution.
        """
        return self.grid.resolution

    @property
    def bounds(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Raster bounds.
        """
        return self.grid.bounds

    def elevation(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Return terrain elevation.
        """
        return self.raster.elevation(
            x,
            y,
        )

    def cell(
        self,
        row: int,
        column: int,
    ) -> Cell:
        """
        Return a raster cell.
        """
        return self.raster.cell(
            row,
            column,
        )

    def statistics(
        self,
    ) -> dict[str, float | int]:
        """
        Return terrain statistics.
        """
        return self.raster.statistics()

    def array(
        self,
    ) -> NDArray[np.float64]:
        """
        Return raster elevations.
        """
        return self.raster.array()

    def __getitem__(
        self,
        index: tuple[int, int],
    ) -> float:
        """
        Array-like access.
        """
        return self.raster[index]

    def __iter__(
        self,
    ) -> Iterator[Cell]:
        """
        Iterate over raster cells.
        """
        yield from self.raster

    def __len__(
        self,
    ) -> int:
        """
        Number of raster cells.
        """
        return self.raster.size

    def __repr__(
        self,
    ) -> str:
        """
        String representation.
        """
        return f"DTM({self.rows}x{self.columns}, resolution={self.resolution})"


__all__ = [
    "DTM",
]
