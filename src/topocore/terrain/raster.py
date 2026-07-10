"""
topocore.terrain.raster
=======================

Raster terrain model.

This module defines the immutable raster used by Digital Terrain Models
(DTM). The raster stores elevation values independently of the grid
geometry.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from topocore.terrain.cell import Cell
from topocore.terrain.grid import Grid


@dataclass(frozen=True, slots=True)
class Raster:
    """
    Immutable raster.

    Parameters
    ----------
    grid
        Grid definition.
    values
        Elevation array.
    nodata
        NoData value.
    """

    grid: Grid

    values: NDArray[np.float64]

    nodata: float = float("nan")

    def __post_init__(self) -> None:
        expected = self.grid.shape

        if self.values.shape != expected:
            raise ValueError("Raster dimensions do not match grid.")

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        """
        Raster shape.
        """
        return self.values.shape

    @property
    def rows(
        self,
    ) -> int:
        """
        Number of rows.
        """
        return self.shape[0]

    @property
    def columns(
        self,
    ) -> int:
        """
        Number of columns.
        """
        return self.shape[1]

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

    @property
    def size(
        self,
    ) -> int:
        """
        Number of raster cells.
        """
        return self.values.size

    @property
    def minimum(
        self,
    ) -> float:
        """
        Minimum elevation.
        """
        return float(
            np.nanmin(
                self.values,
            )
        )

    @property
    def maximum(
        self,
    ) -> float:
        """
        Maximum elevation.
        """
        return float(
            np.nanmax(
                self.values,
            )
        )

    @property
    def mean(
        self,
    ) -> float:
        """
        Mean elevation.
        """
        return float(
            np.nanmean(
                self.values,
            )
        )

    @property
    def std(
        self,
    ) -> float:
        """
        Elevation standard deviation.
        """
        return float(
            np.nanstd(
                self.values,
            )
        )

    @property
    def nodata_count(
        self,
    ) -> int:
        """
        Number of NoData cells.
        """
        return int(
            np.count_nonzero(
                np.isnan(
                    self.values,
                )
            )
        )

    @property
    def valid_count(
        self,
    ) -> int:
        """
        Number of valid cells.
        """
        return self.size - self.nodata_count

    @property
    def extent(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Alias of bounds.
        """
        return self.bounds

    @property
    def cell_size(
        self,
    ) -> float:
        """
        Raster cell size.
        """
        return self.grid.resolution

    @property
    def transform(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        """
        GDAL affine transform.

        Returns
        -------
        tuple
            (origin_x, pixel_width, rotation_x,
             origin_y, rotation_y, pixel_height)
        """
        return (
            self.grid.min_x,
            self.grid.resolution,
            0.0,
            self.grid.max_y,
            0.0,
            -self.grid.resolution,
        )

    def copy(
        self,
    ) -> Raster:
        """
        Return a deep copy.
        """
        return Raster(
            grid=self.grid,
            values=self.values.copy(),
            nodata=self.nodata,
        )

    def is_nodata(
        self,
        row: int,
        column: int,
    ) -> bool:
        """
        Return whether a cell contains NoData.
        """
        return bool(
            np.isnan(
                self.values[
                    row,
                    column,
                ]
            )
        )

    def fill(
        self,
        value: float,
    ) -> Raster:
        """
        Replace NoData values.

        Returns
        -------
        Raster
        """
        values = self.values.copy()

        values[np.isnan(values)] = value

        return Raster(
            grid=self.grid,
            values=values,
            nodata=self.nodata,
        )

    def mask(
        self,
        mask: NDArray[np.bool_],
    ) -> Raster:
        """
        Apply a boolean mask.

        False cells become NoData.
        """
        if mask.shape != self.shape:
            raise ValueError("Mask dimensions do not match raster.")

        values = self.values.copy()

        values[~mask] = np.nan

        return Raster(
            grid=self.grid,
            values=values,
            nodata=self.nodata,
        )

    def valid_cells(
        self,
    ):
        """
        Iterate over valid cells.
        """
        for row in range(self.rows):
            for column in range(self.columns):
                if self.is_nodata(
                    row,
                    column,
                ):
                    continue

                yield self.cell(
                    row,
                    column,
                )

    def window(
        self,
        row: int,
        column: int,
        radius: int = 1,
    ) -> NDArray[np.float64]:
        """
        Return a neighborhood window.

        Parameters
        ----------
        row
            Center row.
        column
            Center column.
        radius
            Window radius.

        Returns
        -------
        ndarray
        """
        r0 = max(
            row - radius,
            0,
        )

        r1 = min(
            row + radius + 1,
            self.rows,
        )

        c0 = max(
            column - radius,
            0,
        )

        c1 = min(
            column + radius + 1,
            self.columns,
        )

        return self.values[
            r0:r1,
            c0:c1,
        ].copy()

    def value(
        self,
        row: int,
        column: int,
    ) -> float:
        """
        Return a cell elevation.
        """
        return float(
            self.values[
                row,
                column,
            ]
        )

    def cell(
        self,
        row: int,
        column: int,
    ) -> Cell:
        """
        Return a raster cell.
        """
        x, y = self.grid.coordinate(
            row,
            column,
        )

        return Cell(
            row=row,
            column=column,
            x=x,
            y=y,
            z=self.value(
                row,
                column,
            ),
        )

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Return whether a coordinate lies inside the raster.
        """
        return self.grid.contains(
            x,
            y,
        )

    def elevation(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Return the nearest raster elevation.
        """
        row = self.grid.row(y)
        column = self.grid.column(x)

        return self.value(
            row,
            column,
        )

    def array(
        self,
    ) -> NDArray[np.float64]:
        """
        Return a defensive copy.
        """
        return self.values.copy()

    def statistics(
        self,
    ) -> dict[str, float | int]:
        """
        Return raster statistics.
        """
        return {
            "rows": self.rows,
            "columns": self.columns,
            "cells": self.size,
            "valid": self.valid_count,
            "nodata": self.nodata_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
        }

    def __getitem__(
        self,
        index: tuple[int, int],
    ) -> float:
        """
        Array-like access.
        """
        row, column = index

        return self.value(
            row,
            column,
        )

    def __iter__(
        self,
    ):
        """
        Iterate over raster cells.
        """
        for row in range(self.rows):
            for column in range(self.columns):
                yield self.cell(
                    row,
                    column,
                )


__all__ = [
    "Raster",
]
