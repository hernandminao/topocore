"""
topocore.terrain.grid
=====================

Regular grid definition used to generate Digital Terrain Models (DTMs).

The grid defines only the spatial geometry of the raster. It does not
store elevation values nor perform interpolation.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from topocore.geometry.point3d import Point3D
from topocore.terrain.cell import Cell
from topocore.terrain.validation import validate_resolution


@dataclass(frozen=True, slots=True)
class Grid:
    """
    Regular terrain grid.

    Parameters
    ----------
    min_x
        Minimum X coordinate.
    min_y
        Minimum Y coordinate.
    max_x
        Maximum X coordinate.
    max_y
        Maximum Y coordinate.
    resolution
        Cell size.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    resolution: float

    def __post_init__(self) -> None:
        validate_resolution(self.resolution)

        if self.max_x <= self.min_x:
            raise ValueError(
                "max_x must be greater than min_x."
            )

        if self.max_y <= self.min_y:
            raise ValueError(
                "max_y must be greater than min_y."
            )

    @property
    def width(self) -> float:
        """
        Grid width.
        """
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """
        Grid height.
        """
        return self.max_y - self.min_y

    @property
    def columns(self) -> int:
        """
        Number of columns.
        """
        return (
            math.ceil(
                self.width / self.resolution,
            )
            + 1
        )

    @property
    def rows(self) -> int:
        """
        Number of rows.
        """
        return (
            math.ceil(
                self.height / self.resolution,
            )
            + 1
        )

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        """
        Raster shape.

        Returns
        -------
        tuple
            (rows, columns)
        """
        return (
            self.rows,
            self.columns,
        )

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
        Grid bounds.
        """
        return (
            self.min_x,
            self.min_y,
            self.max_x,
            self.max_y,
        )

    @property
    def size(
        self,
    ) -> int:
        """
        Total number of cells.
        """
        return self.rows * self.columns

    def x(
        self,
        column: int,
    ) -> float:
        """
        Return the X coordinate of a column.
        """
        return (
            self.min_x
            + column * self.resolution
        )

    def y(
        self,
        row: int,
    ) -> float:
        """
        Return the Y coordinate of a row.
        """
        return (
            self.min_y
            + row * self.resolution
        )

    def coordinate(
        self,
        row: int,
        column: int,
    ) -> tuple[
        float,
        float,
    ]:
        """
        Return the coordinate of a cell.
        """
        return (
            self.x(column),
            self.y(row),
        )

    def point(
        self,
        row: int,
        column: int,
    ) -> Point3D:
        """
        Return the grid point.

        Elevation is initialized as NaN.
        """
        x, y = self.coordinate(
            row,
            column,
        )

        return Point3D(
            x=x,
            y=y,
            z=float("nan"),
        )

    def cell(
        self,
        row: int,
        column: int,
    ) -> Cell:
        """
        Create an empty cell.
        """
        x, y = self.coordinate(
            row,
            column,
        )

        return Cell(
            row=row,
            column=column,
            x=x,
            y=y,
            z=float("nan"),
        )

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Return whether a coordinate lies inside the grid.
        """
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
        )

    def row(
        self,
        y: float,
    ) -> int:
        """
        Compute the row index.
        """
        return int(
            round(
                (y - self.min_y)
                / self.resolution,
            )
        )

    def column(
        self,
        x: float,
    ) -> int:
        """
        Compute the column index.
        """
        return int(
            round(
                (x - self.min_x)
                / self.resolution,
            )
        )

    def index(
        self,
        row: int,
        column: int,
    ) -> int:
        """
        Return the flattened index.
        """
        return (
            row * self.columns
            + column
        )


__all__ = [
    "Grid",
]