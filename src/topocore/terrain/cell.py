"""
topocore.terrain.cell
=====================

Raster cell model.

Represents a single cell of a Digital Terrain Model (DTM).

Cells are immutable and contain both raster indices and their
corresponding spatial coordinates.

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


@dataclass(
    frozen=True,
    slots=True,
)
class Cell:
    """
    Raster cell.

    Parameters
    ----------
    row
        Row index.
    column
        Column index.
    x
        X coordinate.
    y
        Y coordinate.
    z
        Elevation.
    """

    row: int
    column: int

    x: float
    y: float
    z: float

    @property
    def has_data(
        self,
    ) -> bool:
        """
        Return whether the cell contains a valid elevation.
        """
        return math.isfinite(self.z)

    @property
    def is_nodata(
        self,
    ) -> bool:
        """
        Return whether the cell represents NoData.
        """
        return not self.has_data

    @property
    def xy(
        self,
    ) -> tuple[
        float,
        float,
    ]:
        """
        Return XY coordinates.
        """
        return (
            self.x,
            self.y,
        )

    @property
    def xyz(
        self,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """
        Return XYZ coordinates.
        """
        return (
            self.x,
            self.y,
            self.z,
        )

    def distance_to(
        self,
        other: Cell,
    ) -> float:
        """
        Compute the planar distance to another cell.

        Parameters
        ----------
        other
            Target cell.

        Returns
        -------
        float
        """
        return math.hypot(
            other.x - self.x,
            other.y - self.y,
        )

    def translated(
        self,
        *,
        row: int | None = None,
        column: int | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> Cell:
        """
        Return a copy with updated values.

        Parameters
        ----------
        row
            New row.
        column
            New column.
        x
            New X coordinate.
        y
            New Y coordinate.
        z
            New elevation.

        Returns
        -------
        Cell
        """
        return Cell(
            row=self.row if row is None else row,
            column=self.column if column is None else column,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            z=self.z if z is None else z,
        )

    def __iter__(
        self,
    ):
        """
        Iterate over the cell values.

        Yields
        ------
        int | float
        """
        yield self.row
        yield self.column
        yield self.x
        yield self.y
        yield self.z


__all__ = [
    "Cell",
]