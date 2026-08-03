"""
topocore.analysis.statistics.density
=====================================

Point density statistics.

Computes spatial density statistics from point coordinates
using a regular XY grid.

Provides density distribution metrics and the resulting
density map.

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

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.protocols import PointCloudData
from topocore.analysis.types import DensityStats


class DensityStatistics:
    """
    Computes point density statistics over a regular grid.

    Parameters
    ----------
    resolution
        Grid cell size in coordinate units.
    """

    __slots__ = ("_resolution",)

    def __init__(
        self,
        resolution: float = 1.0,
    ) -> None:
        if resolution <= 0:
            raise StatisticsError("Resolution must be positive.")

        self._resolution = float(resolution)

    @property
    def resolution(self) -> float:
        """
        Grid resolution.
        """
        return self._resolution

    def compute(
        self,
        points: list[tuple[float, float]] | NDArray[np.float64],
    ) -> DensityStats:
        """
        Compute density statistics from XY coordinates.

        Parameters
        ----------
        points
            XY or XYZ coordinates.

        Returns
        -------
        DensityStats
            Density metrics and density grid.

        Raises
        ------
        StatisticsError
            If input data is empty.
        """
        coords = self._to_xy_array(points)

        if coords.shape[0] == 0:
            raise StatisticsError("No points to compute density.")

        min_x = float(np.min(coords[:, 0]))
        min_y = float(np.min(coords[:, 1]))
        max_x = float(np.max(coords[:, 0]))
        max_y = float(np.max(coords[:, 1]))

        rows, cols = self._grid_shape(
            min_x,
            min_y,
            max_x,
            max_y,
        )

        density_map = np.zeros(
            (rows, cols),
            dtype=np.float64,
        )

        self._fill_density_map(
            density_map,
            coords,
            min_x,
            min_y,
        )

        occupied_cells = density_map[density_map > 0]

        if occupied_cells.size == 0:
            raise StatisticsError("No cells contain points.")

        densities = occupied_cells / (self._resolution * self._resolution)

        return DensityStats(
            mean_density=float(np.mean(densities)),
            minimum_density=float(np.min(densities)),
            maximum_density=float(np.max(densities)),
            std_density=float(np.std(densities)),
            density_map=density_map,
        )

    def compute_from_tin(
        self,
        tin: PointCloudData,
    ) -> DensityStats:
        """
        Compute density statistics from point data.

        Parameters
        ----------
        tin
            Object providing XY coordinates.

        Notes
        -----
        The argument name is preserved for API compatibility.
        The implementation uses the exposed coordinate array.
        """
        return self.compute(tin.xy_array)

    def _grid_shape(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> tuple[int, int]:
        """
        Compute density grid dimensions.
        """
        columns = int(math.ceil((max_x - min_x) / self._resolution)) + 1

        rows = int(math.ceil((max_y - min_y) / self._resolution)) + 1

        return max(rows, 1), max(columns, 1)

    def _fill_density_map(
        self,
        density_map: NDArray[np.float64],
        coords: NDArray[np.float64],
        min_x: float,
        min_y: float,
    ) -> None:
        """
        Populate density grid using vectorized indexing.
        """
        columns = ((coords[:, 0] - min_x) / self._resolution).astype(np.intp)

        rows = ((coords[:, 1] - min_y) / self._resolution).astype(np.intp)

        np.clip(
            rows,
            0,
            density_map.shape[0] - 1,
            out=rows,
        )

        np.clip(
            columns,
            0,
            density_map.shape[1] - 1,
            out=columns,
        )

        np.add.at(
            density_map,
            (rows, columns),
            1.0,
        )

    @staticmethod
    def _to_xy_array(
        points: list[tuple[float, float]] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert input coordinates to XY array.
        """
        if isinstance(points, np.ndarray):
            array = np.asarray(
                points,
                dtype=np.float64,
            )

            if array.size == 0:
                return np.empty(
                    (0, 2),
                    dtype=np.float64,
                )

            if array.ndim == 2:
                if array.shape[1] < 2:
                    raise StatisticsError("Input array requires at least two columns.")

                return array[:, :2]

            if array.ndim == 1:
                if array.size % 2 != 0:
                    raise StatisticsError("Flat coordinate array must contain pairs.")

                return array.reshape((-1, 2))

            raise StatisticsError(f"Unsupported array shape: {array.shape}")

        array = np.asarray(
            points,
            dtype=np.float64,
        )

        if array.size == 0:
            return np.empty(
                (0, 2),
                dtype=np.float64,
            )

        if array.ndim != 2 or array.shape[1] < 2:
            raise StatisticsError("Points must contain XY coordinates.")

        return array[:, :2]

    def __call__(
        self,
        points: list[tuple[float, float]] | NDArray[np.float64],
    ) -> DensityStats:
        """
        Execute density computation.
        """
        return self.compute(points)


__all__ = [
    "DensityStatistics",
]
