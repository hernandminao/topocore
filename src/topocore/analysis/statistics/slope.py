"""
topocore.analysis.statistics.slope
===================================

Slope statistics.

Computes terrain slope angles in degrees from elevation
surfaces and provides descriptive statistics including
histogram distribution.

The slope is calculated from elevation gradients using
finite differences.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.protocols import GriddedSurface
from topocore.analysis.types import SlopeStats

_MIN_GRID_SIZE: Final[int] = 3


class SlopeStatistics:
    """
    Computes slope statistics from terrain elevation data.

    Parameters
    ----------
    num_bins
        Number of histogram bins.
    """

    __slots__ = ("_num_bins",)

    def __init__(
        self,
        num_bins: int = 10,
    ) -> None:
        if num_bins < 1:
            raise StatisticsError("Number of bins must be at least 1.")

        self._num_bins = int(num_bins)

    @property
    def num_bins(self) -> int:
        """
        Number of histogram bins.
        """
        return self._num_bins

    def compute_from_dtm(
        self,
        dtm: GriddedSurface,
    ) -> SlopeStats:
        """
        Compute slope statistics from a DTM.

        Parameters
        ----------
        dtm
            Gridded terrain surface.
        """
        return self.compute_from_array(
            dtm.elevations,
            dtm.resolution,
        )

    def compute_from_array(
        self,
        elevations: NDArray[np.float64],
        resolution: float,
    ) -> SlopeStats:
        """
        Compute slope statistics from elevation values.

        Parameters
        ----------
        elevations
            2D elevation grid.
        resolution
            Grid cell size.

        Returns
        -------
        SlopeStats
            Statistical summary of slope angles.
        """
        slope_values = self._compute_slope_array(
            elevations,
            resolution,
        )

        valid = slope_values[np.isfinite(slope_values)]

        if valid.size == 0:
            raise StatisticsError("No valid slope values computed.")

        return self._build_statistics(valid)

    def slope_array(
        self,
        elevations: NDArray[np.float64],
        resolution: float,
    ) -> NDArray[np.float64]:
        """
        Compute slope angle array in degrees.

        Parameters
        ----------
        elevations
            2D elevation grid.
        resolution
            Grid cell size.

        Returns
        -------
        NDArray[np.float64]
            Slope angles in degrees.
        """
        return self._compute_slope_array(
            elevations,
            resolution,
        )

    def _build_statistics(
        self,
        values: NDArray[np.float64],
    ) -> SlopeStats:
        """
        Build slope statistics from valid values.
        """
        counts, bin_edges = np.histogram(
            values,
            bins=self._num_bins,
        )

        return SlopeStats(
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            histogram_bins=[float(value) for value in bin_edges],
            histogram_counts=[int(value) for value in counts],
        )

    @staticmethod
    def _compute_slope_array(
        elevations: NDArray[np.float64],
        resolution: float,
    ) -> NDArray[np.float64]:
        """
        Compute slope angles using finite differences.
        """

        if resolution <= 0:
            raise StatisticsError("Resolution must be positive.")

        if elevations.ndim != 2:
            raise StatisticsError("Elevation array must be 2D.")

        rows, cols = elevations.shape

        if rows < _MIN_GRID_SIZE or cols < _MIN_GRID_SIZE:
            raise StatisticsError("Elevation array must be at least 3x3.")

        dy, dx = np.gradient(
            elevations,
            resolution,
        )

        gradient: NDArray[np.float64] = np.hypot(
            dx,
            dy,
        )

        gradient[~np.isfinite(elevations)] = np.nan

        slope_radians: NDArray[np.float64] = np.arctan(gradient)

        slope_degrees: NDArray[np.float64] = np.degrees(slope_radians)

        return slope_degrees.astype(
            np.float64,
            copy=False,
        )

    def __call__(
        self,
        dtm: GriddedSurface,
    ) -> SlopeStats:
        """
        Execute slope statistics.
        """
        return self.compute_from_dtm(dtm)


__all__ = [
    "SlopeStatistics",
]
