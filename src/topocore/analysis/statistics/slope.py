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

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.protocols import GriddedSurface
from topocore.analysis.types import SlopeStats

_MIN_GRID_SIZE: Final[int] = 3

# Relative tolerance for treating a histogram's own min/max range as
# numerically degenerate. Confirmed directly: a genuinely uniform
# slope (e.g. a perfectly flat ramp) computed via
# `_compute_slope_array()`'s own floating-point arithmetic
# (np.gradient -> np.hypot -> np.arctan -> np.degrees) does NOT
# produce bit-identical values across the grid -- it produces values
# differing by ~1e-15, floating-point rounding noise from computing
# the same mathematical result via slightly different operation
# orders at different grid positions. `np.histogram()` only widens a
# degenerate range automatically when min == max EXACTLY (confirmed
# directly); a range this tiny but nonzero instead makes it fail with
# "Cannot create N finite-sized bins" -- the real, reproducible defect
# this constant exists to fix. 1e-9 is deliberately far above
# realistic floating-point noise (~1e-15 relative to values in the
# 1-100 range) and far below any slope difference that could be
# considered a genuine, meaningful variation.
_HISTOGRAM_RANGE_EPSILON: Final[float] = 1e-9


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
        minimum = float(np.min(values))
        maximum = float(np.max(values))

        # `np.histogram()`'s own default `range` (the data's own
        # min/max) only self-widens when the two are EXACTLY equal --
        # see `_HISTOGRAM_RANGE_EPSILON`'s own docstring for why a
        # genuinely uniform slope's real min/max essentially never
        # are, despite being numerically degenerate for histogram
        # purposes. Widened explicitly here only for the histogram's
        # own bin edges -- `minimum`/`maximum` below are reported
        # exactly as computed, unwidened; only the histogram binning
        # range is stabilized, never the statistics themselves.
        histogram_minimum = minimum
        histogram_maximum = maximum
        span = maximum - minimum
        scale = max(abs(minimum), abs(maximum), 1.0)

        if math.isclose(span, 0.0, rel_tol=0.0, abs_tol=_HISTOGRAM_RANGE_EPSILON * scale):
            delta = max(_HISTOGRAM_RANGE_EPSILON * scale, np.finfo(float).eps)
            histogram_minimum -= delta
            histogram_maximum += delta

        counts, bin_edges = np.histogram(
            values,
            bins=self._num_bins,
            range=(histogram_minimum, histogram_maximum),
        )

        return SlopeStats(
            minimum=minimum,
            maximum=maximum,
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
