"""
topocore.analysis.statistics.elevation
======================================

Elevation statistics.

Computes descriptive statistics from elevation values.

Supports:

- Point cloud elevations.
- TIN vertex elevations.
- DTM raster values.
- Generic NumPy elevation arrays.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.protocols import (
    GriddedSurface,
    PointCloudData,
)
from topocore.analysis.types import ElevationStats


class ElevationStatistics:
    """
    Computes descriptive elevation statistics.
    """

    __slots__ = ()

    @staticmethod
    def compute(
        values: NDArray[np.float64],
    ) -> ElevationStats:
        """
        Compute elevation statistics.

        Parameters
        ----------
        values
            One-dimensional or raster elevation array.

        Returns
        -------
        ElevationStats
            Minimum, maximum, mean, standard deviation,
            median, range, and valid value count.

        Raises
        ------
        StatisticsError
            If no finite elevation values exist.
        """
        valid = values[np.isfinite(values)]

        if valid.size == 0:
            raise StatisticsError("No finite elevation values to compute statistics.")

        minimum = float(np.min(valid))

        maximum = float(np.max(valid))

        mean = float(np.mean(valid))

        std = float(np.std(valid))

        median = float(np.median(valid))

        return ElevationStats(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            median=median,
            range=maximum - minimum,
            count=int(valid.size),
        )

    @staticmethod
    def from_tin(
        tin: PointCloudData,
    ) -> ElevationStats:
        """
        Compute elevation statistics from a TIN-like object.

        Parameters
        ----------
        tin
            Object exposing an elevation array.
        """
        return ElevationStatistics.compute(tin.elevation_array)

    @staticmethod
    def from_dtm(
        dtm: GriddedSurface,
    ) -> ElevationStats:
        """
        Compute elevation statistics from a DTM.

        Parameters
        ----------
        dtm
            Gridded terrain surface.
        """
        return ElevationStatistics.compute(dtm.elevations)

    def __call__(
        self,
        values: NDArray[np.float64],
    ) -> ElevationStats:
        """
        Execute elevation statistics.
        """
        return self.compute(values)


__all__ = [
    "ElevationStatistics",
]
