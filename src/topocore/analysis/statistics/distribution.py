"""
topocore.analysis.statistics.distribution
==========================================

Distribution statistics.

Computes histogram and distribution shape metrics
(skewness and kurtosis) for scalar values.

Useful for analyzing elevations, slopes, intensities,
or any other terrain-related attribute.

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
from topocore.analysis.types import DistributionStats

_STD_EPSILON: Final[float] = 1e-15
_EXCESS_KURTOSIS_OFFSET: Final[float] = 3.0
_IQR_FACTOR: Final[float] = 1.5


class DistributionStatistics:
    """
    Computes distribution statistics including histogram and
    shape descriptors.

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

    def compute(
        self,
        values: NDArray[np.float64],
    ) -> DistributionStats:
        """
        Compute distribution statistics.

        Parameters
        ----------
        values
            Array of scalar values.

        Returns
        -------
        DistributionStats
            Histogram and distribution metrics.

        Raises
        ------
        StatisticsError
            If insufficient finite values exist.
        """
        valid = self._finite_values(values)

        if valid.size < 4:
            raise StatisticsError("At least 4 finite values are required for distribution analysis.")

        counts, bin_edges = np.histogram(
            valid,
            bins=self._num_bins,
        )

        total = float(valid.size)

        cumulative = np.cumsum(counts) / total

        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=1))

        if std <= _STD_EPSILON:
            skewness = 0.0
            kurtosis = 0.0
        else:
            centered = valid - mean

            third_moment = float(np.mean(centered**3))

            fourth_moment = float(np.mean(centered**4))

            skewness = third_moment / (std**3)

            kurtosis = fourth_moment / (std**4) - _EXCESS_KURTOSIS_OFFSET

        return DistributionStats(
            bins=[float(value) for value in bin_edges],
            counts=[int(value) for value in counts],
            cumulative=[float(value) for value in cumulative],
            skewness=float(skewness),
            kurtosis=float(kurtosis),
        )

    def percentiles(
        self,
        values: NDArray[np.float64],
        percentiles: list[float],
    ) -> dict[float, float]:
        """
        Compute percentile values.

        Parameters
        ----------
        values
            Array of scalar values.

        percentiles
            Requested percentiles in range [0,100].

        Returns
        -------
        dict[float, float]
            Percentile-value mapping.

        Raises
        ------
        StatisticsError
            If values are empty or percentile is invalid.
        """
        valid = self._finite_values(values)

        if valid.size == 0:
            raise StatisticsError("No finite values for percentile computation.")

        for percentile in percentiles:
            if percentile < 0.0 or percentile > 100.0:
                raise StatisticsError("Percentiles must be within [0,100].")

        return {
            float(percentile): float(
                np.percentile(
                    valid,
                    percentile,
                )
            )
            for percentile in percentiles
        }

    def quantile_ranges(
        self,
        values: NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Compute common quantile ranges.

        Returns
        -------
        dict[str,float]
            Q1, median, Q3, IQR and whisker limits.
        """
        valid = self._finite_values(values)

        if valid.size == 0:
            raise StatisticsError("No finite values.")

        q1 = float(np.percentile(valid, 25))

        q2 = float(np.percentile(valid, 50))

        q3 = float(np.percentile(valid, 75))

        iqr = q3 - q1

        return {
            "q1": q1,
            "q2_median": q2,
            "q3": q3,
            "iqr": iqr,
            "lower_whisker": max(
                float(np.min(valid)),
                q1 - _IQR_FACTOR * iqr,
            ),
            "upper_whisker": min(
                float(np.max(valid)),
                q3 + _IQR_FACTOR * iqr,
            ),
        }

    @staticmethod
    def _finite_values(
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return finite values only.
        """
        return values[np.isfinite(values)]

    def __call__(
        self,
        values: NDArray[np.float64],
    ) -> DistributionStats:
        """
        Execute distribution analysis.
        """
        return self.compute(values)


__all__ = [
    "DistributionStatistics",
]
