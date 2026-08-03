"""
topocore.analysis.statistics.manager
=====================================

Unified statistics analysis facade.

Provides a single entry point for computing statistics using any
supported method.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    StatisticsConfig,
)
from topocore.analysis.exceptions import StatisticsError
from topocore.analysis.protocols import (
    GriddedSurface,
    PointCloudData,
    TriangulatedSurface,
)
from topocore.analysis.types import (
    AreaStats,
    DensityStats,
    DistributionStats,
    ElevationStats,
    SlopeStats,
)

from .area import AreaStatistics
from .density import DensityStatistics
from .distribution import DistributionStatistics
from .elevation import ElevationStatistics
from .slope import SlopeStatistics

StatisticsResult: TypeAlias = AreaStats | DensityStats | DistributionStats | ElevationStats | SlopeStats


DispatchMethod: TypeAlias = Callable[..., StatisticsResult]


_VALID_METHODS = frozenset(
    {
        "elevation",
        "slope",
        "area",
        "density",
        "distribution",
    }
)


class StatisticsMethod:
    """
    Supported statistics methods.
    """

    ELEVATION = "elevation"
    SLOPE = "slope"
    AREA = "area"
    DENSITY = "density"
    DISTRIBUTION = "distribution"


class StatisticsAnalysis:
    """
    Unified statistics analysis manager.

    Parameters
    ----------
    config
        Statistics-specific configuration.
    """

    __slots__ = (
        "_config",
        "_method",
        "_num_bins",
        "_dispatch",
    )

    def __init__(
        self,
        *,
        config: StatisticsConfig | None = None,
        method: str | None = None,
    ) -> None:

        self._config = config or DEFAULT_ANALYSIS_CONFIG.statistics

        self._method = method or self._config.default_method

        if self._method not in _VALID_METHODS:
            raise StatisticsError(f"Unknown statistics method: '{self._method}'.")

        self._num_bins = self._config.histogram_bins

        self._dispatch: dict[str, DispatchMethod] = {
            StatisticsMethod.ELEVATION: self.elevation,
            StatisticsMethod.SLOPE: self.slope,
            StatisticsMethod.AREA: self.area,
            StatisticsMethod.DENSITY: self.density,
            StatisticsMethod.DISTRIBUTION: self.distribution,
        }

    @property
    def method(self) -> str:
        """
        Configured statistics method.
        """
        return self._method

    @property
    def config(self) -> StatisticsConfig:
        """
        Statistics configuration.
        """
        return self._config

    def elevation(
        self,
        values: (NDArray[np.float64] | PointCloudData | GriddedSurface),
    ) -> ElevationStats:
        """
        Compute elevation statistics.

        Accepts:

        - NumPy elevation array.
        - Point cloud data.
        - Gridded surface.
        """

        if isinstance(values, np.ndarray):
            return ElevationStatistics.compute(
                values.astype(
                    np.float64,
                    copy=False,
                )
            )

        if hasattr(values, "elevation_array"):
            return ElevationStatistics.compute(values.elevation_array)

        if hasattr(values, "elevations"):
            return ElevationStatistics.compute(values.elevations)

        raise StatisticsError("Unsupported elevation data source.")

    def slope(
        self,
        dtm: GriddedSurface,
        num_bins: int | None = None,
    ) -> SlopeStats:
        """
        Compute slope statistics from a DTM.
        """

        bins = num_bins if num_bins is not None else self._num_bins

        return SlopeStatistics(
            num_bins=bins,
        ).compute_from_dtm(dtm)

    def area(
        self,
        tin: TriangulatedSurface,
    ) -> AreaStats:
        """
        Compute area statistics from a triangulated surface.

        Parameters
        ----------
        tin
            Triangulated terrain surface.
        """

        return AreaStatistics.compute(tin)

    def density(
        self,
        points: NDArray[np.float64],
        resolution: float = 1.0,
    ) -> DensityStats:
        """
        Compute point density statistics.
        """

        return DensityStatistics(
            resolution=resolution,
        ).compute(points)

    def distribution(
        self,
        values: NDArray[np.float64],
        num_bins: int | None = None,
    ) -> DistributionStats:
        """
        Compute distribution statistics.
        """

        bins = num_bins if num_bins is not None else self._num_bins

        return DistributionStatistics(
            num_bins=bins,
        ).compute(
            values.astype(
                np.float64,
                copy=False,
            )
        )

    def percentiles(
        self,
        values: NDArray[np.float64],
        percentiles: list[float],
    ) -> dict[float, float]:
        """
        Compute percentile values.
        """

        return DistributionStatistics(
            num_bins=self._num_bins,
        ).percentiles(
            values.astype(
                np.float64,
                copy=False,
            ),
            percentiles,
        )

    def compute(
        self,
        *args: Any,
        method: str | None = None,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Compute statistics using selected method.
        """

        target = method or self._method

        try:
            handler = self._dispatch[target]
        except KeyError as exc:
            raise StatisticsError(f"Unknown statistics method: '{target}'.") from exc

        return handler(
            *args,
            **kwargs,
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> StatisticsResult:
        """
        Execute configured analysis.
        """

        return self.compute(
            *args,
            **kwargs,
        )


__all__ = [
    "StatisticsAnalysis",
    "StatisticsMethod",
]
