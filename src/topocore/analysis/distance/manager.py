"""
topocore.analysis.distance.manager
===================================

Unified distance analysis facade.

Provides a single entry point for all supported
distance computations.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    DistanceConfig,
)
from topocore.analysis.exceptions import DistanceError
from topocore.analysis.protocols import CRSType
from topocore.analysis.types import DistanceResult

from .euclidean import EuclideanDistance
from .geodesic import GeodesicDistance
from .horizontal import HorizontalDistance
from .slope import SlopeDistance
from .vertical import VerticalDistance


class DistanceMethod(StrEnum):
    """Supported distance methods."""

    EUCLIDEAN = "euclidean"
    GEODESIC = "geodesic"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    SLOPE = "slope"


class DistanceAnalysis:
    """
    Unified distance computation manager.
    """

    __slots__ = (
        "_config",
        "_method",
        "_euclidean_2d",
        "_euclidean_3d",
        "_horizontal",
        "_vertical",
        "_slope",
        "_geodesic",
    )

    def __init__(
        self,
        *,
        config: DistanceConfig | None = None,
        method: str | None = None,
        crs: CRSType | None = None,
    ) -> None:

        self._config = config or DEFAULT_ANALYSIS_CONFIG.distance

        self._method = self._resolve_method(method or self._config.default_method)

        self._euclidean_2d = EuclideanDistance(dimension="2d")

        self._euclidean_3d = EuclideanDistance(dimension="3d")

        self._horizontal = HorizontalDistance()
        self._vertical = VerticalDistance()
        self._slope = SlopeDistance()

        self._geodesic = None

        if self._method is DistanceMethod.GEODESIC:
            if crs is None:
                raise DistanceError("CRS is required for geodesic calculations.")

            self._geodesic = GeodesicDistance(crs)

    @property
    def method(self) -> str:
        """Current distance method."""
        return self._method.value

    @property
    def config(self) -> DistanceConfig:
        """Distance configuration."""
        return self._config

    @property
    def geodesic(self) -> GeodesicDistance | None:
        """Geodesic calculator."""
        return self._geodesic

    @staticmethod
    def available_methods() -> tuple[str, ...]:
        """Return supported distance methods."""
        return tuple(method.value for method in DistanceMethod)

    def compute(
        self,
        *args: float,
        method: str | None = None,
    ) -> DistanceResult:
        """
        Compute distance.
        """

        selected = self._resolve_method(method or self._method.value)

        if selected is DistanceMethod.EUCLIDEAN:
            if len(args) == 4:
                return self._euclidean_2d.compute(*args)

            if len(args) == 6:
                return self._euclidean_3d.compute(*args)

            raise DistanceError("Euclidean requires 4 or 6 coordinates.")

        if selected is DistanceMethod.GEODESIC:
            if self._geodesic is None:
                raise DistanceError("Geodesic calculator not initialized.")

            return self._geodesic.compute(*args)

        if selected is DistanceMethod.HORIZONTAL:
            return self._horizontal.compute(*args)

        if selected is DistanceMethod.VERTICAL:
            return self._vertical.compute(*args)

        if selected is DistanceMethod.SLOPE:
            return self._slope.compute(*args)

        raise DistanceError(f"Unsupported method: {selected}")

    def gradient(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> float:
        """Compute slope gradient percentage."""

        return SlopeDistance.slope_gradient(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

    def angle(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> float:
        """Compute slope angle degrees."""

        return SlopeDistance.slope_angle(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

    def __call__(
        self,
        *args: float,
        method: str | None = None,
    ) -> DistanceResult:
        """Shortcut for compute."""

        return self.compute(
            *args,
            method=method,
        )

    @staticmethod
    def _resolve_method(
        method: str,
    ) -> DistanceMethod:
        """Convert string method into enum."""

        try:
            return DistanceMethod(method)

        except ValueError as exc:
            raise DistanceError(
                f"Unsupported distance method '{method}'. Available: {DistanceAnalysis.available_methods()}"
            ) from exc


__all__ = [
    "DistanceAnalysis",
    "DistanceMethod",
]
