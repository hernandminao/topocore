"""
topocore.analysis.distance.geodesic
====================================

Geodesic distance computations over an ellipsoid.

Provides analysis-level geodesic operations while delegating
ellipsoidal calculations to TopoCore geodesy services.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from typing import Any, cast

from topocore.analysis.exceptions import DistanceError
from topocore.analysis.protocols import CRSType
from topocore.analysis.types import DistanceResult


class GeodesicDistance:
    """
    Computes geodesic distances between geographic coordinates.

    Coordinates are expressed as longitude / latitude degrees.
    """

    __slots__ = ("_calculator",)

    def __init__(
        self,
        crs: CRSType,
    ) -> None:

        try:
            from topocore.geodesy.geodesic import (
                GeodesicCalculator,
            )

            self._calculator = GeodesicCalculator(cast(Any, crs))

        except Exception as exc:
            raise DistanceError("Failed to initialize geodesic calculator.") from exc

    @property
    def calculator(self) -> Any:
        """
        Return underlying geodesic calculator.

        This property is intentionally kept dynamic because
        implementations may vary depending on the geodesy backend.
        """
        return self._calculator

    def compute(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> DistanceResult:
        """
        Compute ellipsoidal distance.
        """

        self._validate_coordinates(
            lon1,
            lat1,
            lon2,
            lat2,
        )

        try:
            value = float(
                self._calculator.distance(
                    lon1,
                    lat1,
                    lon2,
                    lat2,
                )
            )

        except Exception as exc:
            raise DistanceError("Geodesic distance computation failed.") from exc

        return DistanceResult(
            value=value,
            method="geodesic",
            dimension="geodetic",
        )

    def azimuth(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> float:
        """
        Compute forward azimuth.
        """

        self._validate_coordinates(
            lon1,
            lat1,
            lon2,
            lat2,
        )

        try:
            return float(
                self._calculator.azimuth(
                    lon1,
                    lat1,
                    lon2,
                    lat2,
                )
            )

        except Exception as exc:
            raise DistanceError("Geodesic azimuth computation failed.") from exc

    def inverse(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> tuple[float, float, float]:
        """
        Compute inverse geodesic solution.
        """

        self._validate_coordinates(
            lon1,
            lat1,
            lon2,
            lat2,
        )

        try:
            result = self._calculator.inverse(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            return (
                float(result[0]),
                float(result[1]),
                float(result[2]),
            )

        except Exception as exc:
            raise DistanceError("Geodesic inverse computation failed.") from exc

    def polygon_area(
        self,
        lons: list[float],
        lats: list[float],
    ) -> float:
        """
        Compute geodesic polygon area.
        """

        if len(lons) != len(lats):
            raise DistanceError("Longitude and latitude lists must have equal length.")

        if len(lons) < 3:
            raise DistanceError("Polygon requires at least three vertices.")

        for lon, lat in zip(lons, lats):
            self._validate_coordinate(
                lon,
                lat,
            )

        try:
            return float(
                self._calculator.polygon_area(
                    lons,
                    lats,
                )
            )

        except Exception as exc:
            raise DistanceError("Geodesic polygon area computation failed.") from exc

    def __call__(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> DistanceResult:
        """Compute geodesic distance."""

        return self.compute(
            lon1,
            lat1,
            lon2,
            lat2,
        )

    @staticmethod
    def _validate_coordinates(
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> None:
        """Validate geographic coordinates."""

        GeodesicDistance._validate_coordinate(
            lon1,
            lat1,
        )

        GeodesicDistance._validate_coordinate(
            lon2,
            lat2,
        )

    @staticmethod
    def _validate_coordinate(
        lon: float,
        lat: float,
    ) -> None:
        """Validate longitude and latitude."""

        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise DistanceError("Coordinates must be finite.")

        if not -180.0 <= lon <= 180.0:
            raise DistanceError(f"Longitude out of range: {lon}")

        if not -90.0 <= lat <= 90.0:
            raise DistanceError(f"Latitude out of range: {lat}")


__all__ = [
    "GeodesicDistance",
]
