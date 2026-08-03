"""
topocore.analysis.profile.longitudinal
======================================

Longitudinal profile generation.

Generates an elevation profile along a straight axis defined by
two endpoints. The profile is sampled at regular intervals along
the axis by interpolating terrain elevations from a TIN or DTM.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.analysis._shared.surface import (
    interpolate_surface,
    validate_xy,
)
from topocore.analysis.exceptions import ProfileError
from topocore.analysis.protocols import TerrainSurface
from topocore.analysis.types import (
    ProfilePoint,
    ProfileResult,
    ProfileType,
)


class LongitudinalProfile:
    """
    Generates longitudinal profiles along a straight axis.

    Parameters
    ----------
    interval
        Distance between consecutive profile stations (meters).
    """

    __slots__ = ("_interval",)

    def __init__(
        self,
        interval: float = 1.0,
    ) -> None:

        if not math.isfinite(interval):
            raise ProfileError("Profile interval must be finite.")

        if interval <= 0:
            raise ProfileError("Profile interval must be positive.")

        self._interval = float(interval)

    @property
    def interval(self) -> float:
        """
        Station interval in meters.
        """
        return self._interval

    def generate(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        surface: TerrainSurface,
    ) -> ProfileResult:
        """
        Generate a longitudinal profile.

        Parameters
        ----------
        origin
            Starting XY coordinate.
        target
            Ending XY coordinate.
        surface
            Terrain surface implementing TerrainSurface.

        Returns
        -------
        ProfileResult
            Generated longitudinal profile.
        """

        validate_xy(
            origin[0],
            origin[1],
        )

        validate_xy(
            target[0],
            target[1],
        )

        dx = target[0] - origin[0]
        dy = target[1] - origin[1]

        axis_length = math.hypot(
            dx,
            dy,
        )

        if axis_length <= 0:
            raise ProfileError("Origin and target are identical.")

        stations = self._generate_stations(
            axis_length,
        )

        ux = dx / axis_length
        uy = dy / axis_length

        points: list[ProfilePoint] = []

        for station in stations:
            x = origin[0] + ux * station
            y = origin[1] + uy * station

            z = interpolate_surface(
                surface,
                x,
                y,
            )

            points.append(
                ProfilePoint(
                    station=station,
                    x=x,
                    y=y,
                    z=z,
                    offset=0.0,
                )
            )

        return ProfileResult(
            points=points,
            profile_type=ProfileType.LONGITUDINAL,
            axis_length=axis_length,
        )

    def _generate_stations(
        self,
        axis_length: float,
    ) -> list[float]:
        """
        Generate stations along the profile axis.
        """

        if self._interval >= axis_length:
            return [
                0.0,
                axis_length,
            ]

        count = int(axis_length // self._interval)

        stations = [index * self._interval for index in range(count + 1)]

        if abs(stations[-1] - axis_length) > 1e-9:
            stations.append(axis_length)

        return stations

    def __call__(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        surface: TerrainSurface,
    ) -> ProfileResult:
        """
        Generate longitudinal profile.
        """

        return self.generate(
            origin,
            target,
            surface,
        )


__all__ = [
    "LongitudinalProfile",
]
