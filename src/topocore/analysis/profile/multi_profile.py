"""
topocore.analysis.profile.multi_profile
========================================

Multi-profile generation.

Generates multiple parallel longitudinal profiles offset from
a main axis. Useful for terrain corridors such as roads,
pipelines, channels, and engineering alignments.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.analysis._shared.surface import validate_xy
from topocore.analysis.exceptions import ProfileError
from topocore.analysis.protocols import TerrainSurface
from topocore.analysis.types import (
    ProfilePoint,
    ProfileResult,
    ProfileType,
)

from .longitudinal import LongitudinalProfile


class MultiProfile:
    """
    Generates multiple parallel longitudinal profiles.

    Parameters
    ----------
    interval
        Station interval along each profile.
    offsets
        Perpendicular offsets from the main axis.
    """

    __slots__ = (
        "_interval",
        "_offsets",
    )

    def __init__(
        self,
        interval: float = 1.0,
        offsets: list[float] | tuple[float, ...] | None = None,
    ) -> None:

        if not math.isfinite(interval):
            raise ProfileError("Interval must be finite.")

        if interval <= 0:
            raise ProfileError("Interval must be positive.")

        self._interval = float(interval)

        values = (0.0,) if offsets is None else tuple(offsets)

        if len(values) == 0:
            raise ProfileError("Offsets cannot be empty.")

        for offset in values:
            if not math.isfinite(offset):
                raise ProfileError(f"Offset must be finite; got {offset}.")

        self._offsets = tuple(float(value) for value in values)

    @property
    def interval(self) -> float:
        """
        Station interval.
        """
        return self._interval

    @property
    def offsets(self) -> tuple[float, ...]:
        """
        Profile offsets.

        Returns
        -------
        tuple
            Immutable offsets.
        """
        return self._offsets

    def generate(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        surface: TerrainSurface,
    ) -> list[ProfileResult]:
        """
        Generate parallel profiles.

        Parameters
        ----------
        origin
            Start of main axis.
        target
            End of main axis.
        surface
            Terrain surface.

        Returns
        -------
        list[ProfileResult]
            One longitudinal profile per offset.
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

        ux = dx / axis_length
        uy = dy / axis_length

        # Perpendicular unit vector
        px = -uy
        py = ux

        longitudinal = LongitudinalProfile(
            interval=self._interval,
        )

        results: list[ProfileResult] = []

        for offset in sorted(self._offsets):
            offset_origin = (
                origin[0] + px * offset,
                origin[1] + py * offset,
            )

            offset_target = (
                target[0] + px * offset,
                target[1] + py * offset,
            )

            profile = longitudinal.generate(
                origin=offset_origin,
                target=offset_target,
                surface=surface,
            )

            adjusted_points: list[ProfilePoint] = []

            for point in profile.points:
                adjusted_points.append(
                    ProfilePoint(
                        station=point.station,
                        x=point.x,
                        y=point.y,
                        z=point.z,
                        offset=offset,
                    )
                )

            results.append(
                ProfileResult(
                    points=adjusted_points,
                    profile_type=ProfileType.MULTI,
                    axis_length=profile.axis_length,
                )
            )

        return results

    def __call__(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        surface: TerrainSurface,
    ) -> list[ProfileResult]:
        """
        Generate multiple profiles.
        """

        return self.generate(
            origin,
            target,
            surface,
        )


__all__ = [
    "MultiProfile",
]
