"""
topocore.analysis.profile.transversal
======================================

Transversal profile generation.

Generates a transversal profile perpendicular to an axis direction,
sampling terrain elevations at regular offsets.

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


class TransversalProfile:
    """
    Generates transversal profiles perpendicular to an axis.

    Parameters
    ----------
    interval
        Sampling interval along the transversal.
    width
        Half-width of the transversal.
    """

    __slots__ = (
        "_interval",
        "_width",
    )

    def __init__(
        self,
        interval: float = 1.0,
        width: float = 10.0,
    ) -> None:

        if not math.isfinite(interval):
            raise ProfileError("Interval must be finite.")

        if not math.isfinite(width):
            raise ProfileError("Width must be finite.")

        if interval <= 0:
            raise ProfileError("Interval must be positive.")

        if width <= 0:
            raise ProfileError("Width must be positive.")

        self._interval = float(interval)
        self._width = float(width)

    @property
    def interval(self) -> float:
        """Sampling interval."""
        return self._interval

    @property
    def width(self) -> float:
        """Half width."""
        return self._width

    def generate(
        self,
        axis_origin: tuple[float, float],
        axis_target: tuple[float, float],
        station: float,
        surface: TerrainSurface,
    ) -> ProfileResult:
        """
        Generate transversal profile.
        """

        validate_xy(
            axis_origin[0],
            axis_origin[1],
        )

        validate_xy(
            axis_target[0],
            axis_target[1],
        )

        if not math.isfinite(station):
            raise ProfileError("Station must be finite.")

        dx = axis_target[0] - axis_origin[0]
        dy = axis_target[1] - axis_origin[1]

        axis_length = math.hypot(
            dx,
            dy,
        )

        if axis_length <= 0:
            raise ProfileError("Axis length is zero.")

        ux = dx / axis_length
        uy = dy / axis_length

        center_x = axis_origin[0] + ux * station

        center_y = axis_origin[1] + uy * station

        # Perpendicular vector
        px = -uy
        py = ux

        offsets = self._generate_offsets()

        points: list[ProfilePoint] = []

        for offset in offsets:
            x = center_x + px * offset
            y = center_y + py * offset

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
                    offset=offset,
                )
            )

        return ProfileResult(
            points=points,
            profile_type=ProfileType.TRANSVERSAL,
            axis_length=self._width * 2,
        )

    def _generate_offsets(self) -> list[float]:
        """
        Generate transversal offsets.
        """

        if self._interval >= self._width:
            return [
                -self._width,
                self._width,
            ]

        count = int(self._width // self._interval)

        offsets = [-self._width + i * self._interval for i in range(count * 2 + 1)]

        if abs(offsets[-1] - self._width) > 1e-9:
            offsets.append(self._width)

        return offsets

    def __call__(
        self,
        axis_origin: tuple[float, float],
        axis_target: tuple[float, float],
        station: float,
        surface: TerrainSurface,
    ) -> ProfileResult:
        """
        Generate transversal profile.
        """

        return self.generate(
            axis_origin,
            axis_target,
            station,
            surface,
        )


__all__ = [
    "TransversalProfile",
]
