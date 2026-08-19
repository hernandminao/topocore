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

        Always includes ``offset=0.0`` (the axis/centerline itself)
        exactly once, along with both exact endpoints (``-width``,
        ``+width``), regardless of whether ``width`` is an exact
        multiple of ``interval``.

        Found and fixed in PR19: the previous implementation stepped
        from ``-width`` in increments of ``interval``
        (``-width + i*interval``), which only hits exactly 0 when
        ``width`` happens to be an exact multiple of ``interval`` --
        confirmed directly with ``width=10, interval=3``: the
        generated offsets were ``[-10,-7,-4,-1,2,5,8,10]``, with the
        two points nearest zero being -1 and +2, straddling but
        never touching the axis. This broke the guarantee
        ``LongitudinalProfile`` already provides for its own
        equivalent reference point (``station=0`` is always present
        by construction), and propagated into
        ``CrossSectionProfile`` (which delegates to this method for
        every vertex of an alignment). ``MultiProfile`` was
        confirmed unaffected -- it takes explicit, caller-supplied
        offsets (defaulting to ``(0.0,)``) rather than
        auto-generating a grid.

        Fixed by building the positive-side offsets outward from 0
        (0, interval, 2*interval, ..., up to width, with the exact
        ``width`` endpoint appended if not already hit), then
        mirroring the strictly-positive values to the negative side
        and combining -- 0 is included by construction, exactly
        once, on both the exact-multiple and non-exact-multiple
        paths. This also naturally subsumes the old
        ``interval >= width`` special case (which previously
        returned only ``[-width, width]``, missing 0 -- now handled
        correctly by the same general logic, no separate branch
        needed).
        """
        if self._width <= 0:
            return [0.0]

        count = int(self._width // self._interval)
        positive = [index * self._interval for index in range(count + 1)]

        if abs(positive[-1] - self._width) > 1e-9:
            positive.append(self._width)

        negative = [-value for value in positive if value > 1e-9]

        return sorted(negative + positive)

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
