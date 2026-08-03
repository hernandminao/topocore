"""
topocore.analysis.profile.cross_section
========================================

Cross-section profile generation.

Generates cross-sectional profiles along a polyline axis. At each
vertex of the polyline, a transversal is extracted perpendicular
to the local axis direction.

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
from topocore.analysis.types import ProfileResult

from .transversal import TransversalProfile


class CrossSectionProfile:
    """
    Generates cross-sectional profiles along a polyline axis.

    Parameters
    ----------
    interval
        Sampling interval along each transversal.
    width
        Half-width of each transversal.
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
        """
        Sampling interval.
        """
        return self._interval

    @property
    def width(self) -> float:
        """
        Half width of transversal.
        """
        return self._width

    def generate(
        self,
        axis: list[tuple[float, float]],
        surface: TerrainSurface,
    ) -> list[ProfileResult]:
        """
        Generate cross sections along a polyline.

        Parameters
        ----------
        axis
            Ordered XY vertices of the alignment.
        surface
            Terrain surface implementing TerrainSurface.

        Returns
        -------
        list[ProfileResult]
            Generated transversal profiles.

        Raises
        ------
        ProfileError
            If axis geometry is invalid.
        """

        if len(axis) < 2:
            raise ProfileError("Axis must contain at least two vertices.")

        for index, point in enumerate(axis):
            validate_xy(
                point[0],
                point[1],
            )

        transversal = TransversalProfile(
            interval=self._interval,
            width=self._width,
        )

        results: list[ProfileResult] = []

        cumulative_station = 0.0

        for index, vertex in enumerate(axis):
            origin, target = self._local_direction(
                axis,
                index,
            )

            section = transversal.generate(
                axis_origin=origin,
                axis_target=target,
                station=cumulative_station,
                surface=surface,
            )

            # Create new immutable result instead of modifying it
            results.append(
                ProfileResult(
                    points=section.points,
                    profile_type=section.profile_type,
                    axis_length=self._width * 2,
                )
            )

            if index < len(axis) - 1:
                next_point = axis[index + 1]

                cumulative_station += math.hypot(
                    next_point[0] - vertex[0],
                    next_point[1] - vertex[1],
                )

        return results

    def _local_direction(
        self,
        axis: list[tuple[float, float]],
        index: int,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
    ]:
        """
        Compute local axis direction.

        Uses adjacent segments to determine the transversal
        orientation.
        """

        if index == 0:
            return (
                axis[0],
                axis[1],
            )

        if index == len(axis) - 1:
            return (
                axis[-2],
                axis[-1],
            )

        previous_point = axis[index - 1]
        current_point = axis[index]
        next_point = axis[index + 1]

        incoming_x = current_point[0] - previous_point[0]

        incoming_y = current_point[1] - previous_point[1]

        outgoing_x = next_point[0] - current_point[0]

        outgoing_y = next_point[1] - current_point[1]

        origin = (
            current_point[0] - incoming_x,
            current_point[1] - incoming_y,
        )

        target = (
            current_point[0] + outgoing_x,
            current_point[1] + outgoing_y,
        )

        return origin, target

    def __call__(
        self,
        axis: list[tuple[float, float]],
        surface: TerrainSurface,
    ) -> list[ProfileResult]:
        """
        Generate cross sections.
        """

        return self.generate(
            axis,
            surface,
        )


__all__ = [
    "CrossSectionProfile",
]
