"""
topocore.analysis.visibility.los
=================================

Line-of-Sight (LOS) analysis.

Determines visibility between two terrain points using a
triangulated surface and sampled terrain profile.

The algorithm evaluates terrain elevation along the observer-target
segment and detects terrain intersections that block visibility.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.protocols import TriangulatedSurface
from topocore.analysis.types import LOSResult

# ============================================================================
# Constants
# ============================================================================

_EARTH_RADIUS_KM = 6371.0
_DISTANCE_EPSILON = 1e-12
_KM_TO_METERS = 1000.0

_DEFAULT_OBSERVER_HEIGHT = 1.7
_DEFAULT_TARGET_HEIGHT = 0.0
_DEFAULT_SAMPLE_COUNT = 100

_BARYCENTRIC_EPSILON = 1e-15


class LineOfSight:
    """
    Line-of-Sight visibility analysis over a TIN.

    Parameters
    ----------
    observer_height
        Height of observer above terrain in meters.
    target_height
        Height of target above terrain in meters.
    earth_curvature
        Enable earth curvature correction.
    num_samples
        Number of terrain samples along LOS.
    """

    __slots__ = (
        "_earth_curvature",
        "_num_samples",
        "_observer_height",
        "_target_height",
    )

    def __init__(
        self,
        observer_height: float = _DEFAULT_OBSERVER_HEIGHT,
        target_height: float = _DEFAULT_TARGET_HEIGHT,
        earth_curvature: bool = True,
        num_samples: int = _DEFAULT_SAMPLE_COUNT,
    ) -> None:

        if observer_height < 0:
            raise VisibilityError("Observer height cannot be negative.")

        if target_height < 0:
            raise VisibilityError("Target height cannot be negative.")

        if num_samples < 2:
            raise VisibilityError("num_samples must be at least 2.")

        self._observer_height = float(observer_height)
        self._target_height = float(target_height)
        self._earth_curvature = bool(earth_curvature)
        self._num_samples = int(num_samples)

    @property
    def observer_height(self) -> float:
        """Observer height."""
        return self._observer_height

    @property
    def target_height(self) -> float:
        """Target height."""
        return self._target_height

    def compute(
        self,
        observer: tuple[float, float],
        target: tuple[float, float],
        tin: TriangulatedSurface,
    ) -> LOSResult:
        """
        Compute line-of-sight visibility.

        Parameters
        ----------
        observer
            Observer XY coordinate.
        target
            Target XY coordinate.
        tin
            Triangulated terrain surface.

        Returns
        -------
        LOSResult
            Visibility result.
        """

        self._validate_points(
            observer,
            target,
            tin,
        )

        observer_z = (
            tin.interpolate(
                observer[0],
                observer[1],
            )
            + self._observer_height
        )

        target_ground_z = tin.interpolate(
            target[0],
            target[1],
        )

        target_z = target_ground_z + self._target_height

        dx = target[0] - observer[0]
        dy = target[1] - observer[1]

        distance = math.hypot(dx, dy)

        if distance <= _DISTANCE_EPSILON:
            return LOSResult(
                visible=True,
                distance=0.0,
                obstacles=(),
                elevation_at_target=target_ground_z,
                clearance=0.0,
            )

        obstacles: set[int] = set()

        minimum_clearance = float("inf")

        for index in range(1, self._num_samples):
            factor = index / self._num_samples

            x = observer[0] + dx * factor
            y = observer[1] + dy * factor

            los_z = observer_z + (target_z - observer_z) * factor

            if self._earth_curvature:
                los_z -= self._curvature_correction(
                    distance * factor,
                    distance * (1.0 - factor),
                )

            triangle = tin.find_triangle(x, y)

            if triangle < 0:
                continue

            terrain_z = tin.interpolate(
                x,
                y,
            )

            clearance = los_z - terrain_z

            minimum_clearance = min(
                minimum_clearance,
                clearance,
            )

            if clearance < 0.0:
                obstacles.add(triangle)

        return LOSResult(
            visible=not obstacles,
            distance=distance,
            obstacles=tuple(sorted(obstacles)),
            elevation_at_target=target_ground_z,
            clearance=(minimum_clearance if minimum_clearance != float("inf") else 0.0),
        )

    @staticmethod
    def _curvature_correction(
        distance_from_observer_meters: float,
        distance_from_target_meters: float,
    ) -> float:
        """
        Compute earth curvature correction (the "earth bulge") at a
        point along the observer-target path.

        Parameters
        ----------
        distance_from_observer_meters
            Distance from the observer to the sample point (d1).
        distance_from_target_meters
            Distance from the sample point to the target (d2).

        Returns
        -------
        float
            Curvature correction in meters.

        Notes
        -----
        A real, severe bug was found and fixed here in PR19: this
        method used to take only ``distance_from_observer_meters``
        and compute ``d1**2 / (2R)`` -- a formula that grows without
        bound as the sample point approaches the TARGET, instead of
        correctly returning to zero at both path endpoints (the
        observer's and target's own given elevations are already
        correct/complete and need no further correction at their own
        position). The standard, correct "earth bulge" formula used
        in radio/microwave path engineering and surveying visibility
        studies is the PRODUCT form ``d1 * d2 / (2R)`` -- zero when
        either d1=0 (at the observer) or d2=0 (at the target), and
        maximal at the path's midpoint.

        Confirmed the bug directly against the well-known horizon-
        distance formula (``d = sqrt(2*R*h)``, ~4.65 km for a 1.7 m
        eye height): the old formula reported a target at GROUND
        level as already invisible at just 1 km, because its
        (incorrect) correction grew largest right next to the
        target rather than vanishing there.
        """

        d1_km = distance_from_observer_meters / _KM_TO_METERS
        d2_km = distance_from_target_meters / _KM_TO_METERS

        return d1_km * d2_km / (2.0 * _EARTH_RADIUS_KM) * _KM_TO_METERS

    @staticmethod
    def _validate_points(
        observer: tuple[float, float],
        target: tuple[float, float],
        tin: TriangulatedSurface,
    ) -> None:
        """Validate observer and target positions."""

        if not tin.contains(
            observer[0],
            observer[1],
        ):
            raise VisibilityError("Observer is outside the TIN.")

        if not tin.contains(
            target[0],
            target[1],
        ):
            raise VisibilityError("Target is outside the TIN.")

    @staticmethod
    def _interpolate_tin(
        tin: TriangulatedSurface,
        x: float,
        y: float,
        triangle_index: int,
    ) -> float:
        """
        Interpolate elevation using barycentric coordinates.

        Kept as an internal fallback utility.
        """
        p1, p2, p3 = tin.triangle_vertices(triangle_index)

        denominator = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y)

        if abs(denominator) < _BARYCENTRIC_EPSILON:
            return (p1.z + p2.z + p3.z) / 3.0

        w1 = ((p2.y - p3.y) * (x - p3.x) + (p3.x - p2.x) * (y - p3.y)) / denominator

        w2 = ((p3.y - p1.y) * (x - p3.x) + (p1.x - p3.x) * (y - p3.y)) / denominator

        w3 = 1.0 - w1 - w2

        return w1 * p1.z + w2 * p2.z + w3 * p3.z

    def __call__(
        self,
        observer: tuple[float, float],
        target: tuple[float, float],
        tin: TriangulatedSurface,
    ) -> LOSResult:
        """Execute LOS computation."""
        return self.compute(
            observer,
            target,
            tin,
        )


__all__ = ["LineOfSight"]
