"""
topocore.alignment.algorithms.horizontal
==========================================

Geometry evaluation for horizontal alignment elements
(``LineElement``, ``ArcElement``).

Convention for arcs: the swept angle is always measured from the
center-to-start vector, walking in the direction ``clockwise``
indicates, and is always taken as the positive value in
``(0, 2*pi]`` -- this correctly represents major arcs (sweep > pi)
as well as minor arcs, since it never assumes the "short way around"
is the ones meant.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.alignment.algorithms.spiral import evaluate_spiral
from topocore.alignment.elements import (
    ArcElement,
    HorizontalElement,
    LineElement,
    SpiralElement,
)
from topocore.alignment.exceptions import AlignmentGeometryError, AlignmentStationError
from topocore.geometry.point2d import Point2D
from topocore.math.tolerance import compare, is_close

_TWO_PI = 2.0 * math.pi


def element_length(element: HorizontalElement) -> float:
    """
    Arc length of a single horizontal element.
    """
    if isinstance(element, LineElement):
        return element.start.distance_to(element.end)

    if isinstance(element, ArcElement):
        return element.radius * _arc_sweep_angle(element)

    if isinstance(element, SpiralElement):
        return element.length

    raise AlignmentGeometryError(f"Unsupported horizontal element type: {type(element).__name__}")


def evaluate_element(element: HorizontalElement, local_distance: float) -> Point2D:
    """
    Compute the XY position at ``local_distance`` along a single
    element, measured from that element's own ``start``.

    ``local_distance`` is clamped to ``[0, element_length(element)]``
    -- callers (``station_to_point``) are responsible for routing
    the correct element and offset; this function does not itself
    validate that ``local_distance`` is in range, only clamps
    floating-point overshoot at the boundaries.
    """
    if isinstance(element, LineElement):
        return _evaluate_line(element, local_distance)

    if isinstance(element, ArcElement):
        return _evaluate_arc(element, local_distance)

    if isinstance(element, SpiralElement):
        return evaluate_spiral(element, local_distance)

    raise AlignmentGeometryError(f"Unsupported horizontal element type: {type(element).__name__}")


def station_to_point(
    elements: tuple[HorizontalElement, ...],
    start_station: float,
    station: float,
) -> Point2D:
    """
    Compute the XY position at ``station`` along a chain of
    horizontal elements starting at ``start_station``.

    Raises
    ------
    AlignmentStationError
        If ``station`` is outside ``[start_station, end_station]``
        (with tolerance).
    """
    if compare(station, start_station) < 0:
        raise AlignmentStationError(f"Station {station} is before the alignment's start station {start_station}.")

    remaining = station - start_station

    for element in elements:
        length = element_length(element)

        if compare(remaining, length) <= 0:
            local_distance = min(max(remaining, 0.0), length)
            return evaluate_element(element, local_distance)

        remaining -= length

    end_station = start_station + sum(element_length(element) for element in elements)
    raise AlignmentStationError(f"Station {station} is beyond the alignment's end station {end_station}.")


# ----------------------------------------------------------------------
# Line
# ----------------------------------------------------------------------


def _evaluate_line(element: LineElement, local_distance: float) -> Point2D:
    length = element.start.distance_to(element.end)
    t = local_distance / length

    return Point2D(
        element.start.x + t * (element.end.x - element.start.x),
        element.start.y + t * (element.end.y - element.start.y),
    )


# ----------------------------------------------------------------------
# Arc
# ----------------------------------------------------------------------


def _center_to_point_angle(element: ArcElement, point: Point2D) -> float:
    vector = element.center.vector_to(point)
    return math.atan2(vector.y, vector.x)


def _arc_sweep_angle(element: ArcElement) -> float:
    """
    Total angle (radians, always in ``(0, 2*pi]``) swept from
    ``start`` to ``end``, walking in the ``clockwise`` direction.
    """
    start_angle = _center_to_point_angle(element, element.start)
    end_angle = _center_to_point_angle(element, element.end)

    if element.clockwise:
        sweep = (start_angle - end_angle) % _TWO_PI
    else:
        sweep = (end_angle - start_angle) % _TWO_PI

    if is_close(sweep, 0.0):
        # start != end was already enforced in ArcElement.__post_init__,
        # so a ~zero sweep here means start and end sit on the same ray
        # from center -- geometrically degenerate, not a full circle
        # (which this module does not support -- see ArcElement docstring).
        raise AlignmentGeometryError("ArcElement has a degenerate (near-zero) sweep angle.")

    return sweep


def _evaluate_arc(element: ArcElement, local_distance: float) -> Point2D:
    start_angle = _center_to_point_angle(element, element.start)
    angle_traveled = local_distance / element.radius

    current_angle = start_angle - angle_traveled if element.clockwise else start_angle + angle_traveled

    return Point2D(
        element.center.x + element.radius * math.cos(current_angle),
        element.center.y + element.radius * math.sin(current_angle),
    )


__all__ = [
    "element_length",
    "evaluate_element",
    "station_to_point",
]
