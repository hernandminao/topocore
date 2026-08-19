"""
Tests for Alignment construction and Alignment.station_to_point(),
with analytically known geometry (straight lines, quarter/half/major
circular arcs).
"""

from __future__ import annotations

import math

import pytest

from topocore.alignment.elements import ArcElement, LineElement
from topocore.alignment.exceptions import (
    AlignmentError,
    AlignmentGeometryError,
    AlignmentStationError,
)
from topocore.alignment.models import Alignment
from topocore.geometry.point2d import Point2D

# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_alignment_rejects_blank_name() -> None:
    with pytest.raises(AlignmentError):
        Alignment(name="   ", elements=(LineElement(Point2D(0, 0), Point2D(10, 0)),))


def test_alignment_rejects_empty_elements() -> None:
    with pytest.raises(AlignmentError):
        Alignment(name="A", elements=())


def test_alignment_rejects_discontinuous_elements() -> None:
    line_a = LineElement(Point2D(0.0, 0.0), Point2D(10.0, 0.0))
    line_b = LineElement(Point2D(10.0, 1.0), Point2D(20.0, 1.0))  # gap: (10,0) != (10,1)

    with pytest.raises(AlignmentGeometryError):
        Alignment(name="A", elements=(line_a, line_b))


def test_alignment_accepts_continuous_elements() -> None:
    line_a = LineElement(Point2D(0.0, 0.0), Point2D(10.0, 0.0))
    line_b = LineElement(Point2D(10.0, 0.0), Point2D(20.0, 0.0))

    Alignment(name="A", elements=(line_a, line_b))  # must not raise


# ----------------------------------------------------------------------
# Line: start / mid / end
# ----------------------------------------------------------------------


def test_line_alignment_length_is_analytic() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,))

    assert alignment.length == pytest.approx(100.0)
    assert alignment.end_station == pytest.approx(100.0)


def test_line_alignment_station_to_point_start() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,))

    point = alignment.station_to_point(0.0)

    assert point.x == pytest.approx(0.0)
    assert point.y == pytest.approx(0.0)


def test_line_alignment_station_to_point_mid() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,))

    point = alignment.station_to_point(50.0)

    assert point.x == pytest.approx(50.0)
    assert point.y == pytest.approx(0.0)


def test_line_alignment_station_to_point_end() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,))

    point = alignment.station_to_point(100.0)

    assert point.x == pytest.approx(100.0)
    assert point.y == pytest.approx(0.0)


def test_diagonal_line_station_to_point_mid() -> None:
    """
    A 3-4-5 triangle line: length 5, midpoint at (1.5, 2.0).
    """
    line = LineElement(Point2D(0.0, 0.0), Point2D(3.0, 4.0))
    alignment = Alignment(name="A", elements=(line,))

    assert alignment.length == pytest.approx(5.0)

    point = alignment.station_to_point(2.5)

    assert point.x == pytest.approx(1.5)
    assert point.y == pytest.approx(2.0)


# ----------------------------------------------------------------------
# Arc: start / mid / end, both directions
# ----------------------------------------------------------------------


def test_ccw_quarter_arc_length_is_analytic() -> None:
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
    )
    alignment = Alignment(name="A", elements=(arc,))

    assert alignment.length == pytest.approx(50.0 * math.pi / 2.0)


def test_ccw_quarter_arc_station_to_point_start() -> None:
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
    )
    alignment = Alignment(name="A", elements=(arc,))

    point = alignment.station_to_point(0.0)

    assert point.x == pytest.approx(50.0)
    assert point.y == pytest.approx(0.0)


def test_ccw_quarter_arc_station_to_point_mid() -> None:
    """
    45 degrees into a CCW quarter circle of radius 50, centered at
    origin: (50*cos45, 50*sin45).
    """
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
    )
    alignment = Alignment(name="A", elements=(arc,))

    half_length = 50.0 * (math.pi / 2.0) / 2.0
    point = alignment.station_to_point(half_length)

    expected = 50.0 * math.cos(math.pi / 4.0)

    assert point.x == pytest.approx(expected)
    assert point.y == pytest.approx(expected)


def test_ccw_quarter_arc_station_to_point_end() -> None:
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
    )
    alignment = Alignment(name="A", elements=(arc,))

    point = alignment.station_to_point(alignment.length)

    assert point.x == pytest.approx(0.0, abs=1e-9)
    assert point.y == pytest.approx(50.0)


def test_cw_quarter_arc_station_to_point_mid() -> None:
    """
    Same quarter circle, opposite direction: start=(0,50), end=(50,0),
    clockwise=True. Midpoint must land at the SAME physical point as
    the CCW case above (45 degrees), confirming direction doesn't
    change the geometric midpoint of a symmetric quarter arc, only
    the parametrization along the way.
    """
    arc = ArcElement(
        start=Point2D(0.0, 50.0),
        end=Point2D(50.0, 0.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=True,
    )
    alignment = Alignment(name="A", elements=(arc,))

    half_length = alignment.length / 2.0
    point = alignment.station_to_point(half_length)

    expected = 50.0 * math.cos(math.pi / 4.0)

    assert point.x == pytest.approx(expected)
    assert point.y == pytest.approx(expected)


def test_cw_quarter_arc_downward_station_to_point_mid() -> None:
    """
    A CW quarter arc going *down* from (50,0) to (0,-50): distinct
    from the CCW upward case, verifying clockwise direction is
    actually respected (not just symmetric-case luck).
    """
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, -50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=True,
    )
    alignment = Alignment(name="A", elements=(arc,))

    half_length = alignment.length / 2.0
    point = alignment.station_to_point(half_length)

    # -45 degrees
    expected_x = 50.0 * math.cos(-math.pi / 4.0)
    expected_y = 50.0 * math.sin(-math.pi / 4.0)

    assert point.x == pytest.approx(expected_x)
    assert point.y == pytest.approx(expected_y)


def test_major_arc_three_quarter_sweep() -> None:
    """
    A CCW arc from (50,0) to (0,-50) the "long way around" (270
    degrees / 3*pi/2, not the 90-degree short way) -- confirms the
    sweep-angle convention correctly represents major arcs, not just
    minor ones.
    """
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, -50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,  # CCW: 0 -> -90deg the "long way" is +270deg
    )
    alignment = Alignment(name="A", elements=(arc,))

    assert alignment.length == pytest.approx(50.0 * 3.0 * math.pi / 2.0)

    # Halfway along this major arc: 135 degrees CCW from start.
    half_length = alignment.length / 2.0
    point = alignment.station_to_point(half_length)

    angle = math.pi * 3.0 / 4.0  # 135 degrees
    assert point.x == pytest.approx(50.0 * math.cos(angle))
    assert point.y == pytest.approx(50.0 * math.sin(angle))


# ----------------------------------------------------------------------
# Mixed line + arc chain
# ----------------------------------------------------------------------


def test_line_then_arc_chain_length_and_junction() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(50.0, 0.0))
    arc = ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
    )
    alignment = Alignment(name="A", elements=(line, arc))

    expected_length = 50.0 + 50.0 * math.pi / 2.0
    assert alignment.length == pytest.approx(expected_length)

    # Station exactly at the junction (end of line == start of arc).
    junction = alignment.station_to_point(50.0)
    assert junction.x == pytest.approx(50.0)
    assert junction.y == pytest.approx(0.0)

    # Station just past the junction, into the arc.
    into_arc = alignment.station_to_point(50.0 + 1.0)
    angle = 1.0 / 50.0  # radians traveled into the arc
    assert into_arc.x == pytest.approx(50.0 * math.cos(angle))
    assert into_arc.y == pytest.approx(50.0 * math.sin(angle))


# ----------------------------------------------------------------------
# start_station offset
# ----------------------------------------------------------------------


def test_non_zero_start_station_offsets_queries() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,), start_station=1000.0)

    assert alignment.end_station == pytest.approx(1100.0)

    start_point = alignment.station_to_point(1000.0)
    assert start_point.x == pytest.approx(0.0)

    end_point = alignment.station_to_point(1100.0)
    assert end_point.x == pytest.approx(100.0)


# ----------------------------------------------------------------------
# Out-of-range stations
# ----------------------------------------------------------------------


def test_station_before_start_raises() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,))

    with pytest.raises(AlignmentStationError):
        alignment.station_to_point(-1.0)


def test_station_beyond_end_raises() -> None:
    line = LineElement(Point2D(0.0, 0.0), Point2D(100.0, 0.0))
    alignment = Alignment(name="A", elements=(line,))

    with pytest.raises(AlignmentStationError):
        alignment.station_to_point(100.1)
