"""
Tests for LineElement/ArcElement construction validation.
"""

from __future__ import annotations

import pytest

from topocore.alignment.elements import ArcElement, LineElement
from topocore.alignment.exceptions import AlignmentGeometryError
from topocore.geometry.point2d import Point2D


def test_line_element_accepts_valid_endpoints() -> None:
    LineElement(Point2D(0.0, 0.0), Point2D(10.0, 0.0))  # must not raise


def test_line_element_rejects_zero_length() -> None:
    with pytest.raises(AlignmentGeometryError):
        LineElement(Point2D(5.0, 5.0), Point2D(5.0, 5.0))


def test_arc_element_accepts_consistent_geometry() -> None:
    ArcElement(
        start=Point2D(50.0, 0.0),
        end=Point2D(0.0, 50.0),
        center=Point2D(0.0, 0.0),
        radius=50.0,
        clockwise=False,
    )  # must not raise


def test_arc_element_rejects_non_positive_radius() -> None:
    with pytest.raises(Exception):  # noqa: B017 - MathError from validate_positive
        ArcElement(
            start=Point2D(50.0, 0.0),
            end=Point2D(0.0, 50.0),
            center=Point2D(0.0, 0.0),
            radius=0.0,
            clockwise=False,
        )


def test_arc_element_rejects_start_not_at_radius() -> None:
    with pytest.raises(AlignmentGeometryError):
        ArcElement(
            start=Point2D(51.0, 0.0),  # 51, not 50
            end=Point2D(0.0, 50.0),
            center=Point2D(0.0, 0.0),
            radius=50.0,
            clockwise=False,
        )


def test_arc_element_rejects_end_not_at_radius() -> None:
    with pytest.raises(AlignmentGeometryError):
        ArcElement(
            start=Point2D(50.0, 0.0),
            end=Point2D(0.0, 49.0),  # 49, not 50
            center=Point2D(0.0, 0.0),
            radius=50.0,
            clockwise=False,
        )


def test_arc_element_rejects_coincident_start_and_end() -> None:
    with pytest.raises(AlignmentGeometryError):
        ArcElement(
            start=Point2D(50.0, 0.0),
            end=Point2D(50.0, 0.0),
            center=Point2D(0.0, 0.0),
            radius=50.0,
            clockwise=False,
        )
