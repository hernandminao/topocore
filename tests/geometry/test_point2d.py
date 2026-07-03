"""
Tests for topocore.geometry.point2d.
"""

from __future__ import annotations

import math

import pytest

from topocore.geometry.point2d import Point2D
from topocore.linalg.vector2d import Vector2D


# ==========================================================
# Construction
# ==========================================================


def test_create_point() -> None:
    point = Point2D(
        10.0,
        20.0,
    )

    assert point.x == 10.0
    assert point.y == 20.0


# ==========================================================
# Distance
# ==========================================================


def test_distance() -> None:
    first = Point2D(
        0.0,
        0.0,
    )

    second = Point2D(
        3.0,
        4.0,
    )

    assert math.isclose(
        first.distance_to(second),
        5.0,
    )


def test_squared_distance() -> None:
    first = Point2D(
        0.0,
        0.0,
    )

    second = Point2D(
        3.0,
        4.0,
    )

    assert (
        first.squared_distance_to(second)
        == 25.0
    )


# ==========================================================
# Midpoint
# ==========================================================


def test_midpoint() -> None:
    first = Point2D(
        0.0,
        0.0,
    )

    second = Point2D(
        10.0,
        20.0,
    )

    midpoint = first.midpoint(
        second,
    )

    assert midpoint == Point2D(
        5.0,
        10.0,
    )


# ==========================================================
# Vector
# ==========================================================


def test_vector_to() -> None:
    first = Point2D(
        1.0,
        2.0,
    )

    second = Point2D(
        4.0,
        6.0,
    )

    vector = first.vector_to(
        second,
    )

    assert vector == Vector2D(
        3.0,
        4.0,
    )


# ==========================================================
# Translation
# ==========================================================


def test_translate() -> None:
    point = Point2D(
        5.0,
        6.0,
    )

    translated = point.translate(
        Vector2D(
            2.0,
            3.0,
        )
    )

    assert translated == Point2D(
        7.0,
        9.0,
    )


# ==========================================================
# Conversion
# ==========================================================


def test_to_tuple() -> None:
    point = Point2D(
        7.0,
        8.0,
    )

    assert point.to_tuple() == (
        7.0,
        8.0,
    )


def test_to_dict() -> None:
    point = Point2D(
        7.0,
        8.0,
    )

    assert point.to_dict() == {
        "x": 7.0,
        "y": 8.0,
    }


# ==========================================================
# almost_equals
# ==========================================================


def test_almost_equals() -> None:
    first = Point2D(
        1.0,
        2.0,
    )

    second = Point2D(
        1.0 + 1e-13,
        2.0,
    )

    assert first.almost_equals(
        second,
    )


# ==========================================================
# Operators
# ==========================================================


def test_add_vector() -> None:
    point = Point2D(
        5.0,
        6.0,
    )

    result = point + Vector2D(
        2.0,
        3.0,
    )

    assert result == Point2D(
        7.0,
        9.0,
    )


def test_subtract_points() -> None:
    first = Point2D(
        5.0,
        8.0,
    )

    second = Point2D(
        2.0,
        4.0,
    )

    vector = first - second

    assert vector == Vector2D(
        3.0,
        4.0,
    )


def test_subtract_vector() -> None:
    point = Point2D(
        10.0,
        20.0,
    )

    result = point - Vector2D(
        2.0,
        5.0,
    )

    assert result == Point2D(
        8.0,
        15.0,
    )


# ==========================================================
# Invalid operations
# ==========================================================


def test_invalid_addition() -> None:
    point = Point2D(
        1.0,
        2.0,
    )

    with pytest.raises(
        TypeError,
    ):
        _ = point + 5


def test_invalid_subtraction() -> None:
    point = Point2D(
        1.0,
        2.0,
    )

    with pytest.raises(
        TypeError,
    ):
        _ = point - object()