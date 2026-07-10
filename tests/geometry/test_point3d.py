"""
Tests for topocore.geometry.point3d.
"""

from __future__ import annotations

import math

import pytest

from topocore.geometry.point3d import Point3D
from topocore.linalg.vector3d import Vector3D

# ==========================================================
# Construction
# ==========================================================


def test_create_point() -> None:
    point = Point3D(
        1.0,
        2.0,
        3.0,
    )

    assert point.x == 1.0
    assert point.y == 2.0
    assert point.z == 3.0


# ==========================================================
# Distance
# ==========================================================


def test_distance() -> None:
    first = Point3D(
        0.0,
        0.0,
        0.0,
    )

    second = Point3D(
        2.0,
        3.0,
        6.0,
    )

    assert math.isclose(
        first.distance_to(second),
        7.0,
    )


def test_squared_distance() -> None:
    first = Point3D(
        0.0,
        0.0,
        0.0,
    )

    second = Point3D(
        2.0,
        3.0,
        6.0,
    )

    assert first.squared_distance_to(second) == 49.0


# ==========================================================
# Midpoint
# ==========================================================


def test_midpoint() -> None:
    first = Point3D(
        0.0,
        0.0,
        0.0,
    )

    second = Point3D(
        10.0,
        20.0,
        30.0,
    )

    midpoint = first.midpoint(second)

    assert midpoint == Point3D(
        5.0,
        10.0,
        15.0,
    )


# ==========================================================
# Vector
# ==========================================================


def test_vector_to() -> None:
    first = Point3D(
        1.0,
        2.0,
        3.0,
    )

    second = Point3D(
        4.0,
        6.0,
        8.0,
    )

    vector = first.vector_to(second)

    assert vector == Vector3D(
        3.0,
        4.0,
        5.0,
    )


# ==========================================================
# Translation
# ==========================================================


def test_translate() -> None:
    point = Point3D(
        10.0,
        20.0,
        30.0,
    )

    translated = point.translate(
        Vector3D(
            2.0,
            3.0,
            4.0,
        )
    )

    assert translated == Point3D(
        12.0,
        23.0,
        34.0,
    )


# ==========================================================
# Conversion
# ==========================================================


def test_to_tuple() -> None:
    point = Point3D(
        7.0,
        8.0,
        9.0,
    )

    assert point.to_tuple() == (
        7.0,
        8.0,
        9.0,
    )


def test_to_dict() -> None:
    point = Point3D(
        7.0,
        8.0,
        9.0,
    )

    assert point.to_dict() == {
        "x": 7.0,
        "y": 8.0,
        "z": 9.0,
    }


# ==========================================================
# almost_equals
# ==========================================================


def test_almost_equals() -> None:
    first = Point3D(
        1.0,
        2.0,
        3.0,
    )

    second = Point3D(
        1.0 + 1e-13,
        2.0,
        3.0,
    )

    assert first.almost_equals(second)


# ==========================================================
# Operators
# ==========================================================


def test_add_vector() -> None:
    point = Point3D(
        10.0,
        20.0,
        30.0,
    )

    result = point + Vector3D(
        1.0,
        2.0,
        3.0,
    )

    assert result == Point3D(
        11.0,
        22.0,
        33.0,
    )


def test_subtract_points() -> None:
    first = Point3D(
        10.0,
        20.0,
        30.0,
    )

    second = Point3D(
        1.0,
        2.0,
        3.0,
    )

    vector = first - second

    assert vector == Vector3D(
        9.0,
        18.0,
        27.0,
    )


def test_subtract_vector() -> None:
    point = Point3D(
        10.0,
        20.0,
        30.0,
    )

    result = point - Vector3D(
        1.0,
        2.0,
        3.0,
    )

    assert result == Point3D(
        9.0,
        18.0,
        27.0,
    )


# ==========================================================
# Invalid operations
# ==========================================================


def test_invalid_addition() -> None:
    point = Point3D(
        1.0,
        2.0,
        3.0,
    )

    with pytest.raises(TypeError):
        _ = point + object()


def test_invalid_subtraction() -> None:
    point = Point3D(
        1.0,
        2.0,
        3.0,
    )

    with pytest.raises(TypeError):
        _ = point - "abc"
