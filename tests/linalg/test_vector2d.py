"""
Tests for topocore.linalg.vector2d.
"""

from __future__ import annotations

import math

import pytest

from topocore.core.exceptions import MathError
from topocore.linalg.vector2d import Vector2D

# ==========================================================
# Construction
# ==========================================================


def test_create_vector() -> None:
    vector = Vector2D(
        3.0,
        4.0,
    )

    assert vector.x == 3.0
    assert vector.y == 4.0


# ==========================================================
# Properties
# ==========================================================


def test_length() -> None:
    vector = Vector2D(
        3.0,
        4.0,
    )

    assert math.isclose(
        vector.length,
        5.0,
    )


def test_length_squared() -> None:
    vector = Vector2D(
        3.0,
        4.0,
    )

    assert vector.length_squared == 25.0


def test_zero_vector() -> None:
    vector = Vector2D(
        0.0,
        0.0,
    )

    assert vector.is_zero


# ==========================================================
# Dot product
# ==========================================================


def test_dot_product() -> None:
    first = Vector2D(
        1.0,
        2.0,
    )

    second = Vector2D(
        3.0,
        4.0,
    )

    assert first.dot(second) == 11.0


# ==========================================================
# Cross product
# ==========================================================


def test_cross_product() -> None:
    first = Vector2D(
        1.0,
        0.0,
    )

    second = Vector2D(
        0.0,
        1.0,
    )

    assert first.cross(second) == 1.0


# ==========================================================
# Normalization
# ==========================================================


def test_normalized() -> None:
    vector = Vector2D(
        3.0,
        4.0,
    ).normalized()

    assert math.isclose(
        vector.length,
        1.0,
    )


def test_normalize_zero_vector() -> None:
    with pytest.raises(MathError):
        Vector2D(
            0.0,
            0.0,
        ).normalized()


# ==========================================================
# Angle
# ==========================================================


def test_angle_between_vectors() -> None:
    angle = Vector2D(
        1.0,
        0.0,
    ).angle_to(
        Vector2D(
            0.0,
            1.0,
        )
    )

    assert math.isclose(
        angle,
        math.pi / 2.0,
    )


# ==========================================================
# Conversion
# ==========================================================


def test_to_tuple() -> None:
    vector = Vector2D(
        5.0,
        7.0,
    )

    assert vector.to_tuple() == (
        5.0,
        7.0,
    )


def test_to_dict() -> None:
    vector = Vector2D(
        5.0,
        7.0,
    )

    assert vector.to_dict() == {
        "x": 5.0,
        "y": 7.0,
    }


# ==========================================================
# almost_equals
# ==========================================================


def test_almost_equals() -> None:
    first = Vector2D(
        1.0,
        2.0,
    )

    second = Vector2D(
        1.0 + 1e-13,
        2.0,
    )

    assert first.almost_equals(
        second,
    )


# ==========================================================
# Operators
# ==========================================================


def test_add() -> None:
    result = Vector2D(
        1.0,
        2.0,
    ) + Vector2D(
        3.0,
        4.0,
    )

    assert result == Vector2D(
        4.0,
        6.0,
    )


def test_subtract() -> None:
    result = Vector2D(
        5.0,
        6.0,
    ) - Vector2D(
        1.0,
        2.0,
    )

    assert result == Vector2D(
        4.0,
        4.0,
    )


def test_scalar_multiplication() -> None:
    result = (
        Vector2D(
            2.0,
            3.0,
        )
        * 2
    )

    assert result == Vector2D(
        4.0,
        6.0,
    )


def test_right_scalar_multiplication() -> None:
    result = 2 * Vector2D(
        2.0,
        3.0,
    )

    assert result == Vector2D(
        4.0,
        6.0,
    )


def test_scalar_division() -> None:
    result = (
        Vector2D(
            8.0,
            4.0,
        )
        / 2
    )

    assert result == Vector2D(
        4.0,
        2.0,
    )


def test_scalar_division_by_zero() -> None:
    with pytest.raises(
        MathError,
    ):
        Vector2D(
            1.0,
            1.0,
        ) / 0


def test_negative_vector() -> None:
    vector = -Vector2D(
        3.0,
        -4.0,
    )

    assert vector == Vector2D(
        -3.0,
        4.0,
    )


def test_absolute_value() -> None:
    vector = Vector2D(
        3.0,
        4.0,
    )

    assert abs(vector) == 5.0


# ==========================================================
# Invalid operations
# ==========================================================


def test_invalid_addition() -> None:
    vector = Vector2D(
        1.0,
        2.0,
    )

    with pytest.raises(TypeError):
        _ = vector + 10


def test_invalid_subtraction() -> None:
    vector = Vector2D(
        1.0,
        2.0,
    )

    with pytest.raises(TypeError):
        _ = vector - "abc"


def test_invalid_multiplication() -> None:
    vector = Vector2D(
        1.0,
        2.0,
    )

    with pytest.raises(TypeError):
        _ = vector * "abc"


def test_invalid_division() -> None:
    vector = Vector2D(
        1.0,
        2.0,
    )

    with pytest.raises(TypeError):
        _ = vector / object()
