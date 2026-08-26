"""
Regression/coverage suite for topocore.geometry.point2d.Point2D and
.point3d.Point3D -- PR20 coverage phase.

Verified with known Pythagorean cases (3-4-5 in 2D, the 3-4-12-13
Pythagorean quadruple in 3D), degenerate coincident points, operator
support (Point+Vector, Point-Point, Point-Vector, and rejection of
unsupported operand combinations via NotImplemented -> TypeError),
floating-point tolerance in almost_equals, immutability, and
coordinate validation (NaN, infinity, excessive magnitude all
rejected at construction). No bugs found -- both classes were
already correct; only test coverage was added.
"""

from __future__ import annotations

import pytest

from topocore.core.exceptions import TopoCoreError
from topocore.geometry.point2d import Point2D
from topocore.geometry.point3d import Point3D
from topocore.linalg.vector2d import Vector2D
from topocore.linalg.vector3d import Vector3D
from topocore.math.validation import MAX_COORDINATE_MAGNITUDE

# ----------------------------------------------------------------------
# Point2D -- known distances, midpoint, vectors.
# ----------------------------------------------------------------------


def test_point2d_distance_known_3_4_5_triangle() -> None:
    p1, p2 = Point2D(0.0, 0.0), Point2D(3.0, 4.0)
    assert p1.distance_to(p2) == pytest.approx(5.0)
    assert p1.squared_distance_to(p2) == pytest.approx(25.0)


def test_point2d_distance_to_self_is_zero() -> None:
    p = Point2D(5.0, 5.0)
    assert p.distance_to(p) == 0.0


def test_point2d_midpoint() -> None:
    p1, p2 = Point2D(0.0, 0.0), Point2D(3.0, 4.0)
    mid = p1.midpoint(p2)
    assert mid.x == pytest.approx(1.5)
    assert mid.y == pytest.approx(2.0)


def test_point2d_vector_to() -> None:
    p1, p2 = Point2D(1.0, 1.0), Point2D(4.0, 5.0)
    v = p1.vector_to(p2)
    assert v.x == pytest.approx(3.0)
    assert v.y == pytest.approx(4.0)


def test_point2d_translate() -> None:
    p = Point2D(1.0, 1.0)
    result = p.translate(Vector2D(2.0, 3.0))
    assert result.x == pytest.approx(3.0)
    assert result.y == pytest.approx(4.0)


def test_point2d_to_tuple_and_to_dict() -> None:
    p = Point2D(1.0, 2.0)
    assert p.to_tuple() == (1.0, 2.0)
    assert p.to_dict() == {"x": 1.0, "y": 2.0}


# ----------------------------------------------------------------------
# Point2D -- almost_equals tolerance.
# ----------------------------------------------------------------------


def test_point2d_almost_equals_within_tolerance() -> None:
    p1 = Point2D(1.0, 1.0)
    p2 = Point2D(1.0 + 1e-12, 1.0)
    assert p1.almost_equals(p2) is True


def test_point2d_almost_equals_beyond_tolerance() -> None:
    p1 = Point2D(1.0, 1.0)
    p2 = Point2D(1.1, 1.0)
    assert p1.almost_equals(p2) is False


# ----------------------------------------------------------------------
# Point2D -- operators.
# ----------------------------------------------------------------------


def test_point2d_add_vector() -> None:
    p = Point2D(0.0, 0.0)
    result = p + Vector2D(1.0, 1.0)
    assert result == Point2D(1.0, 1.0)


def test_point2d_sub_point_gives_vector() -> None:
    p1, p2 = Point2D(3.0, 4.0), Point2D(0.0, 0.0)
    result = p1 - p2
    assert isinstance(result, Vector2D)
    assert result.x == pytest.approx(3.0)
    assert result.y == pytest.approx(4.0)


def test_point2d_sub_vector_gives_point() -> None:
    p = Point2D(3.0, 4.0)
    result = p - Vector2D(1.0, 1.0)
    assert isinstance(result, Point2D)
    assert result == Point2D(2.0, 3.0)


def test_point2d_add_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Point2D(0.0, 0.0) + Point2D(1.0, 1.0)  # type: ignore[operator]


def test_point2d_sub_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Point2D(0.0, 0.0) - "not a point"  # type: ignore[operator]


# ----------------------------------------------------------------------
# Point2D -- validation and immutability.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_point2d_rejects_non_finite_coordinates(bad_value: float) -> None:
    with pytest.raises(TopoCoreError):
        Point2D(bad_value, 0.0)


def test_point2d_rejects_excessive_magnitude() -> None:
    with pytest.raises(TopoCoreError):
        Point2D(MAX_COORDINATE_MAGNITUDE * 2, 0.0)


def test_point2d_is_immutable() -> None:
    p = Point2D(1.0, 2.0)
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError, but let dataclasses own it
        p.x = 5.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# Point3D -- known distances (Pythagorean quadruple), midpoint, vectors.
# ----------------------------------------------------------------------


def test_point3d_distance_known_pythagorean_quadruple() -> None:
    """3-4-12-13 is a Pythagorean quadruple: 3^2 + 4^2 + 12^2 = 13^2."""
    p1, p2 = Point3D(0.0, 0.0, 0.0), Point3D(3.0, 4.0, 12.0)
    assert p1.distance_to(p2) == pytest.approx(13.0)
    assert p1.squared_distance_to(p2) == pytest.approx(169.0)


def test_point3d_distance_to_self_is_zero() -> None:
    p = Point3D(5.0, 5.0, 5.0)
    assert p.distance_to(p) == 0.0


def test_point3d_midpoint() -> None:
    p1, p2 = Point3D(0.0, 0.0, 0.0), Point3D(3.0, 4.0, 12.0)
    mid = p1.midpoint(p2)
    assert mid.x == pytest.approx(1.5)
    assert mid.y == pytest.approx(2.0)
    assert mid.z == pytest.approx(6.0)


def test_point3d_vector_to() -> None:
    p1, p2 = Point3D(1.0, 1.0, 1.0), Point3D(4.0, 5.0, 9.0)
    v = p1.vector_to(p2)
    assert v.x == pytest.approx(3.0)
    assert v.y == pytest.approx(4.0)
    assert v.z == pytest.approx(8.0)


def test_point3d_translate() -> None:
    p = Point3D(1.0, 1.0, 1.0)
    result = p.translate(Vector3D(2.0, 3.0, 4.0))
    assert result.x == pytest.approx(3.0)
    assert result.y == pytest.approx(4.0)
    assert result.z == pytest.approx(5.0)


def test_point3d_to_tuple_and_to_dict() -> None:
    p = Point3D(1.0, 2.0, 3.0)
    assert p.to_tuple() == (1.0, 2.0, 3.0)
    assert p.to_dict() == {"x": 1.0, "y": 2.0, "z": 3.0}


# ----------------------------------------------------------------------
# Point3D -- almost_equals tolerance.
# ----------------------------------------------------------------------


def test_point3d_almost_equals_within_tolerance() -> None:
    p1 = Point3D(1.0, 1.0, 1.0)
    p2 = Point3D(1.0 + 1e-12, 1.0, 1.0)
    assert p1.almost_equals(p2) is True


def test_point3d_almost_equals_beyond_tolerance() -> None:
    p1 = Point3D(1.0, 1.0, 1.0)
    p2 = Point3D(1.0, 1.1, 1.0)
    assert p1.almost_equals(p2) is False


# ----------------------------------------------------------------------
# Point3D -- operators.
# ----------------------------------------------------------------------


def test_point3d_add_vector() -> None:
    p = Point3D(0.0, 0.0, 0.0)
    result = p + Vector3D(1.0, 1.0, 1.0)
    assert result == Point3D(1.0, 1.0, 1.0)


def test_point3d_sub_point_gives_vector() -> None:
    p1, p2 = Point3D(3.0, 4.0, 12.0), Point3D(0.0, 0.0, 0.0)
    result = p1 - p2
    assert isinstance(result, Vector3D)
    assert result.x == pytest.approx(3.0)
    assert result.y == pytest.approx(4.0)
    assert result.z == pytest.approx(12.0)


def test_point3d_sub_vector_gives_point() -> None:
    p = Point3D(3.0, 4.0, 12.0)
    result = p - Vector3D(1.0, 1.0, 1.0)
    assert isinstance(result, Point3D)
    assert result == Point3D(2.0, 3.0, 11.0)


def test_point3d_add_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Point3D(0.0, 0.0, 0.0) + Point3D(1.0, 1.0, 1.0)  # type: ignore[operator]


def test_point3d_sub_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Point3D(0.0, 0.0, 0.0) - "not a point"  # type: ignore[operator]


# ----------------------------------------------------------------------
# Point3D -- validation and immutability.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_point3d_rejects_non_finite_coordinates(bad_value: float) -> None:
    with pytest.raises(TopoCoreError):
        Point3D(bad_value, 0.0, 0.0)


def test_point3d_rejects_excessive_magnitude() -> None:
    with pytest.raises(TopoCoreError):
        Point3D(0.0, 0.0, MAX_COORDINATE_MAGNITUDE * 2)


def test_point3d_is_immutable() -> None:
    p = Point3D(1.0, 2.0, 3.0)
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError, but let dataclasses own it
        p.z = 5.0  # type: ignore[misc]
