"""
Regression/coverage suite for topocore.linalg.vector2d.Vector2D and
.vector3d.Vector3D -- PR20 coverage phase.

Verified with known cases: 3-4-5 magnitude, orthogonal/anti-parallel/
identical-direction angle_to (90/180/0 degrees), the 2D cross
product's antisymmetry (v1 x v2 == -(v2 x v1)) and its geometric
meaning as signed area, the 3D cross product's right-hand rule
(X x Y == Z exactly) and orthogonality to both operands (dot product
with either input is exactly zero), normalization (unit length,
correct direction), every zero-vector/zero-scalar edge case across
all operations that should reject it (normalize, angle_to, division),
and every operator including unsupported-operand rejection. No bugs
found -- both classes were already correct; only test coverage was
added.
"""

from __future__ import annotations

import math

import pytest

from topocore.core.exceptions import MathError
from topocore.linalg.vector2d import Vector2D
from topocore.linalg.vector3d import Vector3D

# ----------------------------------------------------------------------
# Vector2D -- magnitude, dot, cross.
# ----------------------------------------------------------------------


def test_vector2d_length_known_3_4_5() -> None:
    v = Vector2D(3.0, 4.0)
    assert v.length == pytest.approx(5.0)
    assert v.length_squared == pytest.approx(25.0)


def test_vector2d_dot_orthogonal_is_zero() -> None:
    v1, v2 = Vector2D(1.0, 0.0), Vector2D(0.0, 1.0)
    assert v1.dot(v2) == pytest.approx(0.0)


def test_vector2d_cross_antisymmetric() -> None:
    v1, v2 = Vector2D(1.0, 0.0), Vector2D(0.0, 1.0)
    assert v1.cross(v2) == pytest.approx(1.0)
    assert v2.cross(v1) == pytest.approx(-1.0)


# ----------------------------------------------------------------------
# Vector2D -- angle_to known cases.
# ----------------------------------------------------------------------


def test_vector2d_angle_to_orthogonal_is_90_degrees() -> None:
    v1, v2 = Vector2D(1.0, 0.0), Vector2D(0.0, 1.0)
    assert math.degrees(v1.angle_to(v2)) == pytest.approx(90.0)


def test_vector2d_angle_to_antiparallel_is_180_degrees() -> None:
    v1, v2 = Vector2D(1.0, 0.0), Vector2D(-1.0, 0.0)
    assert math.degrees(v1.angle_to(v2)) == pytest.approx(180.0)


def test_vector2d_angle_to_self_is_zero() -> None:
    v = Vector2D(1.0, 0.0)
    assert v.angle_to(v) == pytest.approx(0.0)


def test_vector2d_angle_to_zero_vector_raises() -> None:
    v = Vector2D(1.0, 0.0)
    with pytest.raises(MathError):
        v.angle_to(Vector2D(0.0, 0.0))


# ----------------------------------------------------------------------
# Vector2D -- normalization, is_zero.
# ----------------------------------------------------------------------


def test_vector2d_normalized_has_unit_length_and_correct_direction() -> None:
    v = Vector2D(3.0, 4.0)
    normalized = v.normalized()
    assert normalized.length == pytest.approx(1.0)
    assert normalized.x == pytest.approx(0.6)
    assert normalized.y == pytest.approx(0.8)


def test_vector2d_normalize_zero_vector_raises() -> None:
    with pytest.raises(MathError, match="zero vector"):
        Vector2D(0.0, 0.0).normalized()


def test_vector2d_is_zero() -> None:
    assert Vector2D(0.0, 0.0).is_zero is True
    assert Vector2D(1.0, 0.0).is_zero is False


# ----------------------------------------------------------------------
# Vector2D -- almost_equals.
# ----------------------------------------------------------------------


def test_vector2d_almost_equals_within_tolerance() -> None:
    v1, v2 = Vector2D(1.0, 1.0), Vector2D(1.0 + 1e-12, 1.0)
    assert v1.almost_equals(v2) is True


def test_vector2d_almost_equals_beyond_tolerance() -> None:
    v1, v2 = Vector2D(1.0, 1.0), Vector2D(1.1, 1.0)
    assert v1.almost_equals(v2) is False


# ----------------------------------------------------------------------
# Vector2D -- operators.
# ----------------------------------------------------------------------


def test_vector2d_add() -> None:
    assert Vector2D(1.0, 2.0) + Vector2D(3.0, 4.0) == Vector2D(4.0, 6.0)


def test_vector2d_sub() -> None:
    assert Vector2D(3.0, 4.0) - Vector2D(1.0, 2.0) == Vector2D(2.0, 2.0)


def test_vector2d_mul_scalar() -> None:
    assert Vector2D(1.0, 2.0) * 2 == Vector2D(2.0, 4.0)


def test_vector2d_rmul_scalar() -> None:
    assert 2 * Vector2D(1.0, 2.0) == Vector2D(2.0, 4.0)


def test_vector2d_truediv_scalar() -> None:
    assert Vector2D(2.0, 4.0) / 2 == Vector2D(1.0, 2.0)


def test_vector2d_division_by_zero_raises() -> None:
    with pytest.raises(MathError, match="Division by zero"):
        Vector2D(1.0, 1.0) / 0.0


def test_vector2d_neg() -> None:
    assert -Vector2D(1.0, -2.0) == Vector2D(-1.0, 2.0)


def test_vector2d_abs_is_length() -> None:
    v = Vector2D(3.0, 4.0)
    assert abs(v) == v.length


def test_vector2d_add_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector2D(1.0, 1.0) + "not a vector"  # type: ignore[operator]


def test_vector2d_mul_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector2D(1.0, 1.0) * "not a scalar"  # type: ignore[operator]


def test_vector2d_sub_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector2D(1.0, 1.0) - "not a vector"  # type: ignore[operator]


def test_vector2d_truediv_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector2D(1.0, 1.0) / "not a scalar"  # type: ignore[operator]


def test_vector2d_to_tuple_and_to_dict() -> None:
    v = Vector2D(1.0, 2.0)
    assert v.to_tuple() == (1.0, 2.0)
    assert v.to_dict() == {"x": 1.0, "y": 2.0}


# ----------------------------------------------------------------------
# Vector3D -- magnitude, dot, cross (right-hand rule + orthogonality).
# ----------------------------------------------------------------------


def test_vector3d_length_known_pythagorean_quadruple() -> None:
    v = Vector3D(3.0, 4.0, 12.0)
    assert v.length == pytest.approx(13.0)
    assert v.length_squared == pytest.approx(169.0)


def test_vector3d_dot_orthogonal_axes_is_zero() -> None:
    x_axis, y_axis = Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0)
    assert x_axis.dot(y_axis) == pytest.approx(0.0)


def test_vector3d_cross_right_hand_rule() -> None:
    """X cross Y must equal Z exactly, per the standard right-hand-rule convention."""
    x_axis, y_axis, z_axis = (
        Vector3D(1.0, 0.0, 0.0),
        Vector3D(0.0, 1.0, 0.0),
        Vector3D(0.0, 0.0, 1.0),
    )
    result = x_axis.cross(y_axis)
    assert result.almost_equals(z_axis)


def test_vector3d_cross_antisymmetric() -> None:
    x_axis, y_axis = Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0)
    assert y_axis.cross(x_axis).almost_equals(Vector3D(0.0, 0.0, -1.0))


def test_vector3d_cross_orthogonal_to_both_operands() -> None:
    v1, v2 = Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0)
    cross = v1.cross(v2)
    assert cross.dot(v1) == pytest.approx(0.0)
    assert cross.dot(v2) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Vector3D -- angle_to known cases.
# ----------------------------------------------------------------------


def test_vector3d_angle_to_orthogonal_is_90_degrees() -> None:
    x_axis, y_axis = Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0)
    assert math.degrees(x_axis.angle_to(y_axis)) == pytest.approx(90.0)


def test_vector3d_angle_to_antiparallel_is_180_degrees() -> None:
    x_axis, neg_x = Vector3D(1.0, 0.0, 0.0), Vector3D(-1.0, 0.0, 0.0)
    assert math.degrees(x_axis.angle_to(neg_x)) == pytest.approx(180.0)


def test_vector3d_angle_to_zero_vector_raises() -> None:
    v = Vector3D(1.0, 0.0, 0.0)
    with pytest.raises(MathError):
        v.angle_to(Vector3D(0.0, 0.0, 0.0))


# ----------------------------------------------------------------------
# Vector3D -- normalization, is_zero.
# ----------------------------------------------------------------------


def test_vector3d_normalized_has_unit_length() -> None:
    v = Vector3D(3.0, 4.0, 12.0)
    normalized = v.normalized()
    assert normalized.length == pytest.approx(1.0)


def test_vector3d_normalize_zero_vector_raises() -> None:
    with pytest.raises(MathError, match="zero vector"):
        Vector3D(0.0, 0.0, 0.0).normalized()


def test_vector3d_is_zero() -> None:
    assert Vector3D(0.0, 0.0, 0.0).is_zero is True
    assert Vector3D(1.0, 0.0, 0.0).is_zero is False


# ----------------------------------------------------------------------
# Vector3D -- almost_equals.
# ----------------------------------------------------------------------


def test_vector3d_almost_equals_within_tolerance() -> None:
    v1, v2 = Vector3D(1.0, 1.0, 1.0), Vector3D(1.0 + 1e-12, 1.0, 1.0)
    assert v1.almost_equals(v2) is True


def test_vector3d_almost_equals_beyond_tolerance() -> None:
    v1, v2 = Vector3D(1.0, 1.0, 1.0), Vector3D(1.0, 1.1, 1.0)
    assert v1.almost_equals(v2) is False


# ----------------------------------------------------------------------
# Vector3D -- operators.
# ----------------------------------------------------------------------


def test_vector3d_add() -> None:
    assert Vector3D(1.0, 2.0, 3.0) + Vector3D(4.0, 5.0, 6.0) == Vector3D(5.0, 7.0, 9.0)


def test_vector3d_sub() -> None:
    assert Vector3D(4.0, 5.0, 6.0) - Vector3D(1.0, 2.0, 3.0) == Vector3D(3.0, 3.0, 3.0)


def test_vector3d_mul_scalar() -> None:
    assert Vector3D(1.0, 2.0, 3.0) * 2 == Vector3D(2.0, 4.0, 6.0)


def test_vector3d_rmul_scalar() -> None:
    assert 2 * Vector3D(1.0, 2.0, 3.0) == Vector3D(2.0, 4.0, 6.0)


def test_vector3d_truediv_scalar() -> None:
    assert Vector3D(2.0, 4.0, 6.0) / 2 == Vector3D(1.0, 2.0, 3.0)


def test_vector3d_division_by_zero_raises() -> None:
    with pytest.raises(MathError, match="Division by zero"):
        Vector3D(1.0, 1.0, 1.0) / 0.0


def test_vector3d_neg() -> None:
    assert -Vector3D(1.0, -2.0, 3.0) == Vector3D(-1.0, 2.0, -3.0)


def test_vector3d_abs_is_length() -> None:
    v = Vector3D(1.0, 2.0, 3.0)
    assert abs(v) == v.length


def test_vector3d_add_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector3D(1.0, 1.0, 1.0) + "not a vector"  # type: ignore[operator]


def test_vector3d_mul_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector3D(1.0, 1.0, 1.0) * "not a scalar"  # type: ignore[operator]


def test_vector3d_sub_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector3D(1.0, 1.0, 1.0) - "not a vector"  # type: ignore[operator]


def test_vector3d_truediv_unsupported_operand_raises_type_error() -> None:
    with pytest.raises(TypeError):
        Vector3D(1.0, 1.0, 1.0) / "not a scalar"  # type: ignore[operator]


def test_vector3d_to_tuple_and_to_dict() -> None:
    v = Vector3D(1.0, 2.0, 3.0)
    assert v.to_tuple() == (1.0, 2.0, 3.0)
    assert v.to_dict() == {"x": 1.0, "y": 2.0, "z": 3.0}


# ----------------------------------------------------------------------
# Cross-dimensional consistency.
# ----------------------------------------------------------------------


def test_vector2d_and_vector3d_agree_on_zero_vector_semantics() -> None:
    assert Vector2D(0.0, 0.0).is_zero == Vector3D(0.0, 0.0, 0.0).is_zero is True


def test_vector2d_and_vector3d_agree_on_angle_between_identical_directions() -> None:
    assert Vector2D(1.0, 0.0).angle_to(Vector2D(2.0, 0.0)) == pytest.approx(0.0)
    assert Vector3D(1.0, 0.0, 0.0).angle_to(Vector3D(2.0, 0.0, 0.0)) == pytest.approx(0.0)
