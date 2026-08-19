"""
Regression suite for topocore.terrain.weights -- PR19.

barycentric_weights verified against known reference points (each
triangle vertex, the centroid, and reproducing a known linear plane
exactly) -- the same values independently confirmed during this
session's audit before any test code was written.
"""

from __future__ import annotations

import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.weights import (
    barycentric_weights,
    inside_triangle,
    inverse_distance_weight,
    point_distance,
    triangle_area,
    triangle_area2,
)


@pytest.fixture
def unit_right_triangle() -> tuple[Point3D, Point3D, Point3D]:
    return Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0)


def test_barycentric_weights_at_each_vertex(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle

    assert barycentric_weights(0, 0, p1, p2, p3) == pytest.approx((1.0, 0.0, 0.0))
    assert barycentric_weights(1, 0, p1, p2, p3) == pytest.approx((0.0, 1.0, 0.0))
    assert barycentric_weights(0, 1, p1, p2, p3) == pytest.approx((0.0, 0.0, 1.0))


def test_barycentric_weights_at_centroid(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle
    weights = barycentric_weights(1 / 3, 1 / 3, p1, p2, p3)
    assert weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_barycentric_weights_always_sum_to_one(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle
    for x, y in [(0.2, 0.3), (0.5, 0.1), (0.1, 0.1)]:
        w1, w2, w3 = barycentric_weights(x, y, p1, p2, p3)
        assert w1 + w2 + w3 == pytest.approx(1.0)


def test_barycentric_weights_reproduce_known_linear_plane() -> None:
    # z = 10 + 10*x + 20*y, exactly fitting p1/p2/p3.
    p1, p2, p3 = Point3D(0, 0, 10), Point3D(1, 0, 20), Point3D(0, 1, 30)
    w1, w2, w3 = barycentric_weights(0.5, 0.25, p1, p2, p3)

    interpolated_z = w1 * p1.z + w2 * p2.z + w3 * p3.z
    expected_z = 10 + 10 * 0.5 + 20 * 0.25

    assert interpolated_z == pytest.approx(expected_z)


def test_triangle_area_right_triangle_legs_1_1(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle
    assert triangle_area(p1, p2, p3) == pytest.approx(0.5)


def test_triangle_area2_is_twice_triangle_area(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle
    assert abs(triangle_area2(p1, p2, p3)) == pytest.approx(2.0 * triangle_area(p1, p2, p3))


def test_inside_triangle_true_for_interior_point(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle
    weights = barycentric_weights(0.2, 0.2, p1, p2, p3)
    assert inside_triangle(weights) is True


def test_inside_triangle_false_for_exterior_point(
    unit_right_triangle: tuple[Point3D, Point3D, Point3D],
) -> None:
    p1, p2, p3 = unit_right_triangle
    weights = barycentric_weights(5.0, 5.0, p1, p2, p3)
    assert inside_triangle(weights) is False


def test_point_distance_known_3_4_5() -> None:
    assert point_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


def test_inverse_distance_weight_zero_distance_is_infinite() -> None:
    assert inverse_distance_weight(0.0, 2.0) == float("inf")


def test_inverse_distance_weight_known_value() -> None:
    assert inverse_distance_weight(2.0, 2.0) == pytest.approx(0.25)


def test_inverse_distance_weight_decreases_with_distance() -> None:
    near = inverse_distance_weight(1.0, 2.0)
    far = inverse_distance_weight(10.0, 2.0)
    assert near > far
