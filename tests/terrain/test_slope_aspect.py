"""
Regression suite for topocore.terrain.slope / aspect / _geometry --
PR19, module Terrain.

Every case verified against hand-derived analytic geometry (cardinal
descent directions, a known 45-degree plane) before writing the
test, not assumed from the implementation. x=East, y=North matches
the convention already established throughout the alignment/geodesy
work this session.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain._geometry import aspect_radians, oriented_normal, slope_radians
from topocore.terrain.aspect import AspectCalculator, triangle_aspect
from topocore.terrain.exceptions import TerrainError
from topocore.terrain.models import Triangle
from topocore.terrain.slope import SlopeCalculator, triangle_slope
from topocore.terrain.tin import TIN

# ----------------------------------------------------------------------
# Cardinal descent directions -- independently hand-derived normals,
# see session notes for the cross-product-by-hand verification.
# ----------------------------------------------------------------------


def _steep_triangle(descend_toward: str) -> Triangle:
    """
    A triangle with a drop of 5 over a horizontal run of 1
    (slope = arctan(5) ~= 78.69 degrees), descending toward the
    given cardinal direction.
    """
    planes = {
        "east": ((0, 0, 10), (1, 0, 5), (0, 1, 10)),
        "north": ((0, 0, 10), (1, 0, 10), (0, 1, 5)),
        "south": ((0, 0, 10), (1, 0, 10), (0, 1, 15)),
        "west": ((0, 0, 5), (1, 0, 10), (0, 1, 5)),
    }
    p1, p2, p3 = planes[descend_toward]
    return Triangle(Point3D(*p1), Point3D(*p2), Point3D(*p3))


_EXPECTED_STEEP_SLOPE = math.degrees(math.atan(5.0))


@pytest.mark.parametrize(
    ("direction", "expected_aspect"),
    [("east", 90.0), ("north", 0.0), ("south", 180.0), ("west", 270.0)],
)
def test_triangle_aspect_matches_cardinal_direction(direction: str, expected_aspect: float) -> None:
    triangle = _steep_triangle(direction)
    assert triangle_aspect(triangle) == pytest.approx(expected_aspect, abs=1e-6)


@pytest.mark.parametrize("direction", ["east", "north", "south", "west"])
def test_triangle_slope_matches_known_gradient(direction: str) -> None:
    triangle = _steep_triangle(direction)
    assert triangle_slope(triangle) == pytest.approx(_EXPECTED_STEEP_SLOPE, abs=1e-6)


def test_aspect_independent_of_vertex_winding_order() -> None:
    """
    Docstring guarantee: "This keeps slope and aspect independent of
    vertex order." Verified directly, not assumed.
    """
    forward = Triangle(Point3D(0, 0, 10), Point3D(1, 0, 5), Point3D(0, 1, 10))
    reversed_winding = Triangle(Point3D(0, 0, 10), Point3D(0, 1, 10), Point3D(1, 0, 5))

    assert triangle_aspect(forward) == pytest.approx(triangle_aspect(reversed_winding))
    assert triangle_slope(forward) == pytest.approx(triangle_slope(reversed_winding))


# ----------------------------------------------------------------------
# 45-degree plane -- exact known slope.
# ----------------------------------------------------------------------


def test_45_degree_plane_gives_exact_slope() -> None:
    # z = x: rises 1 unit per 1 unit horizontal -> exactly 45 degrees.
    triangle = Triangle(Point3D(0, 0, 0), Point3D(1, 0, 1), Point3D(0, 1, 0))
    assert triangle_slope(triangle) == pytest.approx(45.0, abs=1e-9)


def test_ascending_plane_aspect_is_downslope_not_upslope() -> None:
    """
    z = x ascends toward +x (East) -- the DOWNSLOPE direction (what
    aspect measures) must be West (270), not East.
    """
    triangle = Triangle(Point3D(0, 0, 0), Point3D(1, 0, 1), Point3D(0, 1, 0))
    assert triangle_aspect(triangle) == pytest.approx(270.0, abs=1e-6)


# ----------------------------------------------------------------------
# Flat triangle
# ----------------------------------------------------------------------


def test_flat_triangle_slope_is_zero() -> None:
    triangle = Triangle(Point3D(0, 0, 5), Point3D(1, 0, 5), Point3D(0, 1, 5))
    assert triangle_slope(triangle) == pytest.approx(0.0, abs=1e-9)


def test_flat_triangle_aspect_is_nan() -> None:
    triangle = Triangle(Point3D(0, 0, 5), Point3D(1, 0, 5), Point3D(0, 1, 5))
    assert math.isnan(triangle_aspect(triangle))


# ----------------------------------------------------------------------
# Vertical (near-90 degree) slope and degenerate triangle.
# ----------------------------------------------------------------------


def test_near_vertical_slope_approaches_90() -> None:
    # A very steep drop (1000 over horizontal run of 1).
    triangle = Triangle(Point3D(0, 0, 1000), Point3D(1, 0, 0), Point3D(0, 1, 1000))
    assert triangle_slope(triangle) == pytest.approx(90.0, abs=0.1)


def test_degenerate_zero_area_triangle_raises() -> None:
    # All three points collinear -- zero area, no defined normal.
    triangle = Triangle(Point3D(0, 0, 0), Point3D(1, 0, 1), Point3D(2, 0, 2))
    with pytest.raises(TerrainError):
        oriented_normal(triangle)


# ----------------------------------------------------------------------
# oriented_normal: always points "up" (z >= 0), regardless of winding.
# ----------------------------------------------------------------------


def test_oriented_normal_always_has_nonnegative_z() -> None:
    # Winding that would naturally produce a downward-pointing normal.
    triangle = Triangle(Point3D(0, 0, 0), Point3D(0, 1, 0), Point3D(1, 0, 1))
    normal = oriented_normal(triangle)
    assert normal.z >= 0.0


# ----------------------------------------------------------------------
# aspect_radians / slope_radians -- direct unit tests of the shared
# low-level helpers, not just through the triangle wrappers.
# ----------------------------------------------------------------------


def test_slope_radians_horizontal_normal_is_zero() -> None:
    from topocore.linalg.vector3d import Vector3D

    assert slope_radians(Vector3D(0.0, 0.0, 1.0)) == pytest.approx(0.0)


def test_slope_radians_vertical_normal_is_pi_over_2() -> None:
    from topocore.linalg.vector3d import Vector3D

    assert slope_radians(Vector3D(1.0, 0.0, 0.0)) == pytest.approx(math.pi / 2.0)


def test_aspect_radians_returns_none_for_flat_normal() -> None:
    from topocore.linalg.vector3d import Vector3D

    assert aspect_radians(Vector3D(0.0, 0.0, 1.0)) is None


def test_aspect_radians_wraps_to_positive_range() -> None:
    """
    atan2 alone returns (-pi, pi]; aspect_radians must wrap negative
    results into [0, 2*pi).
    """
    from topocore.linalg.vector3d import Vector3D

    # normal.x < 0 -> raw atan2 would be negative before the % wrap.
    bearing = aspect_radians(Vector3D(-1.0, 1.0, 1.0))
    assert bearing is not None
    assert 0.0 <= bearing < 2.0 * math.pi


# ----------------------------------------------------------------------
# SlopeCalculator / AspectCalculator over a real TIN (integration).
# ----------------------------------------------------------------------


@pytest.fixture
def two_triangle_tin() -> TIN:
    # Same fixed diagonal used throughout this session's TIN.from_mesh
    # tests: triangle 0 = (0,1,2), triangle 1 = (1,3,2).
    vertices = (
        Point3D(0.0, 0.0, 10.0),
        Point3D(1.0, 0.0, 5.0),  # descends toward East from vertex 0
        Point3D(0.0, 1.0, 10.0),
        Point3D(1.0, 1.0, 5.0),
    )
    simplices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


def test_slope_calculator_computes_one_value_per_triangle(
    two_triangle_tin: TIN,
) -> None:
    calculator = SlopeCalculator(two_triangle_tin)
    values = calculator.compute()

    assert values.shape == (2,)
    assert calculator.at(0) == pytest.approx(values[0])


def test_slope_calculator_callable_matches_compute(two_triangle_tin: TIN) -> None:
    calculator = SlopeCalculator(two_triangle_tin)
    np.testing.assert_array_equal(calculator(), calculator.compute())


def test_aspect_calculator_computes_one_value_per_triangle(
    two_triangle_tin: TIN,
) -> None:
    calculator = AspectCalculator(two_triangle_tin)
    values = calculator.compute()

    assert values.shape == (2,)


def test_aspect_calculator_first_triangle_matches_direct_computation(
    two_triangle_tin: TIN,
) -> None:
    calculator = AspectCalculator(two_triangle_tin)
    triangle0 = Triangle(
        two_triangle_tin.vertices[0],
        two_triangle_tin.vertices[1],
        two_triangle_tin.vertices[2],
    )

    assert calculator.at(0) == pytest.approx(triangle_aspect(triangle0))
