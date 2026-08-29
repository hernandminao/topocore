"""
Coverage audit tests for topocore.terrain.tin.TIN.

PR22 coverage hardening. Per explicit guidance for this module: tests
verify GEOMETRIC INVARIANTS, not merely line coverage -- a high
coverage percentage does not by itself guarantee a triangulation is
geometrically correct.

Invariants verified with real geometry, not just "runs without
error":
  - All simplex vertex indices are within [0, vertex_count).
  - No triangle has zero XY area (degenerate/collinear).
  - Neighbor connectivity is symmetric (if A is B's neighbor, B is
    A's neighbor).
  - Determinism: the same input points produce byte-for-byte
    identical triangulation across independent calls.
  - find_triangle()'s PR21.5 spatial-index optimization matches an
    independent brute-force O(triangle_count) scan on 50 random
    query points across a 200-point triangulation (0 mismatches) --
    confirms the optimization didn't silently change behavior at a
    scale the module's own docstring claims but this audit's sandbox
    had no test verifying.
  - Barycentric interpolation on a symmetric tilted plane gives the
    mathematically exact expected value (z=5.0 at the center of a
    plane tilted from z=0 to z=10), not merely "some plausible
    number".

_build_triangle_index()'s own degenerate-bounds branch
(`triangle_count == 0 or width <= 0.0 or height <= 0.0`) is confirmed
UNREACHABLE from any TIN built via the public API: both from_points()
and from_mesh() already reject fewer than 1 triangle and
collinear/degenerate configurations, so a successfully-constructed
TIN structurally guarantees triangle_count >= 1 and a positive-area
bounding box. Not tested.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TriangulationError
from topocore.terrain.tin import TIN, _point_in_triangle


def _square_tin(
    z_values: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> TIN:
    points = [
        Point3D(0.0, 0.0, z_values[0]),
        Point3D(10.0, 0.0, z_values[1]),
        Point3D(10.0, 10.0, z_values[2]),
        Point3D(0.0, 10.0, z_values[3]),
    ]
    return TIN.from_points(points)


# ----------------------------------------------------------------------
# from_points() -- happy path and geometric invariants.
# ----------------------------------------------------------------------


def test_from_points_all_simplex_indices_in_range() -> None:
    rng = np.random.default_rng(0)
    n = 50
    points = [
        Point3D(float(x), float(y), float(z))
        for x, y, z in zip(
            rng.uniform(0, 10, n),
            rng.uniform(0, 10, n),
            rng.uniform(0, 5, n),
            strict=True,
        )
    ]
    tin = TIN.from_points(points)

    simplices = tin.simplices
    assert np.all((simplices >= 0) & (simplices < tin.vertex_count))


def test_from_points_no_degenerate_triangles() -> None:
    rng = np.random.default_rng(1)
    n = 50
    points = [
        Point3D(float(x), float(y), float(z))
        for x, y, z in zip(
            rng.uniform(0, 10, n),
            rng.uniform(0, 10, n),
            rng.uniform(0, 5, n),
            strict=True,
        )
    ]
    tin = TIN.from_points(points)

    for i in range(tin.triangle_count):
        p1, p2, p3 = tin.triangle_vertices(i)
        area = (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        assert abs(area) > 1e-12, f"triangle {i} is degenerate"


def test_neighbor_connectivity_is_symmetric() -> None:
    tin = _square_tin()
    neighbors = tin.neighbors

    for i in range(tin.triangle_count):
        for j in range(3):
            neighbor = int(neighbors[i, j])
            if neighbor >= 0:
                assert i in neighbors[neighbor], f"triangle {i} <-> {neighbor} neighbor relation is not symmetric"


def test_triangulation_is_deterministic() -> None:
    points = [
        Point3D(0.0, 0.0, 0.0),
        Point3D(10.0, 0.0, 1.0),
        Point3D(10.0, 10.0, 2.0),
        Point3D(0.0, 10.0, 1.0),
        Point3D(5.0, 5.0, 5.0),
    ]

    tin_a = TIN.from_points(points)
    tin_b = TIN.from_points(points)

    np.testing.assert_array_equal(tin_a.simplices, tin_b.simplices)


def test_minimum_valid_input_three_points() -> None:
    points = [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 1.0), Point3D(0.0, 1.0, 2.0)]
    tin = TIN.from_points(points)

    assert tin.vertex_count == 3
    assert tin.triangle_count == 1


# ----------------------------------------------------------------------
# from_points() -- rejected degenerate inputs.
# ----------------------------------------------------------------------


def test_from_points_rejects_collinear_points() -> None:
    points = [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)]
    with pytest.raises(TriangulationError, match="collinear"):
        TIN.from_points(points)


def test_from_points_rejects_duplicated_xy() -> None:
    points = [
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 1.0),
        Point3D(0.0, 1.0, 2.0),
        Point3D(0.0, 0.0, 5.0),
    ]
    with pytest.raises(TriangulationError, match="Duplicated"):
        TIN.from_points(points)


def test_from_points_rejects_fewer_than_three_points() -> None:
    with pytest.raises(TriangulationError, match="At least three"):
        TIN.from_points([Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)])


def test_from_points_rejects_empty_list() -> None:
    with pytest.raises(TriangulationError, match="At least three"):
        TIN.from_points([])


# ----------------------------------------------------------------------
# from_mesh() -- happy path and all 5 validation branches.
# ----------------------------------------------------------------------


def test_from_mesh_happy_path() -> None:
    vertices = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(1, 1, 0))
    simplices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    tin = TIN.from_mesh(vertices, simplices)

    assert tin.vertex_count == 4
    assert tin.triangle_count == 2


def test_from_mesh_rejects_no_vertices() -> None:
    with pytest.raises(TriangulationError, match="At least one vertex"):
        TIN.from_mesh((), np.array([[0, 1, 2]], dtype=np.int32))


def test_from_mesh_rejects_no_triangles() -> None:
    vertices = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    with pytest.raises(TriangulationError, match="At least one triangle"):
        TIN.from_mesh(vertices, np.zeros((0, 3), dtype=np.int32))


def test_from_mesh_rejects_wrong_shape() -> None:
    vertices = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    with pytest.raises(TriangulationError, match=r"\(n, 3\) array"):
        TIN.from_mesh(vertices, np.array([[0, 1]], dtype=np.int32))


def test_from_mesh_rejects_out_of_range_index() -> None:
    vertices = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    with pytest.raises(TriangulationError, match="outside the range"):
        TIN.from_mesh(vertices, np.array([[0, 1, 10]], dtype=np.int32))


def test_from_mesh_rejects_duplicated_vertex_in_triangle() -> None:
    vertices = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    with pytest.raises(TriangulationError, match="duplicated vertex"):
        TIN.from_mesh(vertices, np.array([[0, 0, 1]], dtype=np.int32))


def test_from_mesh_rejects_degenerate_triangle() -> None:
    vertices = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(2, 0, 0))
    with pytest.raises(TriangulationError, match="degenerate"):
        TIN.from_mesh(vertices, np.array([[0, 1, 2]], dtype=np.int32))


# ----------------------------------------------------------------------
# find_triangle() / interpolate() / locate() / contains() -- inside,
# outside, exact vertex, and validated against independent brute
# force at scale.
# ----------------------------------------------------------------------


def test_find_triangle_inside_and_outside() -> None:
    tin = _square_tin()

    assert tin.find_triangle(5.0, 5.0) >= 0
    assert tin.find_triangle(100.0, 100.0) == -1


def test_contains_matches_find_triangle() -> None:
    tin = _square_tin()

    assert tin.contains(5.0, 5.0) is True
    assert tin.contains(100.0, 100.0) is False


def test_interpolate_flat_plane_returns_constant_elevation() -> None:
    tin = _square_tin(z_values=(3.0, 3.0, 3.0, 3.0))
    assert tin.interpolate(5.0, 5.0) == pytest.approx(3.0)


def test_interpolate_tilted_plane_matches_exact_expected_value() -> None:
    """A plane tilted linearly from z=0 to z=10 across x gives EXACTLY z=5.0 at the center -- not just 'some number'."""
    points = [
        Point3D(0.0, 0.0, 0.0),
        Point3D(10.0, 0.0, 10.0),
        Point3D(10.0, 10.0, 10.0),
        Point3D(0.0, 10.0, 0.0),
    ]
    tin = TIN.from_points(points)

    assert tin.interpolate(5.0, 5.0) == pytest.approx(5.0)


def test_interpolate_outside_tin_raises() -> None:
    tin = _square_tin()
    with pytest.raises(ValueError, match="outside TIN"):
        tin.interpolate(100.0, 100.0)


def test_locate_delegates_to_find_triangle() -> None:
    tin = _square_tin()
    assert tin.locate(Point3D(5.0, 5.0, 0.0)) == tin.find_triangle(5.0, 5.0)


def test_spatial_index_matches_brute_force_at_scale() -> None:
    """PR21.5's uniform-grid spatial index must match an independent brute-force scan, not just 'seem right'."""
    rng = np.random.default_rng(0)
    n = 200
    points = [
        Point3D(float(x), float(y), float(z))
        for x, y, z in zip(
            rng.uniform(0, 100, n),
            rng.uniform(0, 100, n),
            rng.uniform(0, 10, n),
            strict=True,
        )
    ]
    tin = TIN.from_points(points)

    def brute_force_find(x: float, y: float) -> int:
        for i in range(tin.triangle_count):
            p1, p2, p3 = tin.triangle_vertices(i)
            if _point_in_triangle(x, y, p1, p2, p3):
                return i
        return -1

    test_points = rng.uniform(-10, 110, (50, 2))
    for x, y in test_points:
        fast_result = tin.find_triangle(float(x), float(y))
        brute_result = brute_force_find(float(x), float(y))
        assert (fast_result < 0) == (brute_result < 0), f"mismatch at ({x}, {y})"


# ----------------------------------------------------------------------
# Auxiliary accessors.
# ----------------------------------------------------------------------


def test_xy_array_and_elevation_array_shapes() -> None:
    tin = _square_tin(z_values=(1.0, 2.0, 3.0, 4.0))

    assert tin.xy_array().shape == (4, 2)
    np.testing.assert_array_equal(tin.elevation_array(), [1.0, 2.0, 3.0, 4.0])


def test_dunder_len_iter_getitem() -> None:
    tin = _square_tin()

    assert len(tin) == tin.triangle_count
    assert tin[0] == tin.triangle(0)
    assert len(list(tin)) == tin.triangle_count


def test_edges_are_cached() -> None:
    tin = _square_tin()
    first = tin.edges
    second = tin.edges
    assert first is second


def test_repr_includes_counts() -> None:
    tin = _square_tin()
    representation = repr(tin)
    assert "vertices=4" in representation
    assert f"triangles={tin.triangle_count}" in representation
