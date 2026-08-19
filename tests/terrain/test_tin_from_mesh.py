"""
Tests for TIN.from_mesh().

Synthetic geometry with analytically known answers, per project
convention -- not mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TriangulationError
from topocore.terrain.tin import TIN


def _two_triangle_square() -> tuple[tuple[Point3D, ...], np.ndarray]:
    """
    A unit square split into two triangles along the (0,0)-(1,1)
    diagonal:

        2---3
        |  /|
        | / |
        |/  |
        0---1

    Triangle 0: (0, 1, 2)   -- lower-left
    Triangle 1: (1, 3, 2)   -- upper-right

    They share edge (1, 2). This is intentionally NOT what
    scipy.spatial.Delaunay would necessarily choose as the diagonal
    (Delaunay could equally pick the (0,3) diagonal for a perfect
    square, since both are valid/incircle-tied) -- from_mesh must
    preserve exactly this one.
    """
    vertices = (
        Point3D(0.0, 0.0, 0.0),  # 0
        Point3D(1.0, 0.0, 1.0),  # 1
        Point3D(0.0, 1.0, 2.0),  # 2
        Point3D(1.0, 1.0, 3.0),  # 3
    )
    simplices = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=np.int32,
    )
    return vertices, simplices


def test_from_mesh_preserves_exact_vertices_and_simplices() -> None:
    vertices, simplices = _two_triangle_square()

    tin = TIN.from_mesh(vertices, simplices)

    assert tin.vertices == vertices
    assert tin.vertex_count == 4
    assert tin.triangle_count == 2
    np.testing.assert_array_equal(tin.simplices, simplices)


def test_from_mesh_computes_correct_neighbors() -> None:
    """
    Triangle 0 = (0, 1, 2), Triangle 1 = (1, 3, 2).
    Shared edge is (1, 2).

    scipy convention: neighbors[i, j] is the neighbor opposite
    vertex j (i.e. across the edge formed by the other two).

    Triangle 0 = (v0=0, v1=1, v2=2):
      opposite v0 (0) -> edge (1,2) -> shared -> neighbor = 1
      opposite v1 (1) -> edge (0,2) -> boundary -> -1
      opposite v2 (2) -> edge (0,1) -> boundary -> -1

    Triangle 1 = (v0=1, v1=3, v2=2):
      opposite v0 (1) -> edge (3,2) -> boundary -> -1
      opposite v1 (3) -> edge (1,2) -> shared -> neighbor = 0
      opposite v2 (2) -> edge (1,3) -> boundary -> -1
    """
    vertices, simplices = _two_triangle_square()

    tin = TIN.from_mesh(vertices, simplices)

    np.testing.assert_array_equal(tin.neighbors[0], [1, -1, -1])
    np.testing.assert_array_equal(tin.neighbors[1], [-1, 0, -1])


def test_from_mesh_single_triangle_has_no_neighbors() -> None:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0),
    )
    simplices = np.array([[0, 1, 2]], dtype=np.int32)

    tin = TIN.from_mesh(vertices, simplices)

    np.testing.assert_array_equal(tin.neighbors[0], [-1, -1, -1])


def test_from_mesh_preserves_non_delaunay_diagonal() -> None:
    """
    The core regression this exists to prevent: from_mesh must
    reproduce exactly the given faces, not whatever a fresh Delaunay
    triangulation of the same points would choose.

    For a perfect unit square both diagonals are valid Delaunay
    choices; from_mesh must not silently swap to the other one, and
    must not depend on scipy.spatial.Delaunay at all.
    """
    vertices, simplices = _two_triangle_square()

    tin = TIN.from_mesh(vertices, simplices)

    triangle_0 = tin.triangle(0)
    triangle_1 = tin.triangle(1)

    assert (triangle_0.p1, triangle_0.p2, triangle_0.p3) == (
        vertices[0],
        vertices[1],
        vertices[2],
    )
    assert (triangle_1.p1, triangle_1.p2, triangle_1.p3) == (
        vertices[1],
        vertices[3],
        vertices[2],
    )


def test_from_mesh_area_matches_analytic_value() -> None:
    """
    Two right triangles of legs 1x1 -> total XY area = 1.0 exactly.
    """
    vertices, simplices = _two_triangle_square()

    tin = TIN.from_mesh(vertices, simplices)

    total_area = sum(triangle.area for triangle in tin.triangles())

    assert total_area == pytest.approx(1.0)


def test_from_mesh_interpolate_matches_analytic_plane() -> None:
    """
    Z = x + 2y is an exact plane through all four corners of the
    square (0,0,0),(1,0,1),(0,1,2),(1,1,3). Barycentric interpolation
    on either triangle must reproduce it exactly at the centroid.
    """
    vertices, simplices = _two_triangle_square()
    tin = TIN.from_mesh(vertices, simplices)

    # Centroid of triangle 0 = (0,1,2): x=(0+1+0)/3, y=(0+0+1)/3
    x = (0.0 + 1.0 + 0.0) / 3.0
    y = (0.0 + 0.0 + 1.0) / 3.0
    expected_z = x + 2.0 * y

    assert tin.interpolate(x, y) == pytest.approx(expected_z)


def test_from_mesh_rejects_empty_vertices() -> None:
    with pytest.raises(TriangulationError):
        TIN.from_mesh((), np.empty((0, 3), dtype=np.int32))


def test_from_mesh_rejects_empty_simplices() -> None:
    vertices = (Point3D(0.0, 0.0, 0.0),)

    with pytest.raises(TriangulationError):
        TIN.from_mesh(vertices, np.empty((0, 3), dtype=np.int32))


def test_from_mesh_rejects_wrong_shape() -> None:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0),
    )
    bad_simplices = np.array([[0, 1, 2, 0]], dtype=np.int32)  # 4 columns

    with pytest.raises(TriangulationError):
        TIN.from_mesh(vertices, bad_simplices)


def test_from_mesh_rejects_out_of_range_index() -> None:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0),
    )
    simplices = np.array([[0, 1, 3]], dtype=np.int32)  # 3 does not exist

    with pytest.raises(TriangulationError):
        TIN.from_mesh(vertices, simplices)


def test_from_mesh_rejects_negative_index() -> None:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0),
    )
    simplices = np.array([[0, 1, -1]], dtype=np.int32)

    with pytest.raises(TriangulationError):
        TIN.from_mesh(vertices, simplices)


def test_from_mesh_rejects_duplicated_vertex_in_triangle() -> None:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0),
    )
    simplices = np.array([[0, 1, 1]], dtype=np.int32)

    with pytest.raises(TriangulationError):
        TIN.from_mesh(vertices, simplices)


def test_from_mesh_rejects_degenerate_collinear_triangle() -> None:
    vertices = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(2.0, 0.0, 0.0),  # collinear with the first two
    )
    simplices = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(TriangulationError):
        TIN.from_mesh(vertices, simplices)


def test_from_mesh_does_not_mutate_caller_simplices() -> None:
    vertices, simplices = _two_triangle_square()
    original = simplices.copy()

    tin = TIN.from_mesh(vertices, simplices)
    tin.simplices[0, 0] = 99  # mutate the returned defensive copy

    np.testing.assert_array_equal(simplices, original)


def test_from_points_still_uses_delaunay_unaffected_by_from_mesh() -> None:
    """
    Regression guard for the explicit constraint: from_points() must
    remain untouched (points -> Delaunay -> TIN).
    """
    points = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 1.0),
        Point3D(0.0, 1.0, 2.0),
        Point3D(1.0, 1.0, 3.0),
    )

    tin = TIN.from_points(points)

    assert tin.vertex_count == 4
    assert tin.triangle_count == 2
