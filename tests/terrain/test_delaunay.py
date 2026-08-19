"""
Regression suite for topocore.terrain.algorithms.delaunay -- PR19.

The actual triangulation math is delegated to scipy.spatial.Delaunay
(a mature, independently-tested library); this suite focuses on
TopoCore's own wrapping: input validation, error conversion, and the
small pure-Python helpers (compute_bbox, validate_result).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.algorithms.delaunay import DelaunayResult, DelaunayTriangulator
from topocore.terrain.exceptions import TriangulationError


def test_simple_square_gives_two_triangles() -> None:
    points = (Point3D(0, 0, 1), Point3D(1, 0, 2), Point3D(1, 1, 3), Point3D(0, 1, 4))
    result = DelaunayTriangulator.triangulate(points)

    assert result.triangle_count == 2
    assert result.vertex_count == 4


def test_result_preserves_original_vertex_order_and_z() -> None:
    points = (
        Point3D(0, 0, 10),
        Point3D(1, 0, 20),
        Point3D(1, 1, 30),
        Point3D(0, 1, 40),
    )
    result = DelaunayTriangulator.triangulate(points)

    assert result.vertices == points  # exact tuple identity/order preserved


def test_rejects_fewer_than_three_points() -> None:
    with pytest.raises(TriangulationError):
        DelaunayTriangulator.triangulate((Point3D(0, 0, 0), Point3D(1, 0, 0)))


def test_accepts_exactly_three_points() -> None:
    result = DelaunayTriangulator.triangulate((Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0)))
    assert result.triangle_count == 1


def test_rejects_duplicated_xy_coordinates() -> None:
    points = (
        Point3D(0, 0, 1),
        Point3D(1, 0, 2),
        Point3D(0, 0, 99),  # same XY as the first, different Z
        Point3D(0, 1, 3),
    )
    with pytest.raises(TriangulationError, match="Duplicated"):
        DelaunayTriangulator.triangulate(points)


def test_rejects_fully_collinear_points() -> None:
    points = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(2, 0, 0), Point3D(3, 0, 0))
    with pytest.raises(TriangulationError, match="collinear"):
        DelaunayTriangulator.triangulate(points)


def test_rejects_collinear_points_on_a_diagonal_not_just_axis_aligned() -> None:
    """
    Collinearity detection must work for any line direction, not
    only horizontal/vertical -- matrix-rank based, so this should
    hold automatically, but verified directly rather than assumed.
    """
    points = (Point3D(0, 0, 0), Point3D(1, 1, 0), Point3D(2, 2, 0), Point3D(3, 3, 0))
    with pytest.raises(TriangulationError, match="collinear"):
        DelaunayTriangulator.triangulate(points)


def test_accepts_near_collinear_but_genuinely_2d_points() -> None:
    """
    A tiny but real deviation from a line must NOT be rejected as
    collinear -- confirms the rank check isn't overly aggressive.
    """
    points = (
        Point3D(0, 0, 0),
        Point3D(1, 0.001, 0),
        Point3D(2, 0, 0),
        Point3D(3, 0.001, 0),
    )
    result = DelaunayTriangulator.triangulate(points)
    assert result.triangle_count > 0


# ----------------------------------------------------------------------
# compute_bbox
# ----------------------------------------------------------------------


def test_compute_bbox() -> None:
    points = (Point3D(-5, 2, 0), Point3D(10, -3, 0), Point3D(0, 8, 0))
    bbox = DelaunayTriangulator.compute_bbox(points)
    assert bbox == (-5, -3, 10, 8)


# ----------------------------------------------------------------------
# validate_result
# ----------------------------------------------------------------------


def _valid_result() -> DelaunayResult:
    points = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    return DelaunayTriangulator.triangulate(points)


def test_validate_result_accepts_genuine_output() -> None:
    DelaunayTriangulator.validate_result(_valid_result())  # must not raise


def test_validate_result_rejects_wrong_simplex_width() -> None:
    result = _valid_result()
    bad = DelaunayResult(
        vertices=result.vertices,
        simplices=np.array([[0, 1]], dtype=np.int32),  # 2 columns, not 3
        neighbors=result.neighbors,
    )
    with pytest.raises(TriangulationError):
        DelaunayTriangulator.validate_result(bad)


def test_validate_result_rejects_mismatched_neighbors_shape() -> None:
    result = _valid_result()
    bad = DelaunayResult(
        vertices=result.vertices,
        simplices=result.simplices,
        neighbors=np.array([[-1, -1]], dtype=np.int32),  # wrong shape
    )
    with pytest.raises(TriangulationError):
        DelaunayTriangulator.validate_result(bad)


def test_validate_result_rejects_negative_vertex_index() -> None:
    result = _valid_result()
    bad = DelaunayResult(
        vertices=result.vertices,
        simplices=np.array([[-1, 0, 1]], dtype=np.int32),
        neighbors=result.neighbors,
    )
    with pytest.raises(TriangulationError):
        DelaunayTriangulator.validate_result(bad)


def test_validate_result_rejects_out_of_range_vertex_index() -> None:
    result = _valid_result()
    bad = DelaunayResult(
        vertices=result.vertices,  # only 3 vertices, indices 0-2 valid
        simplices=np.array([[0, 1, 5]], dtype=np.int32),
        neighbors=result.neighbors,
    )
    with pytest.raises(TriangulationError):
        DelaunayTriangulator.validate_result(bad)


# ----------------------------------------------------------------------
# triangle_vertices / neighbor_indices
# ----------------------------------------------------------------------


def test_triangle_vertices_matches_simplices() -> None:
    """
    scipy.spatial.Delaunay does not guarantee which vertex starts
    the triple -- only that it's a valid cyclic rotation of the
    triangle's actual vertices. Checking as a set (not exact tuple
    order) avoids over-asserting an implementation detail scipy
    doesn't promise.
    """
    result = _valid_result()
    assert set(DelaunayTriangulator.triangle_vertices(result, 0)) == {0, 1, 2}


def test_neighbor_indices_all_boundary_for_single_triangle() -> None:
    """
    A single triangle has no neighbors -- every edge is on the
    convex hull boundary (-1).
    """
    result = _valid_result()
    assert DelaunayTriangulator.neighbor_indices(result, 0) == (-1, -1, -1)
