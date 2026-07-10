"""
Tests for topocore.terrain.tin.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain import TIN
from topocore.terrain.models import Edge
from topocore.terrain.models import Triangle


@pytest.fixture
def sample_points() -> tuple[Point3D, ...]:
    return (
        Point3D(0.0, 0.0, 10.0),
        Point3D(10.0, 0.0, 11.0),
        Point3D(10.0, 10.0, 12.0),
        Point3D(0.0, 10.0, 13.0),
        Point3D(5.0, 5.0, 20.0),
    )


@pytest.fixture
def tin(sample_points: tuple[Point3D, ...]) -> TIN:
    return TIN.from_points(sample_points)

def test_from_points(tin: TIN) -> None:
    assert isinstance(tin, TIN)

def test_vertices(tin: TIN) -> None:
    vertices = tin.vertices

    assert isinstance(vertices, tuple)
    assert len(vertices) == 5
    assert all(isinstance(v, Point3D) for v in vertices)

def test_vertex_count(tin: TIN) -> None:
    assert tin.vertex_count == 5

def test_triangle_count(tin: TIN) -> None:
    assert tin.triangle_count > 0

def test_edge_count(tin: TIN) -> None:
    assert tin.edge_count > 0

def test_simplices_returns_copy(tin: TIN) -> None:
    simplices1 = tin.simplices
    simplices2 = tin.simplices

    assert simplices1 is not simplices2

    np.testing.assert_array_equal(
        simplices1,
        simplices2,
    )

def test_neighbors_returns_copy(tin: TIN) -> None:
    n1 = tin.neighbors
    n2 = tin.neighbors

    assert n1 is not n2

    np.testing.assert_array_equal(
        n1,
        n2,
    )

def test_bounds(tin: TIN) -> None:
    assert tin.bounds == (
        0.0,
        0.0,
        10.0,
        10.0,
    )

def test_triangle(tin: TIN) -> None:
    triangle = tin.triangle(0)

    assert isinstance(triangle, Triangle)

def test_triangle_invalid_index(tin: TIN) -> None:
    with pytest.raises(IndexError):
        tin.triangle(999)

def test_triangle_vertices(tin: TIN) -> None:
    vertices = tin.triangle_vertices(0)

    assert len(vertices) == 3

    assert all(
        isinstance(v, Point3D)
        for v in vertices
    )

def test_triangles_generator(tin: TIN) -> None:
    triangles = list(tin.triangles())

    assert len(triangles) == tin.triangle_count

    assert all(
        isinstance(t, Triangle)
        for t in triangles
    )

def test_edges(tin: TIN) -> None:
    edges = tin.edges

    assert isinstance(edges, tuple)

    assert all(
        isinstance(edge, Edge)
        for edge in edges
    )

def test_edges_are_cached(tin: TIN) -> None:
    assert tin.edges is tin.edges

def test_edge_vertices(tin: TIN) -> None:
    edge = tin.edges[0]

    start, end = tin.edge_vertices(edge)

    assert start == edge.start
    assert end == edge.end

def test_neighbors_of(tin: TIN) -> None:
    neighbors = tin.neighbors_of(0)

    assert len(neighbors) == 3

def test_find_triangle_inside(tin: TIN) -> None:
    index = tin.find_triangle(
        5.0,
        5.0,
    )

    assert index >= 0

def test_find_triangle_outside(tin: TIN) -> None:
    assert tin.find_triangle(
        100.0,
        100.0,
    ) == -1

def test_locate(tin: TIN) -> None:
    point = Point3D(
        5.0,
        5.0,
        0.0,
    )

    assert tin.locate(point) >= 0

def test_contains_true(tin: TIN) -> None:
    assert tin.contains(
        5.0,
        5.0,
    )

def test_contains_false(tin: TIN) -> None:
    assert not tin.contains(
        100.0,
        100.0,
    )

def test_iter(tin: TIN) -> None:
    triangles = list(iter(tin))

    assert len(triangles) == tin.triangle_count

def test_getitem(tin: TIN) -> None:
    triangle = tin[0]

    assert isinstance(
        triangle,
        Triangle,
    )

def test_len(tin: TIN) -> None:
    assert len(tin) == tin.triangle_count

def test_repr(tin: TIN) -> None:
    text = repr(tin)

    assert "TIN" in text
    assert "vertices=" in text
    assert "triangles=" in text
    assert "edges=" in text

def test_vertex_array(tin: TIN) -> None:
    array = tin.vertex_array()

    assert array.shape == (
        tin.vertex_count,
        3,
    )

    assert array.dtype == np.float64

def test_xy_array(tin: TIN) -> None:
    array = tin.xy_array()

    assert array.shape == (
        tin.vertex_count,
        2,
    )

    assert array.dtype == np.float64

def test_elevation_array(tin: TIN) -> None:
    array = tin.elevation_array()

    assert array.shape == (
        tin.vertex_count,
    )

    assert array.dtype == np.float64
