"""
Regression suite for topocore.terrain.models -- PR19.
"""

from __future__ import annotations

import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.enums import BreaklineType
from topocore.terrain.models import (
    Breakline,
    ContourLine,
    Edge,
    GridDefinition,
    Triangle,
)


def test_edge_length_3_4_5_triangle() -> None:
    edge = Edge(start=Point3D(0, 0, 0), end=Point3D(3, 4, 0))
    assert edge.length == pytest.approx(5.0)


def test_edge_length_includes_z() -> None:
    edge = Edge(start=Point3D(0, 0, 0), end=Point3D(0, 0, 5))
    assert edge.length == pytest.approx(5.0)


def test_triangle_area_known_right_triangle() -> None:
    triangle = Triangle(Point3D(0, 0, 0), Point3D(4, 0, 0), Point3D(0, 3, 0))
    assert triangle.area == pytest.approx(6.0)


def test_triangle_area_independent_of_winding_order() -> None:
    forward = Triangle(Point3D(0, 0, 0), Point3D(4, 0, 0), Point3D(0, 3, 0))
    reversed_winding = Triangle(Point3D(0, 0, 0), Point3D(0, 3, 0), Point3D(4, 0, 0))
    assert forward.area == pytest.approx(reversed_winding.area)


def test_triangle_centroid() -> None:
    triangle = Triangle(Point3D(0, 0, 0), Point3D(3, 0, 0), Point3D(0, 3, 0))
    centroid = triangle.centroid
    assert centroid.x == pytest.approx(1.0)
    assert centroid.y == pytest.approx(1.0)


def test_triangle_bbox() -> None:
    triangle = Triangle(Point3D(-1, 5, 0), Point3D(4, -2, 0), Point3D(0, 0, 0))
    assert triangle.bbox == (-1, -2, 4, 5)


def test_breakline_is_closed_true_when_first_equals_last() -> None:
    points = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0), Point3D(0, 0, 0))
    breakline = Breakline(id="b1", name="loop", points=points)
    assert breakline.is_closed is True


def test_breakline_is_closed_false_for_open_chain() -> None:
    points = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0))
    breakline = Breakline(id="b1", name="open", points=points)
    assert breakline.is_closed is False


def test_breakline_is_closed_false_for_only_two_points() -> None:
    """
    Two identical points would technically satisfy first==last, but
    a 2-point "loop" isn't a meaningful closed shape -- is_closed
    requires > 2 points.
    """
    points = (Point3D(0, 0, 0), Point3D(0, 0, 0))
    breakline = Breakline(id="b1", name="degenerate", points=points)
    assert breakline.is_closed is False


def test_breakline_vertex_count() -> None:
    points = (Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0))
    assert Breakline(id="b1", name="x", points=points).vertex_count == 3


def test_breakline_default_type_is_hard() -> None:
    breakline = Breakline(id="b1", name="x", points=(Point3D(0, 0, 0), Point3D(1, 0, 0)))
    assert breakline.type == BreaklineType.HARD


def test_grid_definition_width_height() -> None:
    grid = GridDefinition(origin_x=0, origin_y=0, min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    assert grid.width == 6
    assert grid.height == 6


def test_contour_line_vertex_count() -> None:
    points = (Point3D(0, 0, 5), Point3D(1, 0, 5), Point3D(1, 1, 5))
    contour = ContourLine(elevation=5.0, points=points)
    assert contour.vertex_count == 3


def test_contour_line_default_not_closed() -> None:
    contour = ContourLine(elevation=5.0, points=(Point3D(0, 0, 5),))
    assert contour.closed is False
