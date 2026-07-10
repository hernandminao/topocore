"""
Unit tests for topocore.terrain.models.

Author
------
Hernán Mina

License
-------
MIT
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


class TestEdge:
    def test_length_horizontal(self) -> None:
        edge = Edge(
            Point3D(0.0, 0.0, 0.0),
            Point3D(3.0, 4.0, 0.0),
        )

        assert edge.length == pytest.approx(5.0)

    def test_length_3d(self) -> None:
        edge = Edge(
            Point3D(0.0, 0.0, 0.0),
            Point3D(3.0, 4.0, 12.0),
        )

        assert edge.length == pytest.approx(13.0)

    def test_breakline_flag(self) -> None:
        edge = Edge(
            Point3D(0.0, 0.0, 0.0),
            Point3D(1.0, 1.0, 1.0),
            is_breakline=True,
        )

        assert edge.is_breakline is True


class TestTriangle:
    def test_centroid(self) -> None:
        triangle = Triangle(
            Point3D(0.0, 0.0, 0.0),
            Point3D(3.0, 0.0, 0.0),
            Point3D(0.0, 3.0, 3.0),
        )

        centroid = triangle.centroid

        assert centroid.x == pytest.approx(1.0)
        assert centroid.y == pytest.approx(1.0)
        assert centroid.z == pytest.approx(1.0)

    def test_bbox(self) -> None:
        triangle = Triangle(
            Point3D(5.0, 1.0, 0.0),
            Point3D(1.0, 8.0, 0.0),
            Point3D(4.0, 2.0, 0.0),
        )

        assert triangle.bbox == (
            1.0,
            1.0,
            5.0,
            8.0,
        )

    def test_area(self) -> None:
        triangle = Triangle(
            Point3D(0.0, 0.0, 0.0),
            Point3D(4.0, 0.0, 0.0),
            Point3D(0.0, 3.0, 0.0),
        )

        assert triangle.area == pytest.approx(6.0)


class TestBreakline:
    def test_open_breakline(self) -> None:
        breakline = Breakline(
            id="BL1",
            name="Open",
            points=(
                Point3D(0, 0, 0),
                Point3D(1, 0, 0),
                Point3D(2, 0, 0),
            ),
        )

        assert breakline.is_closed is False
        assert breakline.vertex_count == 3

    def test_closed_breakline(self) -> None:
        p1 = Point3D(0, 0, 0)

        breakline = Breakline(
            id="BL2",
            name="Closed",
            points=(
                p1,
                Point3D(1, 0, 0),
                Point3D(1, 1, 0),
                p1,
            ),
            type=BreaklineType.SOFT,
        )

        assert breakline.is_closed
        assert breakline.vertex_count == 4
        assert breakline.type is BreaklineType.SOFT


class TestGridDefinition:
    def test_grid_dimensions(self) -> None:
        grid = GridDefinition(
            origin_x=0.0,
            origin_y=0.0,
            min_x=0.0,
            min_y=0.0,
            max_x=100.0,
            max_y=50.0,
            resolution=5.0,
        )

        assert grid.width == 21
        assert grid.height == 11

    def test_rotation_default(self) -> None:
        grid = GridDefinition(
            origin_x=0.0,
            origin_y=0.0,
            min_x=0.0,
            min_y=0.0,
            max_x=10.0,
            max_y=10.0,
            resolution=1.0,
        )

        assert grid.rotation == 0.0


class TestContourLine:
    def test_vertex_count(self) -> None:
        contour = ContourLine(
            elevation=1500.0,
            points=(
                Point3D(0, 0, 1500),
                Point3D(1, 0, 1500),
                Point3D(2, 0, 1500),
            ),
        )

        assert contour.vertex_count == 3
        assert contour.closed is False

    def test_closed_contour(self) -> None:
        p = Point3D(0, 0, 100)

        contour = ContourLine(
            elevation=100.0,
            points=(
                p,
                Point3D(1, 0, 100),
                Point3D(1, 1, 100),
                p,
            ),
            closed=True,
        )

        assert contour.closed is True
        assert contour.elevation == 100.0
