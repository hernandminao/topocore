"""
Regression suite for topocore.terrain.validation -- PR19.
"""

from __future__ import annotations

import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TerrainValidationError
from topocore.terrain.models import Breakline, GridDefinition, Triangle
from topocore.terrain.validation import (
    validate_breakline,
    validate_grid_definition,
    validate_interval,
    validate_points,
    validate_resolution,
    validate_tin,
    validate_triangle,
)


def test_validate_resolution_accepts_positive() -> None:
    validate_resolution(1.0)  # must not raise


def test_validate_resolution_rejects_zero() -> None:
    with pytest.raises(TerrainValidationError):
        validate_resolution(0.0)


def test_validate_resolution_rejects_negative() -> None:
    with pytest.raises(TerrainValidationError):
        validate_resolution(-1.0)


def test_validate_resolution_rejects_nan() -> None:
    with pytest.raises(TerrainValidationError):
        validate_resolution(float("nan"))


def test_validate_resolution_rejects_infinity() -> None:
    with pytest.raises(TerrainValidationError):
        validate_resolution(float("inf"))


def test_validate_interval_same_rules_as_resolution() -> None:
    validate_interval(2.5)  # must not raise
    with pytest.raises(TerrainValidationError):
        validate_interval(-2.5)


def test_validate_points_rejects_empty() -> None:
    with pytest.raises(TerrainValidationError):
        validate_points([])


def test_validate_points_accepts_nonempty() -> None:
    validate_points([Point3D(0, 0, 0)])  # must not raise


def test_validate_triangle_rejects_duplicated_vertices() -> None:
    triangle = Triangle(Point3D(0, 0, 0), Point3D(0, 0, 0), Point3D(1, 1, 0))
    with pytest.raises(TerrainValidationError):
        validate_triangle(triangle)


def test_validate_triangle_accepts_genuine_triangle() -> None:
    triangle = Triangle(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    validate_triangle(triangle)  # must not raise


def test_validate_breakline_rejects_single_point() -> None:
    with pytest.raises(TerrainValidationError):
        validate_breakline(Breakline(id="b", name="x", points=(Point3D(0, 0, 0),)))


def test_validate_breakline_accepts_two_points() -> None:
    breakline = Breakline(id="b", name="x", points=(Point3D(0, 0, 0), Point3D(1, 0, 0)))
    validate_breakline(breakline)  # must not raise


def test_validate_grid_definition_rejects_invalid_x_extent() -> None:
    grid = GridDefinition(origin_x=0, origin_y=0, min_x=10, min_y=0, max_x=5, max_y=10, resolution=1.0)
    with pytest.raises(TerrainValidationError):
        validate_grid_definition(grid)


def test_validate_grid_definition_rejects_invalid_y_extent() -> None:
    grid = GridDefinition(origin_x=0, origin_y=0, min_x=0, min_y=10, max_x=10, max_y=5, resolution=1.0)
    with pytest.raises(TerrainValidationError):
        validate_grid_definition(grid)


def test_validate_grid_definition_rejects_invalid_resolution() -> None:
    grid = GridDefinition(origin_x=0, origin_y=0, min_x=0, min_y=0, max_x=10, max_y=10, resolution=-1.0)
    with pytest.raises(TerrainValidationError):
        validate_grid_definition(grid)


def test_validate_grid_definition_accepts_valid_grid() -> None:
    grid = GridDefinition(origin_x=0, origin_y=0, min_x=0, min_y=0, max_x=10, max_y=10, resolution=1.0)
    validate_grid_definition(grid)  # must not raise


def test_validate_tin_rejects_empty_triangle_list() -> None:
    with pytest.raises(TerrainValidationError):
        validate_tin([])


def test_validate_tin_rejects_any_degenerate_triangle() -> None:
    good = Triangle(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    bad = Triangle(Point3D(0, 0, 0), Point3D(0, 0, 0), Point3D(1, 1, 0))
    with pytest.raises(TerrainValidationError):
        validate_tin([good, bad])


def test_validate_tin_accepts_valid_triangles() -> None:
    triangles = [Triangle(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))]
    validate_tin(triangles)  # must not raise
