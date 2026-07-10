"""
Unit tests for topocore.terrain.validation.

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
from topocore.terrain.exceptions import TerrainValidationError
from topocore.terrain.models import (
    Breakline,
    GridDefinition,
    Triangle,
)
from topocore.terrain.validation import (
    validate_breakline,
    validate_grid_definition,
    validate_interval,
    validate_points,
    validate_resolution,
    validate_tin,
    validate_triangle,
)


class TestResolutionValidation:
    def test_valid_resolution(self) -> None:
        validate_resolution(1.0)

    def test_zero_resolution(self) -> None:
        with pytest.raises(TerrainValidationError):
            validate_resolution(0.0)

    def test_negative_resolution(self) -> None:
        with pytest.raises(TerrainValidationError):
            validate_resolution(-1.0)


class TestIntervalValidation:
    def test_valid_interval(self) -> None:
        validate_interval(0.50)

    def test_invalid_interval(self) -> None:
        with pytest.raises(TerrainValidationError):
            validate_interval(0.0)


class TestPointValidation:
    def test_valid_points(self) -> None:
        validate_points(
            (
                Point3D(0, 0, 0),
                Point3D(1, 1, 1),
            )
        )

    def test_empty_points(self) -> None:
        with pytest.raises(TerrainValidationError):
            validate_points(())


class TestTriangleValidation:
    def test_valid_triangle(self) -> None:
        triangle = Triangle(
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(0, 1, 0),
        )

        validate_triangle(triangle)

    def test_duplicate_vertices(self) -> None:
        triangle = Triangle(
            Point3D(0, 0, 0),
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
        )

        with pytest.raises(TerrainValidationError):
            validate_triangle(triangle)


class TestBreaklineValidation:
    def test_valid_breakline(self) -> None:
        breakline = Breakline(
            id="B1",
            name="Test",
            points=(
                Point3D(0, 0, 0),
                Point3D(1, 0, 0),
            ),
            type=BreaklineType.HARD,
        )

        validate_breakline(breakline)

    def test_invalid_breakline(self) -> None:
        breakline = Breakline(
            id="B1",
            name="Test",
            points=(Point3D(0, 0, 0),),
        )

        with pytest.raises(TerrainValidationError):
            validate_breakline(breakline)


class TestGridValidation:
    def test_valid_grid(self) -> None:
        grid = GridDefinition(
            origin_x=0,
            origin_y=0,
            min_x=0,
            min_y=0,
            max_x=10,
            max_y=10,
            resolution=1,
        )

        validate_grid_definition(grid)

    def test_invalid_extent(self) -> None:
        grid = GridDefinition(
            origin_x=0,
            origin_y=0,
            min_x=10,
            min_y=0,
            max_x=5,
            max_y=10,
            resolution=1,
        )

        with pytest.raises(TerrainValidationError):
            validate_grid_definition(grid)


class TestTINValidation:
    def test_valid_tin(self) -> None:
        triangle = Triangle(
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(0, 1, 0),
        )

        validate_tin((triangle,))

    def test_empty_tin(self) -> None:
        with pytest.raises(TerrainValidationError):
            validate_tin(())
