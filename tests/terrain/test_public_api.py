"""
Unit tests for topocore.terrain public API.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore import terrain


class TestPublicAPI:
    def test_public_classes(self) -> None:
        assert terrain.Edge is not None
        assert terrain.Triangle is not None
        assert terrain.Breakline is not None
        assert terrain.ContourLine is not None
        assert terrain.GridDefinition is not None

    def test_public_enums(self) -> None:
        assert terrain.BreaklineType is not None

    def test_public_base_classes(self) -> None:
        assert terrain.BaseTIN is not None
        assert terrain.BaseDTM is not None
        assert terrain.BaseInterpolator is not None

    def test_public_exceptions(self) -> None:
        assert terrain.TerrainError is not None
        assert terrain.TerrainValidationError is not None
        assert terrain.TriangulationError is not None
        assert terrain.InterpolationError is not None
        assert terrain.ContourError is not None
        assert terrain.BreaklineError is not None

    def test_public_types(self) -> None:
        assert terrain.Elevation is not None
        assert terrain.Resolution is not None
        assert terrain.Interval is not None
        assert terrain.Slope is not None
        assert terrain.Aspect is not None

    def test_public_constants(self) -> None:
        assert terrain.EPSILON > 0.0
        assert terrain.DEFAULT_GRID_ROTATION == 0.0

    def test_public_validation_functions(self) -> None:
        assert callable(terrain.validate_resolution)
        assert callable(terrain.validate_interval)
        assert callable(terrain.validate_points)
        assert callable(terrain.validate_triangle)
        assert callable(terrain.validate_breakline)
        assert callable(terrain.validate_grid_definition)
        assert callable(terrain.validate_tin)