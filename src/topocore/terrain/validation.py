"""
topocore.terrain.validation
===========================

Validation utilities for Terrain.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from topocore.geometry.point3d import Point3D

from .exceptions import TerrainValidationError
from .models import Breakline, GridDefinition, Triangle


def _validate_positive(
    value: float,
    name: str,
) -> None:
    """
    Validate a positive finite numeric value.
    """
    if not isinstance(value, (int, float)):
        raise TerrainValidationError(f"{name} must be numeric.")

    if not math.isfinite(value):
        raise TerrainValidationError(f"{name} must be finite.")

    if value <= 0.0:
        raise TerrainValidationError(f"{name} must be greater than zero.")


def validate_resolution(
    resolution: float,
) -> None:
    """
    Validate raster resolution.
    """
    _validate_positive(
        resolution,
        "Resolution",
    )


def validate_interval(
    interval: float,
) -> None:
    """
    Validate contour interval.
    """
    _validate_positive(
        interval,
        "Contour interval",
    )


def validate_points(
    points: Sequence[Point3D],
) -> None:
    """
    Validate a point collection.
    """
    if len(points) == 0:
        raise TerrainValidationError("Point collection cannot be empty.")


def validate_triangle(
    triangle: Triangle,
) -> None:
    """
    Validate a triangle.
    """
    if triangle.p1 == triangle.p2 or triangle.p2 == triangle.p3 or triangle.p1 == triangle.p3:
        raise TerrainValidationError("Triangle contains duplicated vertices.")


def validate_breakline(
    breakline: Breakline,
) -> None:
    """
    Validate a breakline.
    """
    if len(breakline.points) < 2:
        raise TerrainValidationError("Breakline must contain at least two points.")


def validate_grid_definition(
    grid: GridDefinition,
) -> None:
    """
    Validate a raster grid.
    """
    validate_resolution(grid.resolution)

    if grid.max_x <= grid.min_x:
        raise TerrainValidationError("Invalid grid X extent.")

    if grid.max_y <= grid.min_y:
        raise TerrainValidationError("Invalid grid Y extent.")


def validate_tin(
    triangles: Sequence[Triangle],
) -> None:
    """
    Validate a TIN.
    """
    if len(triangles) == 0:
        raise TerrainValidationError("TIN contains no triangles.")

    for triangle in triangles:
        validate_triangle(triangle)


__all__ = [
    "validate_resolution",
    "validate_interval",
    "validate_points",
    "validate_triangle",
    "validate_breakline",
    "validate_grid_definition",
    "validate_tin",
]
