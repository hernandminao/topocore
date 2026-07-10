"""
topocore.terrain.exceptions
===========================

Exceptions raised by the Terrain module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


class TerrainError(Exception):
    """
    Base exception for all terrain-related errors.
    """


class TriangulationError(TerrainError):
    """
    Raised when a triangulation operation fails.
    """


class InterpolationError(TerrainError):
    """
    Raised when a terrain interpolation fails.
    """


class ContourError(TerrainError):
    """
    Raised when contour generation fails.
    """


class BreaklineError(TerrainError):
    """
    Raised when a breakline is invalid or cannot be enforced.
    """


class GridError(TerrainError):
    """
    Raised when a raster grid definition is invalid.
    """


class TerrainValidationError(TerrainError):
    """
    Raised when terrain validation fails.
    """


__all__ = [
    "TerrainError",
    "TriangulationError",
    "InterpolationError",
    "ContourError",
    "BreaklineError",
    "GridError",
    "TerrainValidationError",
]