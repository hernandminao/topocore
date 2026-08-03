"""
topocore.analysis._shared.surface
==================================

Shared helpers for terrain surface operations.

Provides common validation and interpolation logic used by
profile, volume, visibility, and other terrain analysis modules.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.analysis.exceptions import ProfileError
from topocore.analysis.protocols import TerrainSurface


def validate_coordinate(
    value: float,
    name: str,
) -> None:
    """
    Validate that a coordinate value is finite.

    Parameters
    ----------
    value
        Coordinate value.
    name
        Coordinate name.

    Raises
    ------
    ProfileError
        If the value is NaN or infinite.
    """
    if not math.isfinite(value):
        raise ProfileError(f"Coordinate '{name}' must be finite; got {value}.")


def validate_xy(
    x: float,
    y: float,
) -> None:
    """
    Validate XY coordinates.

    Raises
    ------
    ProfileError
        If coordinates are invalid.
    """
    validate_coordinate(x, "x")
    validate_coordinate(y, "y")


def interpolate_surface(
    surface: TerrainSurface,
    x: float,
    y: float,
) -> float:
    """
    Interpolate elevation from a terrain surface.

    Parameters
    ----------
    surface
        Terrain surface implementing ``TerrainSurface`` protocol.
    x
        X coordinate.
    y
        Y coordinate.

    Returns
    -------
    float
        Interpolated elevation.

    Raises
    ------
    ProfileError
        If interpolation fails.
    """
    validate_xy(x, y)

    try:
        elevation = surface.interpolate(x, y)
    except Exception as exc:
        raise ProfileError(f"Surface interpolation failed at ({x}, {y}).") from exc

    if not math.isfinite(elevation):
        raise ProfileError(f"Interpolated elevation must be finite; got {elevation}.")

    return float(elevation)


__all__ = [
    "validate_coordinate",
    "validate_xy",
    "interpolate_surface",
]
