"""
topocore.terrain._geometry
============================

Internal geometry helpers shared across the Terrain module.

This module is private: it is not part of the public API and exists
only to avoid duplicating facet-normal computation between modules
that need it (``slope.py``, ``aspect.py``, and future consumers such
as ``hillshade.py``).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import math

from topocore.linalg.vector3d import Vector3D
from topocore.math.tolerance import is_zero
from topocore.terrain.exceptions import TerrainError
from topocore.terrain.models import Triangle
from topocore.terrain.validation import validate_triangle


def oriented_normal(triangle: Triangle) -> Vector3D:
    """
    Compute a triangle's normal vector, oriented so that z >= 0.

    Triangulation winding order is not guaranteed to be consistent,
    so the normal is flipped when necessary to always point upward.
    This keeps slope and aspect independent of vertex order.

    Raises
    ------
    TerrainError
        If the triangle is degenerate (zero area).
    """
    validate_triangle(triangle)

    edge1 = triangle.p1.vector_to(triangle.p2)
    edge2 = triangle.p1.vector_to(triangle.p3)

    normal = edge1.cross(edge2)

    if normal.is_zero:
        raise TerrainError("Cannot compute a facet normal for a degenerate (zero-area) triangle.")

    if normal.z < 0.0:
        normal = Vector3D(-normal.x, -normal.y, -normal.z)

    return normal


def slope_radians(normal: Vector3D) -> float:
    """
    Compute the slope angle, in radians, from an oriented normal.

    Returns
    -------
    float
        Angle in the range [0, pi/2], where 0 is horizontal and
        pi/2 is vertical.
    """
    cosine = normal.z / normal.length
    cosine = max(-1.0, min(1.0, cosine))

    return math.acos(cosine)


def aspect_radians(normal: Vector3D) -> float | None:
    """
    Compute the aspect bearing, in radians, from an oriented normal.

    Returns
    -------
    float or None
        Compass bearing in the range [0, 2*pi), clockwise from
        North, or ``None`` if the facet is flat (no defined
        downslope direction).

    Notes
    -----
    This is the single source of truth for "is this facet flat"
    (``is_zero(normal.x) and is_zero(normal.y)``). ``slope.py``,
    ``aspect.py`` and ``hillshade.py`` all derive their flatness
    handling from this same check, so the three modules can never
    disagree at the tolerance boundary.
    """
    if is_zero(normal.x) and is_zero(normal.y):
        return None

    return math.atan2(normal.x, normal.y) % (2.0 * math.pi)


__all__ = [
    "oriented_normal",
    "slope_radians",
    "aspect_radians",
]
