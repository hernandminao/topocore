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

from topocore.linalg.vector3d import Vector3D
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
        raise TerrainError(
            "Cannot compute a facet normal for a "
            "degenerate (zero-area) triangle."
        )

    if normal.z < 0.0:
        normal = Vector3D(-normal.x, -normal.y, -normal.z)

    return normal


__all__ = [
    "oriented_normal",
]
