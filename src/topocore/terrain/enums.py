"""
topocore.terrain.enums
======================

Enumerations used by the Terrain module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum


class BreaklineType(StrEnum):
    """
    Defines how a breakline constrains triangulation.
    """

    HARD = "hard"
    SOFT = "soft"


class TriangulationMethod(StrEnum):
    """
    Supported triangulation algorithms.
    """

    DELAUNAY = "delaunay"


class InterpolationMethod(StrEnum):
    """
    Supported interpolation methods.
    """

    LINEAR = "linear"

    NATURAL_NEIGHBOR = "natural_neighbor"

    IDW = "idw"

    BARYCENTRIC = "barycentric"


class ContourSmoothing(StrEnum):
    """
    Contour line smoothing methods.
    """

    NONE = "none"

    CHAIKIN = "chaikin"

    BEZIER = "bezier"


class SlopeMethod(StrEnum):
    """
    Slope computation methods.
    """

    TRIANGLE = "triangle"

    HORN = "horn"

    ZEVENBERGEN_THORNE = "zevenbergen_thorne"


class AspectReference(StrEnum):
    """
    Aspect reference direction.
    """

    NORTH = "north"

    EAST = "east"


__all__ = [
    "BreaklineType",
    "TriangulationMethod",
    "InterpolationMethod",
    "ContourSmoothing",
    "SlopeMethod",
    "AspectReference",
]
