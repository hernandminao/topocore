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

from enum import Enum


class BreaklineType(str, Enum):
    """
    Defines how a breakline constrains triangulation.
    """

    HARD = "hard"
    SOFT = "soft"


class TriangulationMethod(str, Enum):
    """
    Supported triangulation algorithms.
    """

    DELAUNAY = "delaunay"


class InterpolationMethod(str, Enum):
    """
    Supported interpolation methods.
    """

    LINEAR = "linear"

    NATURAL_NEIGHBOR = "natural_neighbor"

    IDW = "idw"

    BARYCENTRIC = "barycentric"


class ContourSmoothing(str, Enum):
    """
    Contour line smoothing methods.
    """

    NONE = "none"

    CHAIKIN = "chaikin"

    BEZIER = "bezier"


class SlopeMethod(str, Enum):
    """
    Slope computation methods.
    """

    TRIANGLE = "triangle"

    HORN = "horn"

    ZEVENBERGEN_THORNE = "zevenbergen_thorne"


class AspectReference(str, Enum):
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