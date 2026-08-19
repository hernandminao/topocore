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

    Single source of truth for this enum -- previously,
    ``topocore.terrain.interpolation`` also declared its own
    ``InterpolationMethod`` (same name, overlapping-but-different
    members: ``NEAREST`` instead of ``NATURAL_NEIGHBOR``, which was
    never implemented anywhere in the codebase). Since
    ``TerrainInterpolator.interpolate()`` compared method values
    with ``is`` (identity), not ``==`` (value), passing THIS enum's
    ``LINEAR`` (a natural import path, since every other Terrain
    enum lives here) silently fell through to NEAREST interpolation
    instead of LINEAR -- with no error, just a wrong elevation.
    Confirmed via a real Delaunay TIN (session audit): 7.5 (correct
    linear) vs. 10.0 (silently NEAREST) for the same query point.
    Consolidated here (PR19); ``interpolation.py`` now imports this
    definition rather than declaring its own.
    """

    LINEAR = "linear"

    BARYCENTRIC = "barycentric"

    IDW = "idw"

    NEAREST = "nearest"


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
    "AspectReference",
    "BreaklineType",
    "ContourSmoothing",
    "InterpolationMethod",
    "SlopeMethod",
    "TriangulationMethod",
]
