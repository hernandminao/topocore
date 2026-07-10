"""
topocore.terrain.algorithms
===========================

Terrain algorithms.

This package contains low-level computational algorithms used by the
Terrain module. These algorithms are intentionally independent from
the public domain models (TIN, DTM, etc.) so they can be replaced or
extended without affecting the public API.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .constrained_delaunay import ConstrainedDelaunayTriangulator
from .delaunay import DelaunayResult, DelaunayTriangulator

__all__ = [
    "DelaunayResult",
    "DelaunayTriangulator",
    "ConstrainedDelaunayTriangulator",
]
