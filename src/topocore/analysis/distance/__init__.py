"""
topocore.analysis.distance
==========================

Distance analysis sub-package.

Provides Euclidean, geodesic, horizontal, vertical, and slope
distance computations, along with a unified ``DistanceAnalysis``
facade.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .euclidean import EuclideanDistance
from .geodesic import GeodesicDistance
from .horizontal import HorizontalDistance
from .manager import DistanceAnalysis, DistanceMethod
from .slope import SlopeDistance
from .vertical import VerticalDistance

__all__ = [
    "DistanceAnalysis",
    "DistanceMethod",
    "EuclideanDistance",
    "GeodesicDistance",
    "HorizontalDistance",
    "SlopeDistance",
    "VerticalDistance",
]
