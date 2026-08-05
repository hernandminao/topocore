"""
topocore.features.terrain
===========================

Terrain feature detectors: breaklines, slope changes, embankments,
and contours.

Importing this subpackage registers all terrain detectors with
`DetectorRegistry` through the registration performed by each
detector module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .breaklines import BreaklineDetector
from .contours import ContourDetector
from .embankments import EmbankmentDetector
from .slope_changes import SlopeChangeDetector

__all__ = [
    "BreaklineDetector",
    "SlopeChangeDetector",
    "EmbankmentDetector",
    "ContourDetector",
]
