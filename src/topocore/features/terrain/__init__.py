"""
topocore.features.terrain
===========================

Terrain feature detectors: breaklines, contours, embankments.

Importing this subpackage registers every detector it defines with
`DetectorRegistry` (via the `DetectorRegistry.register(...)` call at
the bottom of each detector module).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .breaklines import BreaklineDetector

__all__ = ["BreaklineDetector"]
