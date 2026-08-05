"""
topocore.features.utilities
==============================

Utility feature detectors: poles, signs, and light poles.

Importing this subpackage registers all utility detectors with
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

from .light_poles import LightPoleDetector
from .poles import PoleDetector
from .signs import SignDetector

__all__ = [
    "PoleDetector",
    "SignDetector",
    "LightPoleDetector",
]
