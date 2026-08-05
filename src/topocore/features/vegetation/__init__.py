"""
topocore.features.vegetation
===============================

Vegetation feature detectors: trees, shrubs, grass.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .grass import GrassDetector
from .shrubs import ShrubDetector
from .trees import TreeDetector

__all__ = [
    "TreeDetector",
    "ShrubDetector",
    "GrassDetector",
]
