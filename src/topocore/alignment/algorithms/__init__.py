"""
topocore.alignment.algorithms
===============================

Horizontal alignment geometry algorithms.

Intentionally independent from the public domain models
(``Alignment``, etc.) -- these functions operate on
``HorizontalElement`` values and plain floats/tuples, never on
``Alignment`` itself, so ``topocore.alignment.models`` can import
this package at module level without a circular import (same
convention as ``topocore.terrain.algorithms`` / ``topocore.terrain.tin``).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.alignment.algorithms.horizontal import (
    element_length,
    evaluate_element,
    station_to_point,
)
from topocore.alignment.algorithms.spiral import curvature_at, evaluate_spiral

__all__ = [
    "curvature_at",
    "element_length",
    "evaluate_element",
    "evaluate_spiral",
    "station_to_point",
]
