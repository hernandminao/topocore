"""
topocore.processing.ground
==========================

Ground classification and extraction.

This package provides various methods for classifying and extracting
ground points from point clouds:

- Grid: Simple grid-based classification (fast, baseline)
- Progressive TIN: Iterative TIN densification (robust, widely used)
- CSF: Cloth Simulation Filter
- PMF: Progressive Morphological filtering
- Hybrid: Combined methods (future)

Ground classification is essential for terrain modeling, DTM generation,
and earthwork calculations.

Public API
----------
- GroundClassifier: abstract base class for ground classifiers
- GroundExtractor: abstract base class for ground extractors
- GroundElevationEstimator: abstract base class for ground elevation estimation
- GridGroundClassifier: grid-based classification
- GridGroundExtractor: grid-based extraction
- ProgressiveTINGroundClassifier: Progressive TIN classification
- ProgressiveTINGroundExtractor: Progressive TIN extraction
- GroundManager: high-level manager with method selection

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import GroundClassifier, GroundElevationEstimator, GroundExtractor
from .csf import CSFGroundClassifier, CSFGroundExtractor
from .grid import (
    GridGroundClassifier,
    GridGroundElevationEstimator,
    GridGroundExtractor,
)
from .grid_adaptive import (
    AdaptiveGridGroundClassifier,
    AdaptiveGridGroundExtractor,
)
from .manager import GroundManager
from .pmf import PMFGroundClassifier, PMFGroundExtractor
from .progressive_tin import (
    ProgressiveTINGroundClassifier,
    ProgressiveTINGroundExtractor,
)

__all__ = [
    "GroundClassifier",
    "GroundExtractor",
    "GroundElevationEstimator",
    "GridGroundClassifier",
    "GridGroundExtractor",
    "GridGroundElevationEstimator",
    "AdaptiveGridGroundClassifier",
    "AdaptiveGridGroundExtractor",
    "ProgressiveTINGroundClassifier",
    "ProgressiveTINGroundExtractor",
    "PMFGroundClassifier",
    "PMFGroundExtractor",
    "CSFGroundClassifier",
    "CSFGroundExtractor",
    "GroundManager",
]
