"""
topocore.terrain.constants
==========================

Terrain module constants.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

###############################################################################
# Numeric tolerances
###############################################################################

EPSILON: float = 1e-9

###############################################################################
# Default raster parameters
###############################################################################

DEFAULT_RESOLUTION: float = 1.0

DEFAULT_CONTOUR_INTERVAL: float = 1.0

DEFAULT_NO_DATA: float = float("nan")

###############################################################################
# Terrain limits
###############################################################################

MIN_RESOLUTION: float = EPSILON

MIN_CONTOUR_INTERVAL: float = EPSILON

MAX_TRIANGLE_EDGE_LENGTH: float = 1_000_000.0

###############################################################################
# Slope
###############################################################################

MIN_SLOPE: float = 0.0

MAX_SLOPE: float = 90.0

###############################################################################
# Aspect
###############################################################################

MIN_ASPECT: float = 0.0

MAX_ASPECT: float = 360.0

###############################################################################
# Grid
###############################################################################

DEFAULT_GRID_ROTATION: float = 0.0

###############################################################################
# Exported symbols
###############################################################################

__all__ = [
    "EPSILON",
    "DEFAULT_RESOLUTION",
    "DEFAULT_CONTOUR_INTERVAL",
    "DEFAULT_NO_DATA",
    "MIN_RESOLUTION",
    "MIN_CONTOUR_INTERVAL",
    "MAX_TRIANGLE_EDGE_LENGTH",
    "MIN_SLOPE",
    "MAX_SLOPE",
    "MIN_ASPECT",
    "MAX_ASPECT",
    "DEFAULT_GRID_ROTATION",
]