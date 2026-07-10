"""
topocore.terrain.types
======================

Common type aliases used throughout the Terrain module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

###############################################################################
# Basic scalar types
###############################################################################

type Elevation = float

type Resolution = float

type Interval = float

type CellSize = float

###############################################################################
# Terrain analysis
###############################################################################

type Slope = float

type Aspect = float

###############################################################################
# Measurements
###############################################################################

type Area = float

type Volume = float

type Distance = float

###############################################################################
# Coordinates
###############################################################################

type Coordinate = float

__all__ = [
    "Elevation",
    "Resolution",
    "Interval",
    "CellSize",
    "Slope",
    "Aspect",
    "Area",
    "Volume",
    "Distance",
    "Coordinate",
]
