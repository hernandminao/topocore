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

from typing import TypeAlias

###############################################################################
# Basic scalar types
###############################################################################

Elevation: TypeAlias = float

Resolution: TypeAlias = float

Interval: TypeAlias = float

CellSize: TypeAlias = float

###############################################################################
# Terrain analysis
###############################################################################

Slope: TypeAlias = float

Aspect: TypeAlias = float

###############################################################################
# Measurements
###############################################################################

Area: TypeAlias = float

Volume: TypeAlias = float

Distance: TypeAlias = float

###############################################################################
# Coordinates
###############################################################################

Coordinate: TypeAlias = float

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