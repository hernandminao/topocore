"""
topocore.geodesy.vertical
============================

Real geoid-based vertical transformation: `GeoidGrid` (loads and
interpolates a real GeoTIFF geodetic grid) and `VerticalTransformer`
(applies it to convert between ellipsoidal and orthometric height).

Does not bundle, fabricate, or approximate any real geoid model data
-- bring your own GeoTIFF geoid grid (e.g. EGM96/EGM2008, converted
or obtained via PROJ's own tooling). See
`topocore.geodesy.vertical.geoid_grid`'s own module docstring for the
full rationale, including why GDAL is a lazily-imported optional
dependency here.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geodesy.vertical.exceptions import GeoidError, MissingGeoidGridError
from topocore.geodesy.vertical.geoid_grid import GeoidGrid
from topocore.geodesy.vertical.transformer import VerticalTransformer

__all__ = [
    "GeoidError",
    "GeoidGrid",
    "MissingGeoidGridError",
    "VerticalTransformer",
]
