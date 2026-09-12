"""
topocore.geodesy.vertical.exceptions
=======================================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geodesy.exceptions import GeodesyError


class GeoidError(GeodesyError):
    """Base exception for geoid grid loading/interpolation errors."""


class MissingGeoidGridError(GeoidError):
    """
    Raised whenever a vertical transformation would need to fall back
    to an uncorrected height because no usable geoid grid is
    available for the requested point.

    This is the safety-net exception this whole package exists to
    guarantee: confirmed directly, before this package was written,
    that plain `pyproj.Transformer` silently returns an unchanged
    height when its own required vertical grid is missing, rather
    than raising. `VerticalTransformer` never does this -- every
    code path that would otherwise produce an uncorrected height
    raises this instead.
    """


__all__ = ["GeoidError", "MissingGeoidGridError"]
