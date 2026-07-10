"""
topocore.geodesy.exceptions
===========================

Exceptions raised by the Geodesy module.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations


class GeodesyError(Exception):
    """Base exception for all geodesy-related errors."""


class CRSError(GeodesyError):
    """Raised when a Coordinate Reference System is invalid or unsupported."""


class TransformationError(GeodesyError):
    """Raised when a coordinate transformation fails."""


class GeodesicError(GeodesyError):
    """Raised when a geodesic calculation fails."""


class ValidationError(GeodesyError):
    """Raised when coordinate validation fails."""


__all__ = [
    "GeodesyError",
    "CRSError",
    "TransformationError",
    "GeodesicError",
    "ValidationError",
]
