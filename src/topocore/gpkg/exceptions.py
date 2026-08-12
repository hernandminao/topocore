"""
topocore.gpkg.exceptions
============================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.core.exceptions import TopoCoreError


class GPKGError(TopoCoreError):
    """Base exception for all GeoPackage export errors."""


class GPKGExportError(GPKGError):
    """Raised when the export process itself fails (I/O, CRS resolution, SQLite errors)."""


class GPKGGeometryError(GPKGError):
    """Raised when a Feature's geometry can't be converted to a valid GeoPackage geometry."""


class GPKGValidationError(GPKGError):
    """Raised when a Feature fails validation before being written."""


__all__ = [
    "GPKGError",
    "GPKGExportError",
    "GPKGGeometryError",
    "GPKGValidationError",
]
