"""
topocore.gpkg
================

GeoPackage (OGC) export for `FeatureCollection` -- PR17a. Native
implementation via `sqlite3` (stdlib) + `shapely` WKB (already a
mandatory TopoCore dependency); no GDAL/fiona/pyogrio.

PR17b (raw PointCloud -> GeoPackage feature table, with
classification/intensity/RGB) is explicitly out of scope here -- see
`gpkg.exporter`'s module docstring and the project backlog.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.gpkg.config import GPKGExportOptions
from topocore.gpkg.exceptions import (
    GPKGError,
    GPKGExportError,
    GPKGGeometryError,
    GPKGValidationError,
)
from topocore.gpkg.exporter import GeoPackageExporter
from topocore.gpkg.report import GPKGExportReport

__all__ = [
    "GPKGError",
    "GPKGExportError",
    "GPKGExportOptions",
    "GPKGExportReport",
    "GPKGGeometryError",
    "GPKGValidationError",
    "GeoPackageExporter",
]
