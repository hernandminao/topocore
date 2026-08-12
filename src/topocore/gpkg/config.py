"""
topocore.gpkg.config
========================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

#: GeoPackage 1.3.0, encoded as SQLite PRAGMA user_version
GPKG_USER_VERSION = 10300

#: "GPKG" magic, little-endian, as SQLite PRAGMA application_id
GPKG_APPLICATION_ID = 0x47504B47


@dataclass(frozen=True, slots=True)
class GPKGExportOptions:
    """
    Parameters
    ----------
    epsg
        Coordinate reference system, as an EPSG code. Mandatory --
        there is no default. A GeoPackage without a real CRS is
        ambiguous in QGIS/ArcGIS and hostile to interoperability;
        `GeoPackageExporter` raises `GPKGExportError` rather than
        falling back to a placeholder like srs_id=0.
    strict
        If True (default), any feature that fails geometry/validation
        checks aborts the whole export. If False, that feature is
        skipped and recorded in the report instead.
    """

    epsg: int
    strict: bool = True


__all__ = ["GPKG_APPLICATION_ID", "GPKG_USER_VERSION", "GPKGExportOptions"]
