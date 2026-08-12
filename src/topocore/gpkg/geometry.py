"""
topocore.gpkg.geometry
==========================

Converts a `FeatureGeometry` into a GeoPackage Binary (GPB) blob:
header (magic + version + flags + srs_id) + standard WKB, generated
by `shapely` (already a mandatory dependency) rather than hand-rolled.

Geometry family mapping
------------------------
``GeometryType.POINT``    -> shapely Point        -> GPKG "POINT"
``GeometryType.POLYLINE`` -> shapely LineString    -> GPKG "LINESTRING"
``GeometryType.POLYGON``  -> shapely Polygon       -> GPKG "POLYGON"
``GeometryType.MESH``     -> shapely MultiPolygon  -> GPKG "MULTIPOLYGON"
    (one Polygon Z per triangular face, same precedent as PR16's
    MESH -> 3DFACE-per-triangle; `feature_type` stays whatever it
    was -- this changes persistence representation, not semantics)

Z is always present -- TopoCore never flattens to 2D.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import struct
from enum import StrEnum

import numpy as np
import shapely.wkb
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from topocore.features.models import FeatureGeometry, GeometryType
from topocore.gpkg.exceptions import GPKGGeometryError

_GPB_MAGIC = b"GP"
_GPB_VERSION = 0x00
#: flags: bit0=1 (little-endian), bits1-3=000 (no envelope), bit4=0 (not empty)
_GPB_FLAGS_LITTLE_ENDIAN_NO_ENVELOPE = 0b00000001


class GeometryFamily(StrEnum):
    """The GPKG-table-naming/geometry-column vocabulary TopoCore emits. See gpkg_geometry_columns.geometry_type_name."""

    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"


#: geometry_type_name values as GeoPackage/OGC expect them (uppercase, WKT-style).
GEOMETRY_TYPE_NAME: dict[GeometryFamily, str] = {
    GeometryFamily.POINT: "POINT",
    GeometryFamily.LINE: "LINESTRING",
    GeometryFamily.POLYGON: "POLYGON",
    GeometryFamily.MULTIPOLYGON: "MULTIPOLYGON",
}

_FAMILY_BY_GEOMETRY_TYPE: dict[GeometryType, GeometryFamily] = {
    GeometryType.POINT: GeometryFamily.POINT,
    GeometryType.POLYLINE: GeometryFamily.LINE,
    GeometryType.POLYGON: GeometryFamily.POLYGON,
    GeometryType.MESH: GeometryFamily.MULTIPOLYGON,
}


def geometry_family(geometry_type: GeometryType) -> GeometryFamily:
    """
    Raises
    ------
    GPKGGeometryError
        If `geometry_type` isn't one of the 4 known `GeometryType`
        members -- can't happen with the current model (all 4 are
        mapped), but if a 5th member is ever added upstream without
        updating this table, this fails loudly and specifically
        instead of leaking a bare `KeyError` that `GeoPackageExporter`
        wouldn't recognize as a domain error (and so couldn't skip
        cleanly in non-strict mode).
    """
    try:
        return _FAMILY_BY_GEOMETRY_TYPE[geometry_type]
    except KeyError as exc:
        raise GPKGGeometryError(f"No GeoPackage mapping for geometry_type={geometry_type!r}.") from exc


def _to_shapely(
    geometry: FeatureGeometry,
) -> Point | LineString | Polygon | MultiPolygon:
    vertices = geometry.vertices

    if geometry.geometry_type == GeometryType.POINT:
        x, y, z = vertices[0]
        return Point(float(x), float(y), float(z))

    if geometry.geometry_type == GeometryType.POLYLINE:
        coords = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        return LineString(coords)

    if geometry.geometry_type == GeometryType.POLYGON:
        if not geometry.closed:
            raise GPKGGeometryError(
                "POLYGON geometry must be closed (FeatureGeometry.closed=True) to "
                "produce an unambiguous GeoPackage POLYGON ring."
            )
        coords = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        return Polygon(coords)

    if geometry.geometry_type == GeometryType.MESH:
        if geometry.faces is None or geometry.faces.size == 0:
            raise GPKGGeometryError("MESH geometry has no faces to triangulate into a MultiPolygon.")
        polygons: list[Polygon] = []
        for face in geometry.faces:
            tri = vertices[face]
            coords = [(float(v[0]), float(v[1]), float(v[2])) for v in tri]
            polygons.append(Polygon(coords))
        return MultiPolygon(polygons)

    raise GPKGGeometryError(f"No GeoPackage mapping for geometry_type={geometry.geometry_type!r}.")


def build_gpb(geometry: FeatureGeometry, srs_id: int) -> bytes:
    """
    Build a complete GeoPackage Binary (GPB) blob: header + WKB.

    Raises
    ------
    GPKGGeometryError
        If the geometry can't be converted (e.g. an unclosed POLYGON,
        or a MESH with no faces).
        # 1 = little-endian
    """
    shape = _to_shapely(geometry)
    wkb = shapely.wkb.dumps(shape, output_dimension=3, byte_order=1)

    header = struct.pack(
        "<ccBBi",
        _GPB_MAGIC[0:1],
        _GPB_MAGIC[1:2],
        _GPB_VERSION,
        _GPB_FLAGS_LITTLE_ENDIAN_NO_ENVELOPE,
        srs_id,
    )
    return header + wkb


def geometry_bounds_2d(geometry: FeatureGeometry) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) -- GeoPackage bounding boxes are 2D (XY only), even for Z geometries."""
    vertices = geometry.vertices
    return (
        float(np.min(vertices[:, 0])),
        float(np.min(vertices[:, 1])),
        float(np.max(vertices[:, 0])),
        float(np.max(vertices[:, 1])),
    )


__all__ = [
    "GEOMETRY_TYPE_NAME",
    "GeometryFamily",
    "build_gpb",
    "geometry_bounds_2d",
    "geometry_family",
]
