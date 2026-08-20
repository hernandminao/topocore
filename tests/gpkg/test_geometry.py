"""
Regression suite for topocore.gpkg.geometry -- PR19.

Verifies the GeoPackage Binary (GPB) header format directly (magic
bytes, version, flags, srs_id -- struct format "<ccBBi") against the
OGC GeoPackage spec, and confirms geometry_family() wraps a missing
mapping into GPKGGeometryError rather than a raw KeyError -- already
correctly implemented (unlike topocore.dxf's layer_for(), which
needed this exact fix in this same session). No bugs found.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from topocore.features.models import FeatureGeometry, GeometryType
from topocore.gpkg.exceptions import GPKGGeometryError
from topocore.gpkg.geometry import build_gpb, geometry_bounds_2d, geometry_family


def test_gpb_header_matches_ogc_spec_byte_layout() -> None:
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]]))
    blob = build_gpb(geom, srs_id=4326)

    magic1, magic2, version, flags, srs_id = struct.unpack_from("<ccBBi", blob, 0)
    assert magic1 + magic2 == b"GP"
    assert version == 0x00
    assert flags == 0b00000001  # little-endian, no envelope
    assert srs_id == 4326


def test_geometry_family_wraps_missing_mapping_cleanly() -> None:
    """
    Cannot construct a genuinely unmapped GeometryType (all 4 current
    members are mapped), so this directly probes the exception-
    wrapping behavior itself via a monkeypatched lookup table.
    """
    from topocore.gpkg import geometry as geometry_module

    original = dict(geometry_module._FAMILY_BY_GEOMETRY_TYPE)
    try:
        del geometry_module._FAMILY_BY_GEOMETRY_TYPE[GeometryType.POINT]
        with pytest.raises(GPKGGeometryError):
            geometry_family(GeometryType.POINT)
    finally:
        geometry_module._FAMILY_BY_GEOMETRY_TYPE.clear()
        geometry_module._FAMILY_BY_GEOMETRY_TYPE.update(original)


def test_geometry_bounds_2d_known_values() -> None:
    verts = np.array([[0.0, 5.0, 100.0], [10.0, -5.0, 200.0], [3.0, 8.0, 150.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYLINE, vertices=verts)

    bounds = geometry_bounds_2d(geom)
    assert bounds == (0.0, -5.0, 10.0, 8.0)


def test_mesh_with_degenerate_face_rejected() -> None:
    verts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # all identical -> zero-area
    faces = np.array([[0, 1, 2]])
    geom = FeatureGeometry(geometry_type=GeometryType.MESH, vertices=verts, faces=faces)

    with pytest.raises(GPKGGeometryError):
        build_gpb(geom, srs_id=4326)
