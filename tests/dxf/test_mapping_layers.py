"""
Regression suite for topocore.dxf.mapping.GeometryMapper and
.layers -- PR19. Verified representation-selection logic and index-
contour boundary detection with known cases. No bugs found (the
layer coverage gap itself is covered separately in test_exporter.py,
since the fix lives in exporter.py's error-wrapping, not here).

test_index_contour_rejects_nonpositive_interval updated during this
project's own PR22 documentation audit: is_index_contour() now
raises DXFExportError (not a bare ValueError) for interval<=0 --
confirmed, with a real generated file, that a bare ValueError here
escaped DXFExporter.export()'s own `except (DXFGeometryError,
DXFExportError)`, silently breaking strict=False's own "skip this
feature, keep going" contract for a CONTOUR feature with an invalid
interval. See topocore.dxf.layers.is_index_contour's own docstring
and this project's own 20-dxf audit documentation for the complete
before/after evidence. The same fix also newly rejects
elevation/base/interval being non-finite or the wrong type -- none
of which were rejected at all before that fix, and none of which
this file's own tests cover (see
test_layers_input_validation_regression.py for those cases).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.dxf.exceptions import DXFExportError, DXFGeometryError
from topocore.dxf.layers import (
    CONTOUR_LAYER_MAJOR,
    CONTOUR_LAYER_MINOR,
    contour_layer_name,
    is_index_contour,
)
from topocore.dxf.mapping import DXFRepresentation, GeometryMapper
from topocore.dxf.models import NonPlanarPolygonMode
from topocore.dxf.tolerance import DXFTolerance
from topocore.features.models import FeatureGeometry, GeometryType


@pytest.fixture
def mapper() -> GeometryMapper:
    return GeometryMapper(DXFTolerance(), non_planar_polygon_mode=NonPlanarPolygonMode.POLYLINE3D)


def test_planar_polygon_maps_to_lwpolyline(mapper: GeometryMapper) -> None:
    verts = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [1.0, 1.0, 5.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYGON, vertices=verts, closed=True)
    decision = mapper.decide(geom)
    assert decision.representation == DXFRepresentation.LWPOLYLINE
    assert decision.elevation == pytest.approx(5.0)


def test_nonplanar_polygon_maps_to_polyline3d_by_default(
    mapper: GeometryMapper,
) -> None:
    verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 5.0], [1.0, 1.0, 2.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYGON, vertices=verts, closed=True)
    decision = mapper.decide(geom)
    assert decision.representation == DXFRepresentation.POLYLINE3D


def test_nonplanar_polygon_raises_in_error_mode() -> None:
    strict_mapper = GeometryMapper(DXFTolerance(), non_planar_polygon_mode=NonPlanarPolygonMode.ERROR)
    verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 5.0], [1.0, 1.0, 2.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYGON, vertices=verts, closed=True)
    with pytest.raises(DXFGeometryError):
        strict_mapper.decide(geom)


def test_mesh_maps_to_face3d(mapper: GeometryMapper) -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    geom = FeatureGeometry(geometry_type=GeometryType.MESH, vertices=verts, faces=faces)
    decision = mapper.decide(geom)
    assert decision.representation == DXFRepresentation.FACE3D


# ----------------------------------------------------------------------
# Contour index-detection boundary cases.
# ----------------------------------------------------------------------


def test_index_contour_at_exact_multiple() -> None:
    assert is_index_contour(15.0, base=0.0, interval=1.0, every=5) is True


def test_index_contour_off_multiple() -> None:
    assert is_index_contour(13.0, base=0.0, interval=1.0, every=5) is False


def test_contour_layer_name_major_vs_minor() -> None:
    assert contour_layer_name(elevation=15.0, base=0.0, interval=1.0, every=5) == CONTOUR_LAYER_MAJOR
    assert contour_layer_name(elevation=13.0, base=0.0, interval=1.0, every=5) == CONTOUR_LAYER_MINOR


def test_index_contour_rejects_nonpositive_interval() -> None:
    with pytest.raises(DXFExportError):
        is_index_contour(1.0, base=0.0, interval=0.0, every=5)
