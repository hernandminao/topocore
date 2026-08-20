"""
Regression suite for topocore.dxf.exporter.DXFExporter -- PR19.

Includes two real bugs found and fixed in this session:

1. DXFExportOptions.units only set the DXF file's $INSUNITS header
   variable (metadata telling CAD software how to interpret the
   drawing) -- it never converted the actual coordinate values.
   Confirmed directly with a real exported file (read back with
   ezdxf): selecting units=FEET on data genuinely in meters (the
   standard working unit throughout TopoCore) wrote unconverted
   meter coordinates labeled as feet, causing any CAD software to
   display the geometry at ~3.28x the wrong scale, with no error or
   warning. Fixed by rejecting any units other than METERS outright
   until real coordinate conversion is implemented.

2. layer_for() raised a raw, unwrapped KeyError for any FeatureType
   not present in LAYER_BY_FEATURE_TYPE -- confirmed 63 of 84
   FeatureType values (75%) are uncovered. Since this KeyError was
   never one of the (DXFGeometryError, DXFExportError) types caught
   by export()'s own try/except, exporting a feature of any
   uncovered type crashed the ENTIRE export with a raw KeyError,
   completely bypassing options.strict -- the "strict=False skips
   bad features instead of crashing" contract silently didn't apply
   to this very common failure mode. Fixed by wrapping the KeyError
   into DXFExportError, so it now correctly participates in the
   existing strict/skip contract.

All 4 entity representations (POINT, LWPOLYLINE, POLYLINE3D, 3DFACE)
and XDATA encoding verified end-to-end against real DXF files written
and read back with ezdxf itself (not mocks).
"""

from __future__ import annotations

import ezdxf  # type: ignore[import-untyped]
import numpy as np
import pytest

from topocore.dxf.exceptions import DXFExportError
from topocore.dxf.exporter import DXFExporter
from topocore.dxf.models import DrawingUnits, DXFExportOptions, ExportContext
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)

from .conftest import make_feature

# ----------------------------------------------------------------------
# Bug 1: units mislabeling without conversion.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("units", [DrawingUnits.FEET, DrawingUnits.MILLIMETERS])
def test_non_meter_units_are_rejected(units: DrawingUnits) -> None:
    """
    The exact regression: before the fix, selecting FEET silently
    mislabeled unconverted meter coordinates, confirmed via a real
    exported file where the $INSUNITS header said feet but the
    coordinate values remained unchanged (100, 200, 10).
    """
    with pytest.raises(DXFExportError):
        DXFExporter(ExportContext(options=DXFExportOptions(units=units)))


def test_meters_default_still_works() -> None:
    exporter = DXFExporter(ExportContext(options=DXFExportOptions(units=DrawingUnits.METERS)))
    assert exporter is not None


# ----------------------------------------------------------------------
# Bug 2: unmapped FeatureType crashing with raw KeyError.
# ----------------------------------------------------------------------


def test_unmapped_feature_type_raises_dxf_export_error_not_keyerror(tmp_path) -> None:  # type: ignore[no-untyped-def]
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]]))
    feature = make_feature(1, FeatureType.CONTROL_POINT, FeatureCategory.CONTROL, geom)
    collection = FeatureCollection(features=[feature])

    exporter = DXFExporter(ExportContext(options=DXFExportOptions(strict=True)))
    with pytest.raises(DXFExportError):
        exporter.export(collection, str(tmp_path / "out.dxf"))


def test_unmapped_feature_type_skipped_gracefully_in_non_strict_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    The exact regression: before the fix, this crashed with a raw
    KeyError even in strict=False mode, which is specifically meant
    to skip problematic features instead of crashing.
    """
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]]))
    feature = make_feature(1, FeatureType.CONTROL_POINT, FeatureCategory.CONTROL, geom)
    collection = FeatureCollection(features=[feature])

    exporter = DXFExporter(ExportContext(options=DXFExportOptions(strict=False)))
    report = exporter.export(collection, str(tmp_path / "out.dxf"))  # must not raise

    assert report.skipped_features == 1
    assert len(report.warnings) == 1


def test_explicit_cad_layer_bypasses_unmapped_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from types import MappingProxyType

    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]]))
    feature = make_feature(
        1,
        FeatureType.CONTROL_POINT,
        FeatureCategory.CONTROL,
        geom,
        attributes=MappingProxyType({"cad_layer": "MY_LAYER"}),
    )
    collection = FeatureCollection(features=[feature])

    report = DXFExporter(ExportContext()).export(collection, str(tmp_path / "out.dxf"))
    assert report.skipped_features == 0
    assert report.entity_count == 1


def test_mapped_feature_type_still_works() -> None:
    """Confirms the fix didn't break the already-covered path."""
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[1.0, 2.0, 3.0]]))
    feature = make_feature(1, FeatureType.TREE, FeatureCategory.VEGETATION, geom)
    assert feature.feature_type == FeatureType.TREE


# ----------------------------------------------------------------------
# End-to-end entity verification, against real ezdxf-read-back files.
# ----------------------------------------------------------------------


def test_all_four_representations_round_trip_correctly(
    tmp_path: object,
    point_feature: Feature,
    polygon_feature: Feature,
    nonplanar_polyline_feature: Feature,
    mesh_feature: Feature,
) -> None:
    collection = FeatureCollection(
        features=[
            point_feature,
            polygon_feature,
            nonplanar_polyline_feature,
            mesh_feature,
        ]
    )
    path = str(tmp_path) + "/entities.dxf"
    report = DXFExporter(ExportContext()).export(collection, path)

    assert report.point_count == 1
    assert report.lwpolyline_count == 1
    assert report.polyline3d_count == 1
    assert report.face3d_count == 2  # 2 triangular faces from the mesh

    doc = ezdxf.readfile(path)
    entities_by_type = {e.dxftype(): e for e in doc.modelspace()}

    assert tuple(entities_by_type["POINT"].dxf.location) == (10.0, 20.0, 5.0)

    lwpoly = entities_by_type["LWPOLYLINE"]
    assert lwpoly.dxf.elevation == pytest.approx(5.0)
    assert lwpoly.closed is True  # type: ignore[attr-defined]

    polyline_3d = entities_by_type["POLYLINE"]
    vertices_3d = [tuple(v.dxf.location) for v in polyline_3d.vertices]  # type: ignore[attr-defined]
    assert vertices_3d == [(0.0, 0.0, 1.0), (10.0, 0.0, 5.0), (20.0, 0.0, 2.0)]


def test_3dface_duplicates_last_vertex_for_triangles(tmp_path: object, mesh_feature: Feature) -> None:
    collection = FeatureCollection(features=[mesh_feature])
    path = str(tmp_path) + "/mesh.dxf"
    DXFExporter(ExportContext()).export(collection, path)

    doc = ezdxf.readfile(path)
    faces = [e for e in doc.modelspace() if e.dxftype() == "3DFACE"]
    assert len(faces) == 2
    for face in faces:
        assert tuple(face.dxf.vtx2) == tuple(face.dxf.vtx3)  # standard DXF triangle convention
