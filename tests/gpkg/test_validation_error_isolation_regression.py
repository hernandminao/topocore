"""
Regression test for a real defect found and fixed during this
project's own GeoPackage documentation audit:
`GPKGValidator.validate()`'s own checks used to be a strict subset
of what `gpkg.geometry._to_shapely()` (called later, inside the
export transaction) actually requires -- missing a finite-coordinate
check and a degenerate-MESH-triangle check.

A `Feature` that only `_to_shapely()` rejected used to reach
`GeoPackageExporter._write()` inside the write transaction, where any
`GPKGGeometryError` is fatal to the WHOLE export regardless of
`GPKGExportOptions.strict` -- silently violating `strict=False`'s own
documented contract ("that feature is skipped and recorded in the
report instead"): a valid feature in the same collection as one with
a degenerate MESH face used to produce no `.gpkg` file at all, not a
partial one with just the valid feature written.

Fixed by extending `GPKGValidator.validate()` to cover the same 2
conditions `_to_shapely()` already checks (`GPKG006` for non-finite
coordinates, `GPKG007` for a degenerate MESH face), so both are
caught in the exporter's own earlier, per-feature, `strict`-aware
validation loop -- before `_write()` is ever reached.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.gpkg.config import GPKGExportOptions
from topocore.gpkg.exceptions import GPKGValidationError
from topocore.gpkg.exporter import GeoPackageExporter
from topocore.gpkg.validation import GPKGValidator


def _valid_mesh_feature(feature_id: int = 1) -> Feature:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    geometry = FeatureGeometry(GeometryType.MESH, vertices, faces=np.array([[0, 1, 2]], dtype=np.int64))
    return Feature(
        feature_id=feature_id, category=FeatureCategory.BUILDING, feature_type=FeatureType.ROOF, geometry=geometry
    )


def _degenerate_mesh_feature(feature_id: int = 2) -> Feature:
    # 3 collinear vertices -- a real (zero cross-product) triangle,
    # not merely small; `models.py`'s own construction-time
    # validation only checks vertex shape/finiteness, not that a
    # MESH face forms a genuine, non-degenerate triangle.
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    geometry = FeatureGeometry(GeometryType.MESH, vertices, faces=np.array([[0, 1, 2]], dtype=np.int64))
    return Feature(
        feature_id=feature_id, category=FeatureCategory.BUILDING, feature_type=FeatureType.ROOF, geometry=geometry
    )


def test_validator_detects_degenerate_mesh_face() -> None:
    issues = GPKGValidator().validate(_degenerate_mesh_feature())

    assert [issue.code for issue in issues] == ["GPKG007"]


def test_validator_detects_non_finite_vertices() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, float("nan"), 0.0]])
    geometry = FeatureGeometry.__new__(FeatureGeometry)
    object.__setattr__(geometry, "geometry_type", GeometryType.POINT)
    object.__setattr__(geometry, "vertices", vertices)
    object.__setattr__(geometry, "closed", False)
    object.__setattr__(geometry, "faces", None)

    feature = Feature.__new__(Feature)
    object.__setattr__(feature, "feature_id", 1)
    object.__setattr__(feature, "geometry", geometry)

    issues = GPKGValidator().validate(feature)

    assert "GPKG006" in [issue.code for issue in issues]


def test_validator_accepts_genuine_valid_mesh() -> None:
    """Confirms the fix did not introduce a false positive."""
    issues = GPKGValidator().validate(_valid_mesh_feature())

    assert issues == ()


def test_strict_false_isolates_invalid_feature_and_writes_the_valid_one() -> None:
    """
    The exact real defect this fix addresses: before the fix, this
    exact scenario raised GPKGExportError and produced NO file at
    all, even though one feature in the collection was genuinely
    valid.
    """
    collection = FeatureCollection()
    collection.add(_valid_mesh_feature(feature_id=1))
    collection.add(_degenerate_mesh_feature(feature_id=2))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.gpkg"

        report = GeoPackageExporter(GPKGExportOptions(epsg=4326, strict=False)).export(collection, path)

        assert path.exists()
        assert report.written_count == 1
        assert report.skipped_count == 1
        assert "GPKG007" in report.warnings[0] or "degenerate" in report.warnings[0]

        conn = sqlite3.connect(path)
        try:
            count = conn.execute('SELECT count(*) FROM "building_multipolygon"').fetchone()[0]
            assert count == 1
        finally:
            conn.close()


def test_strict_true_still_aborts_the_whole_export() -> None:
    """Confirms strict=True's own contract is unchanged by this fix."""
    collection = FeatureCollection()
    collection.add(_valid_mesh_feature(feature_id=1))
    collection.add(_degenerate_mesh_feature(feature_id=2))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.gpkg"

        with pytest.raises(GPKGValidationError, match="GPKG007"):
            GeoPackageExporter(GPKGExportOptions(epsg=4326, strict=True)).export(collection, path)

        assert not path.exists()


def test_valid_collection_unaffected_by_the_fix() -> None:
    """Confirms a fully valid collection still exports exactly as before."""
    collection = FeatureCollection()
    collection.add(_valid_mesh_feature(feature_id=1))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.gpkg"

        report = GeoPackageExporter(GPKGExportOptions(epsg=4326, strict=True)).export(collection, path)

        assert path.exists()
        assert report.written_count == 1
        assert report.skipped_count == 0
