"""
Regression tests for GeoPackageExporter promoting `elevation` and
`survey_id` to native columns (in addition to the pre-existing
survey_code/survey_name/cad_layer), per the hybrid model decided for
TopoCore: common, filterable, GIS-relevant attributes are native
columns; everything else stays in attributes_json.
"""
from __future__ import annotations

import sqlite3

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
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions
from topocore.gpkg.exceptions import GPKGExportError


def _contour_feature(feature_id: int, elevation: object) -> Feature:
    return Feature(
        feature_id=feature_id,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTOUR,
        geometry=FeatureGeometry(
            geometry_type=GeometryType.POLYLINE,
            vertices=np.array([[0.0, 0.0, 205.0], [10.0, 10.0, 205.0]]),
        ),
        attributes={"elevation": elevation},
    )


def test_elevation_and_survey_id_become_native_columns(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(Feature(
        feature_id=1, category=FeatureCategory.VEGETATION, feature_type=FeatureType.TREE,
        geometry=FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[0.0, 0.0, 10.0]])),
        attributes={"survey_id": "ARBOL_42"},
    ))
    collection.add(_contour_feature(2, 205.0))

    output = tmp_path / "test.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=9377)).export(collection, str(output))

    con = sqlite3.connect(str(output))
    cur = con.cursor()

    cur.execute("PRAGMA table_info(vegetation_point)")
    columns = [row[1] for row in cur.fetchall()]
    assert "elevation" in columns
    assert "survey_id" in columns

    cur.execute("SELECT survey_id, elevation, attributes_json FROM vegetation_point")
    survey_id, elevation, attributes_json = cur.fetchone()
    assert survey_id == "ARBOL_42"
    assert elevation is None
    assert attributes_json is None  # nothing left over once survey_id is promoted

    cur.execute("SELECT survey_id, elevation, attributes_json FROM terrain_line")
    survey_id, elevation, attributes_json = cur.fetchone()
    assert survey_id is None
    assert elevation == pytest.approx(205.0)
    assert attributes_json is None


def test_elevation_is_directly_queryable_with_a_real_predicate(tmp_path) -> None:
    """The actual point of promoting elevation: `elevation = 205.0`
    should work directly in SQL/QGIS, without json_extract()."""
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature(1, 205.0))
    collection.add(_contour_feature(2, 206.0))

    output = tmp_path / "test.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=9377)).export(collection, str(output))

    con = sqlite3.connect(str(output))
    cur = con.cursor()
    cur.execute("SELECT feature_id FROM terrain_line WHERE elevation = 205.0")
    assert cur.fetchall() == [(1,)]


def test_an_integer_elevation_is_accepted_and_stored_as_real(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature(1, 205))  # int, not float

    output = tmp_path / "test.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=9377)).export(collection, str(output))

    con = sqlite3.connect(str(output))
    cur = con.cursor()
    cur.execute("SELECT elevation FROM terrain_line")
    assert cur.fetchone()[0] == pytest.approx(205.0)


def test_a_non_numeric_elevation_is_rejected(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature(1, "not-a-number"))

    output = tmp_path / "test.gpkg"
    with pytest.raises(GPKGExportError, match="non-numeric"):
        GeoPackageExporter(GPKGExportOptions(epsg=9377)).export(collection, str(output))


def test_a_boolean_elevation_is_rejected() -> None:
    """bool is a subclass of int in Python -- confirmed this must be
    explicitly excluded, or True/False would silently pass as 1.0/0.0."""
    feature = _contour_feature(1, True)
    from topocore.gpkg.exporter import GeoPackageExporter as Exporter

    with pytest.raises(GPKGExportError, match="non-numeric"):
        Exporter._split_attributes(feature)


def test_other_attributes_still_go_to_attributes_json_alongside_native_ones(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(Feature(
        feature_id=1, category=FeatureCategory.TERRAIN, feature_type=FeatureType.CONTOUR,
        geometry=FeatureGeometry(
            geometry_type=GeometryType.POLYLINE,
            vertices=np.array([[0.0, 0.0, 205.0], [10.0, 10.0, 205.0]]),
        ),
        attributes={"elevation": 205.0, "interval": 0.5, "method": "TIN"},
    ))

    output = tmp_path / "test.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=9377)).export(collection, str(output))

    con = sqlite3.connect(str(output))
    cur = con.cursor()
    cur.execute("SELECT elevation, attributes_json FROM terrain_line")
    elevation, attributes_json = cur.fetchone()
    assert elevation == pytest.approx(205.0)
    assert "interval" in attributes_json
    assert "method" in attributes_json
    assert "elevation" not in attributes_json  # promoted, not duplicated
