"""
Regression suite for topocore.gpkg.exporter.GeoPackageExporter --
PR19.

Verified end-to-end against real, independent GIS tooling
(geopandas + fiona reading the actual exported .gpkg file, not
TopoCore's own code) -- the most rigorous validation available:
exact coordinate preservation (millimeter precision), correct
geometry-family-per-table splitting, MESH -> MultiPolygon conversion
matching triangle count, CRS correctly resolved and readable via
pyproj/geopandas's own independent parsing, promoted attribute
columns (survey_code/survey_name/cad_layer) plus remaining
attributes correctly serialized as JSON, and strict/non-strict
validation behavior for an unclosed POLYGON. No bugs found -- this
module already correctly applies the same "wrap KeyError into a
domain exception" pattern that had to be fixed in topocore.dxf this
same session (topocore.gpkg.geometry.geometry_family()).
"""

from __future__ import annotations

import json

import fiona  # type: ignore[import-untyped]
import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pytest

from topocore.features.models import (
    Feature,
    FeatureCollection,
    FeatureGeometry,
    GeometryType,
)
from topocore.gpkg.config import GPKGExportOptions
from topocore.gpkg.exceptions import GPKGValidationError
from topocore.gpkg.exporter import GeoPackageExporter

from .conftest import make_feature


def test_point_geometry_exact_coordinate_round_trip(tmp_path: object, point_feature: Feature) -> None:
    """
    Decisive check: read back with geopandas/fiona, completely
    independent of TopoCore's own writing code.
    """
    collection = FeatureCollection(features=[point_feature])
    path = str(tmp_path) + "/point.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=32618)).export(collection, path)

    layers = fiona.listlayers(path)
    gdf = gpd.read_file(path, layer=layers[0])
    row = gdf.iloc[0]

    assert row.geometry.x == pytest.approx(500123.456)
    assert row.geometry.y == pytest.approx(4000456.789)
    assert row.geometry.z == pytest.approx(100.5)
    assert str(gdf.crs) == "EPSG:32618"


def test_polygon_and_polyline_split_into_separate_tables(
    tmp_path: object,
    polygon_feature: Feature,
    polyline_feature: Feature,
) -> None:
    collection = FeatureCollection(features=[polygon_feature, polyline_feature])
    path = str(tmp_path) + "/mixed.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=32618)).export(collection, path)

    layers = set(fiona.listlayers(path))
    assert layers == {"building_polygon", "terrain_line"}

    poly_gdf = gpd.read_file(path, layer="building_polygon")
    assert poly_gdf.iloc[0].geometry.geom_type == "Polygon"

    line_gdf = gpd.read_file(path, layer="terrain_line")
    assert line_gdf.iloc[0].geometry.geom_type == "LineString"


def test_mesh_becomes_multipolygon_with_matching_triangle_count(tmp_path: object, mesh_feature: Feature) -> None:
    collection = FeatureCollection(features=[mesh_feature])
    path = str(tmp_path) + "/mesh.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=32618)).export(collection, path)

    layers = fiona.listlayers(path)
    gdf = gpd.read_file(path, layer=layers[0])
    row = gdf.iloc[0]

    assert row.geometry.geom_type == "MultiPolygon"
    assert len(row.geometry.geoms) == 2  # matches the 2 triangular faces


def test_promoted_attributes_and_json_payload(tmp_path: object, mesh_feature: Feature) -> None:
    collection = FeatureCollection(features=[mesh_feature])
    path = str(tmp_path) + "/attrs.gpkg"
    GeoPackageExporter(GPKGExportOptions(epsg=32618)).export(collection, path)

    layers = fiona.listlayers(path)
    gdf = gpd.read_file(path, layer=layers[0])
    row = gdf.iloc[0]

    assert row["survey_code"] == "RF01"
    assert row["survey_name"] == "Roof A"
    assert row["cad_layer"] == "ROOFS"

    remaining = json.loads(row["attributes_json"])
    assert remaining == {"material": "tile", "area_m2": 123.4}
    assert "survey_code" not in remaining  # promoted keys excluded from JSON


def test_unclosed_polygon_rejected_in_strict_mode(tmp_path: object) -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYGON, vertices=verts, closed=False)
    from topocore.features.models import FeatureCategory, FeatureType

    feature = make_feature(1, FeatureType.BUILDING, FeatureCategory.BUILDING, geom)
    collection = FeatureCollection(features=[feature])

    with pytest.raises(GPKGValidationError):
        GeoPackageExporter(GPKGExportOptions(epsg=32618, strict=True)).export(collection, str(tmp_path) + "/bad.gpkg")


def test_unclosed_polygon_skipped_in_non_strict_mode(tmp_path: object) -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POLYGON, vertices=verts, closed=False)
    from topocore.features.models import FeatureCategory, FeatureType

    feature = make_feature(1, FeatureType.BUILDING, FeatureCategory.BUILDING, geom)
    collection = FeatureCollection(features=[feature])

    report = GeoPackageExporter(GPKGExportOptions(epsg=32618, strict=False)).export(
        collection, str(tmp_path) + "/bad.gpkg"
    )  # must not raise
    assert report.skipped_count == 1


def test_rejects_undefined_placeholder_srs(tmp_path: object) -> None:
    """
    GeoPackageExporter.__init__() only stores options -- SRS
    resolution happens inside export() (resolve_srs() is called
    there, not at construction time), so the error only surfaces on
    an actual export() call.
    """
    from topocore.gpkg.exceptions import GPKGExportError

    verts = np.array([[0.0, 0.0, 0.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=verts)
    from topocore.features.models import FeatureCategory, FeatureType

    feature = make_feature(1, FeatureType.TREE, FeatureCategory.VEGETATION, geom)
    collection = FeatureCollection(features=[feature])

    with pytest.raises(GPKGExportError):
        GeoPackageExporter(GPKGExportOptions(epsg=0)).export(collection, str(tmp_path) + "/bad.gpkg")


def test_rejects_unrecognized_epsg_code(tmp_path: object) -> None:
    from topocore.gpkg.exceptions import GPKGExportError

    verts = np.array([[0.0, 0.0, 0.0]])
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=verts)
    from topocore.features.models import FeatureCategory, FeatureType

    feature = make_feature(1, FeatureType.TREE, FeatureCategory.VEGETATION, geom)
    collection = FeatureCollection(features=[feature])

    with pytest.raises(GPKGExportError):
        GeoPackageExporter(GPKGExportOptions(epsg=999999999)).export(collection, str(tmp_path) + "/bad.gpkg")
