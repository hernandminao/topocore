"""
Tests for FeatureCollection.crs and its consumers: detect_features(),
build_features_from_survey(), transform_feature_collection()
(horizontal), transform_feature_collection_vertical(), and
SideResolver.resolve().

Confirmed during this capability's own audit: FeatureCollection.crs
is typed `str | None` (matching PointCloud.crs), not `CRS | None`
(matching SurveyPointSet.crs) -- deliberately, since
FeatureCollection can be built from either kind of source, and
converting a CRS object to a string is always safe while the reverse
is not (a bare name like "unknown" cannot be reconstructed).

A real bug was found and fixed while wiring this: Workflow.artifact()'s
own defensive-copy branch for FEATURE_COLLECTION (written before
`.crs` existed) reconstructed `FeatureCollection(features=list(...))`
without carrying `.crs` through, silently resetting it to None on
every read via the public accessor even though the stored object
itself was correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)
from topocore.features.side import SideResolver
from topocore.geodesy import CRS, CoordinateTransformer
from topocore.geodesy.transform import transform_feature_collection
from topocore.geodesy.vertical import GeoidGrid, VerticalTransformer
from topocore.geodesy.vertical.transform import transform_feature_collection_vertical
from topocore.geodesy.vertical_datum import VerticalDatum
from topocore.io.ascii.xyz.writer import XYZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.formats import SurveyFormat
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.workflow import Workflow


def _make_feature(x: float = 1.0, y: float = 1.0, z: float = 100.0) -> Feature:
    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[x, y, z]]))
    return Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )


# ----------------------------------------------------------------------
# 1. FeatureCollection.crs itself.
# ----------------------------------------------------------------------


def test_crs_defaults_to_none() -> None:
    assert FeatureCollection().crs is None


def test_crs_can_be_set_directly() -> None:
    """FeatureCollection is mutable -- crs is a plain settable field, unlike SurveyPointSet's frozen one."""
    collection = FeatureCollection()
    collection.crs = "EPSG:32618"
    assert collection.crs == "EPSG:32618"


def test_crs_can_be_set_at_construction() -> None:
    collection = FeatureCollection(crs="EPSG:4326")
    assert collection.crs == "EPSG:4326"


# ----------------------------------------------------------------------
# 2/3. Workflow construction sites.
# ----------------------------------------------------------------------


def _ground_like_point_cloud(n: int = 50, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(-74.1, -74.0, n)
    chunk[PointAttribute.Y][:] = rng.uniform(4.6, 4.7, n)
    chunk[PointAttribute.Z][:] = 2500.0 + rng.uniform(-0.5, 0.5, n)
    cloud.add_chunk(chunk)
    return cloud


def test_detect_features_sets_crs_from_point_cloud(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    (tmp_path / "cloud.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_point_cloud(path).detect_features(strict=False)
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)

    assert features.crs == "EPSG:4326"


def test_detect_features_without_prj_leaves_crs_none(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())

    workflow = Workflow().read_point_cloud(path).detect_features(strict=False)
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)

    assert features.crs is None


def test_build_features_from_survey_sets_crs_from_survey_point_set(
    tmp_path: Path,
) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)

    assert features.crs == "EPSG:3116"


def test_this_confirms_the_workflow_artifact_defensive_copy_fix(tmp_path: Path) -> None:
    """
    The regression itself: Workflow.artifact()'s own defensive-copy
    branch for FEATURE_COLLECTION used to drop .crs back to None on
    every read, even though the stored object was correct. Reading
    the artifact TWICE confirms the fix holds consistently, not just
    on the first read.
    """
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

    first_read = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    second_read = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert first_read.crs == "EPSG:3116"
    assert second_read.crs == "EPSG:3116"
    assert first_read is not second_read  # confirms it's still a genuine defensive copy, not a cached reference


# ----------------------------------------------------------------------
# 4. Horizontal transform updates crs to target.
# ----------------------------------------------------------------------


def test_transform_feature_collection_sets_crs_to_target() -> None:
    collection = FeatureCollection(crs="EPSG:4326")
    collection.add(_make_feature(x=-74.0, y=4.7))
    transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))

    result = transform_feature_collection(collection, transformer)

    assert result.crs == "EPSG:32618"
    assert collection.crs == "EPSG:4326"  # input untouched


# ----------------------------------------------------------------------
# 5. Vertical transform preserves crs.
# ----------------------------------------------------------------------


def test_transform_feature_collection_vertical_preserves_crs(
    synthetic_geoid_path: Path,
) -> None:
    collection = FeatureCollection(crs="EPSG:32618")
    collection.add(_make_feature(x=1.0, y=1.0, z=100.0))

    transformer = VerticalTransformer(
        source_datum=VerticalDatum(name="ellipsoidal"),
        target_datum=VerticalDatum(name="orthometric", geoid_model="SYNTHETIC"),
        geoid=GeoidGrid.from_geotiff(synthetic_geoid_path),
    )

    result = transform_feature_collection_vertical(collection, transformer)

    assert result.crs == "EPSG:32618"
    assert result.features[0].geometry.vertices[0][2] == 50.0  # confirms Z genuinely shifted


def test_transform_feature_collection_vertical_preserves_none_crs(
    synthetic_geoid_path: Path,
) -> None:
    collection = FeatureCollection()  # crs=None
    collection.add(_make_feature(x=1.0, y=1.0, z=100.0))

    transformer = VerticalTransformer(
        source_datum=VerticalDatum(name="ellipsoidal"),
        target_datum=VerticalDatum(name="orthometric", geoid_model="SYNTHETIC"),
        geoid=GeoidGrid.from_geotiff(synthetic_geoid_path),
    )

    result = transform_feature_collection_vertical(collection, transformer)
    assert result.crs is None


# ----------------------------------------------------------------------
# 6. SideResolver preserves crs.
# ----------------------------------------------------------------------


def test_side_resolver_preserves_crs() -> None:
    geometry = FeatureGeometry(
        geometry_type=GeometryType.POLYLINE,
        vertices=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    )
    centerline = Feature(
        feature_id=1,
        category=FeatureCategory.INFRASTRUCTURE,
        feature_type=FeatureType.CENTERLINE,
        geometry=geometry,
    )
    collection = FeatureCollection(crs="EPSG:4326")
    collection.add(centerline)

    result = SideResolver().resolve(collection)

    assert result.crs == "EPSG:4326"


def test_side_resolver_preserves_none_crs() -> None:
    geometry = FeatureGeometry(
        geometry_type=GeometryType.POLYLINE,
        vertices=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    )
    centerline = Feature(
        feature_id=1,
        category=FeatureCategory.INFRASTRUCTURE,
        feature_type=FeatureType.CENTERLINE,
        geometry=geometry,
    )
    collection = FeatureCollection()  # crs=None
    collection.add(centerline)

    result = SideResolver().resolve(collection)
    assert result.crs is None


# ----------------------------------------------------------------------
# Full pipeline, end to end.
# ----------------------------------------------------------------------


def test_full_pipeline_crs_propagation(tmp_path: Path, synthetic_geoid_path: Path) -> None:
    path = tmp_path / "survey.csv"
    path.write_text(
        "1,0.0,0.0,100.0,EJE\n2,0.0,10.0,100.0,EJE\n3,-5.0,0.0,100.0,BORDE\n4,-5.0,10.0,100.0,BORDE\n",
        encoding="utf-8",
    )
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = (
        Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey().resolve_sides()
    )
    assert workflow.artifact(ArtifactType.FEATURE_COLLECTION).crs == "EPSG:4326"

    transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))
    workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)

    final = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert final.crs == "EPSG:32618"
    edge = next(f for f in final if f.feature_type.value == "pavement_edge")
    assert edge.attributes["side"] == "left"
