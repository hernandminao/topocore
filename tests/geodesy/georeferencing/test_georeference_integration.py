"""
Integration tests for Workflow.georeference() chained with the REST
of the real Workflow pipeline -- read, classify_ground(),
detect_features(), build_features_from_survey(), transform_crs(),
export_dxf(), export_gpkg().

This suite does not add or modify any capability. It exists purely
to confirm georeference() does not break the global Workflow
contract: stage ordering, ArtifactStore state, StageMetrics/extra,
downstream stage compatibility, and error/rollback behavior --
exercised through real, already-existing pipelines, not new
mechanisms.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import ezdxf
import numpy as np
import pytest
from topocore.geodesy import CRS, CoordinateTransformer
from topocore.geodesy.georeferencing import (
    ControlPoint,
    GeoreferencingOptions,
    apply_helmert_3d,
)
from topocore.geodesy.georeferencing.exceptions import (
    UnderconstrainedGeoreferencingError,
)
from topocore.geodesy.helmert import HelmertParameters
from topocore.io.ascii.xyz.writer import XYZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.formats import SurveyFormat
from topocore.survey.models import SurveyPointSet
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import WorkflowStateError
from topocore.workflow.stages import WorkflowStage
from topocore.workflow.workflow import Workflow

_KNOWN_3D = HelmertParameters(tx=1152345.0, ty=923456.0, tz=245.0, rx=0.0, ry=0.0, rz=3.0, scale=20.0)
_TARGET_CRS = CRS.from_epsg(3116)


def _controls_from(sources: list[tuple[float, float, float]], parameters: HelmertParameters) -> list[ControlPoint]:
    targets = [apply_helmert_3d(point, parameters) for point in sources]
    return [ControlPoint(id=str(i), source=s, target=t) for i, (s, t) in enumerate(zip(sources, targets, strict=True))]


def _ground_like_point_cloud(n: int = 200, seed: int = 0) -> PointCloud:
    """A near-flat surface, matching the fixture already used elsewhere for classify_ground()/build_tin() tests."""
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0.0, 200.0, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0.0, 200.0, n)
    chunk[PointAttribute.Z][:] = 100.0 + rng.uniform(-0.5, 0.5, n)
    cloud.add_chunk(chunk)
    return cloud


_CONTROL_SOURCES = [
    (0.0, 0.0, 0.0),
    (200.0, 0.0, 0.0),
    (0.0, 200.0, 0.0),
    (100.0, 100.0, 50.0),
]


# ----------------------------------------------------------------------
# Full PointCloud pipeline: read -> georeference -> classify_ground ->
# detect_features -> transform_crs -> export_dxf/gpkg.
# ----------------------------------------------------------------------


def test_full_point_cloud_pipeline_with_georeference(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    # deliberately NO .prj -- this is exactly the local/unreferenced case georeference() exists for

    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)

    workflow = Workflow().read_point_cloud(path)
    assert workflow.artifact(ArtifactType.POINT_CLOUD).crs is None  # confirms the starting premise

    workflow.georeference(ArtifactType.POINT_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())
    assert workflow.artifact(ArtifactType.POINT_CLOUD).metadata.crs == "EPSG:3116"

    workflow.classify_ground()
    ground = workflow.artifact(ArtifactType.GROUND_CLOUD)
    assert ground.metadata.crs == "EPSG:3116"  # propagated through classify_ground(), unaffected by georeference()

    workflow.detect_features(strict=False)
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert features.crs == "EPSG:3116"  # propagated correctly from POINT_CLOUD

    # Now a genuine CRS-to-CRS transform on TOP of the georeferenced result -- confirms compatibility.
    transformer = CoordinateTransformer(CRS.from_epsg(3116), CRS.from_epsg(9377))
    workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)
    final_features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert final_features.crs == "EPSG:9377"

    # Stage ordering.
    stage_sequence = [s.stage for s in workflow.result.stages]
    assert stage_sequence == [
        WorkflowStage.READ_POINT_CLOUD,
        WorkflowStage.GEOREFERENCE,
        WorkflowStage.CLASSIFY_GROUND,
        WorkflowStage.DETECT_FEATURES,
        WorkflowStage.TRANSFORM_CRS,
    ]
    assert all(s.status.value == "success" for s in workflow.result.stages)


def test_full_point_cloud_pipeline_exports_correctly(tmp_path: Path) -> None:
    from topocore.features.models import (
        Feature,
        FeatureCategory,
        FeatureGeometry,
        FeatureType,
        GeometryType,
    )

    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)

    workflow = (
        Workflow()
        .read_point_cloud(path)
        .georeference(ArtifactType.POINT_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())
        .detect_features(strict=False)
    )

    # The synthetic point cloud above is pure random noise (built for
    # classify_ground()'s own flatness needs, not for detect_features()'s
    # own pattern matching) -- confirmed directly it genuinely detects
    # 0 features from it, which is correct, not a defect. A feature is
    # injected manually here so this test's own actual focus (does a
    # georeferenced FEATURE_COLLECTION export correctly) is exercised
    # against real, non-empty content, without depending on
    # detect_features()'s own detection quality on unrealistic data.
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    geometry = FeatureGeometry(
        geometry_type=GeometryType.POINT,
        vertices=np.array([[1152400.0, 923500.0, 245.0]]),
    )
    features.add(
        Feature(
            feature_id=1,
            category=FeatureCategory.TERRAIN,
            feature_type=FeatureType.CONTROL_POINT,
            geometry=geometry,
            attributes={"cad_layer": "CONTROL_POINTS"},  # CONTROL_POINT has no default DXF layer mapping
        )
    )
    workflow._store.set(ArtifactType.FEATURE_COLLECTION, features)

    dxf_path = tmp_path / "output.dxf"
    gpkg_path = tmp_path / "output.gpkg"
    workflow.export_dxf(dxf_path)
    workflow.export_gpkg(gpkg_path)

    doc = ezdxf.readfile(dxf_path)
    assert dict(doc.header.custom_vars).get("TopoCore CRS") == "EPSG:3116"

    con = sqlite3.connect(gpkg_path)
    try:
        srs_ids = {row[0] for row in con.execute("SELECT srs_id FROM gpkg_contents")}
    finally:
        con.close()
    assert srs_ids == {3116}


# ----------------------------------------------------------------------
# Full SurveyPointSet pipeline: read_survey -> georeference ->
# build_features_from_survey -> resolve_sides -> export.
# ----------------------------------------------------------------------


def test_full_survey_pipeline_with_georeference(tmp_path: Path) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,0.0,0.0,100.0,EJE\n2,0.0,200.0,100.0,EJE\n3,-5.0,0.0,100.0,BORDE\n4,-5.0,200.0,100.0,BORDE\n")
    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    assert workflow.artifact(ArtifactType.SURVEY_POINT_SET).crs is None

    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())
    assert workflow.artifact(ArtifactType.SURVEY_POINT_SET).crs is _TARGET_CRS

    workflow.build_features_from_survey()
    features_before_resolve = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert features_before_resolve.crs == "EPSG:3116"

    workflow.resolve_sides()
    features_after_resolve = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert features_after_resolve.crs == "EPSG:3116"  # preserved through resolve_sides(), matching its own contract

    edge = next(f for f in features_after_resolve if f.feature_type.value == "pavement_edge")
    assert edge.attributes["side"] in ("left", "right")

    gpkg_path = tmp_path / "survey_output.gpkg"
    workflow.export_gpkg(gpkg_path)
    con = sqlite3.connect(gpkg_path)
    try:
        srs_ids = {row[0] for row in con.execute("SELECT srs_id FROM gpkg_contents")}
    finally:
        con.close()
    assert srs_ids == {3116}


# ----------------------------------------------------------------------
# GROUND_CLOUD georeferenced directly (a separate artifact from
# POINT_CLOUD, sharing the same PointCloud type).
# ----------------------------------------------------------------------


def test_ground_cloud_georeferenced_directly_after_classification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)

    workflow = Workflow().read_point_cloud(path).classify_ground()
    assert workflow.artifact(ArtifactType.GROUND_CLOUD).crs is None  # POINT_CLOUD had no CRS, so neither does this

    workflow.georeference(ArtifactType.GROUND_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())
    ground = workflow.artifact(ArtifactType.GROUND_CLOUD)
    assert ground.metadata.crs == "EPSG:3116"

    # POINT_CLOUD itself is untouched -- georeference() on GROUND_CLOUD never touches the sibling artifact.
    point_cloud = workflow.artifact(ArtifactType.POINT_CLOUD)
    assert point_cloud.metadata.crs is None


# ----------------------------------------------------------------------
# Error / rollback: a rejected georeference() call leaves the
# Workflow exactly as it was.
# ----------------------------------------------------------------------


def test_rejected_georeference_leaves_workflow_state_untouched(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())

    workflow = Workflow().read_point_cloud(path)
    stages_before = list(workflow.result.stages)
    version_before = workflow._store.version_of(ArtifactType.POINT_CLOUD)

    single_control = [ControlPoint(id="1", source=(0.0, 0.0, 0.0), target=(10.0, 20.0, 5.0))]

    with pytest.raises(UnderconstrainedGeoreferencingError):
        workflow.georeference(
            ArtifactType.POINT_CLOUD,
            single_control,
            _TARGET_CRS,
            GeoreferencingOptions(),
        )

    assert list(workflow.result.stages) == stages_before  # no new stage recorded
    assert workflow._store.version_of(ArtifactType.POINT_CLOUD) == version_before  # no new version written
    assert workflow.artifact(ArtifactType.POINT_CLOUD).metadata.crs is None  # artifact itself unchanged

    # The workflow remains genuinely usable afterward -- confirms no partial/corrupted internal state.
    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)
    workflow.georeference(ArtifactType.POINT_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())
    assert workflow.artifact(ArtifactType.POINT_CLOUD).metadata.crs == "EPSG:3116"


def test_double_georeference_rejection_does_not_corrupt_downstream_pipeline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)

    workflow = Workflow().read_point_cloud(path)
    workflow.georeference(ArtifactType.POINT_CLOUD, controls, _TARGET_CRS, GeoreferencingOptions())

    with pytest.raises(WorkflowStateError):
        workflow.georeference(
            ArtifactType.POINT_CLOUD,
            controls,
            CRS.from_epsg(9377),
            GeoreferencingOptions(),
        )

    # Confirms the pipeline remains usable and correctly referenced after the rejected second attempt.
    workflow.classify_ground().detect_features(strict=False)
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert features.crs == "EPSG:3116"


# ----------------------------------------------------------------------
# StageMetrics/extra sanity under repeated inspection.
# ----------------------------------------------------------------------


def test_workflow_result_stages_are_side_effect_free_to_read_repeatedly() -> None:
    workflow = Workflow()
    workflow._store.set(ArtifactType.SURVEY_POINT_SET, SurveyPointSet(points=()))
    controls = _controls_from(_CONTROL_SOURCES, _KNOWN_3D)
    workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, _TARGET_CRS, GeoreferencingOptions())

    first_read = list(workflow.result.stages)
    second_read = list(workflow.result.stages)
    assert first_read == second_read
    assert len(first_read) == 1
    assert first_read[0].stage is WorkflowStage.GEOREFERENCE
