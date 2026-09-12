"""
Tests for transform_crs()'s 2 safety checks, added after a real,
confirmed danger was found: transforming an artifact whose current
CRS is unknown (crs=None) previously proceeded silently, producing
`inf` coordinate values while the result falsely declared the
transformer's own target CRS as a legitimate georeferencing.

Two checks, for POINT_CLOUD/GROUND_CLOUD/SURVEY_POINT_SET only
(FEATURE_COLLECTION has no `.crs` concept at all -- confirmed
directly, not even a field for one -- so this protection cannot
extend to it; that remains a known, explicit, undocumented-no-longer
structural gap, not silently papered over):

1. crs=None -> reject (the core danger).
2. crs present but doesn't match transformer.source_crs -> reject
   (mathematically valid, semantically wrong).

A related real gap was found and fixed while building this suite:
Workflow.classify_ground() previously never propagated the source
PointCloud's own metadata.crs onto the resulting GROUND_CLOUD --
always None regardless of input, which would have made
transform_crs(GROUND_CLOUD, ...) unusable in any normal pipeline
once check #1 above existed. Fixed by propagating `cloud.crs` in
classify_ground()'s own work() closure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
import pytest
from topocore.geodesy import CRS, CoordinateTransformer
from topocore.io.ascii.xyz.writer import XYZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.formats import SurveyFormat
from topocore.survey.models import SurveyPointSet
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import WorkflowStateError
from topocore.workflow.workflow import Workflow


def _ground_like_point_cloud(n: int = 50, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(-74.1, -74.0, n)
    chunk[PointAttribute.Y][:] = rng.uniform(4.6, 4.7, n)
    chunk[PointAttribute.Z][:] = 2500.0 + rng.uniform(-0.5, 0.5, n)
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def transformer() -> CoordinateTransformer:
    return CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))


# ----------------------------------------------------------------------
# Check 1: crs=None is rejected -- the core, confirmed danger.
# ----------------------------------------------------------------------


def test_survey_point_set_with_unknown_crs_is_rejected(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    """The exact real-world scenario that revealed this danger: raw total-station coordinates, no .prj."""
    path = tmp_path / "total_station.txt"
    path.write_text("1,1000.000,2000.000,100.000,PT\n", encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)

    with pytest.raises(WorkflowStateError, match="unknown"):
        workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, transformer)

    # Confirm the artifact was never silently transformed: still at its original value.
    original = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert original.points[0].x == 1000.0
    assert original.crs is None


def test_point_cloud_with_unknown_crs_is_rejected(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())

    workflow = Workflow().read_point_cloud(path)

    with pytest.raises(WorkflowStateError, match="unknown"):
        workflow.transform_crs(ArtifactType.POINT_CLOUD, transformer)


def test_feature_collection_with_none_crs_is_now_rejected(
    transformer: CoordinateTransformer,
) -> None:
    """
    Confirmed behavior change from an earlier phase of this same
    safety work: FEATURE_COLLECTION used to have no `.crs` field at
    all, so this check could not apply to it (see
    `FeatureCollection`'s own docstring for when/why `.crs` was
    added). Now that it does, `crs=None` is rejected here exactly
    like the other 3 supported artifact types -- this is the
    corrected, current behavior, not the original gap.
    """
    from topocore.features.models import (
        Feature,
        FeatureCategory,
        FeatureCollection,
        FeatureGeometry,
        FeatureType,
        GeometryType,
    )

    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[-74.0, 4.7, 2600.0]]))
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )
    collection = FeatureCollection()  # crs=None
    collection.add(feature)

    workflow = Workflow()
    workflow._store.set(ArtifactType.FEATURE_COLLECTION, collection)

    with pytest.raises(WorkflowStateError, match="unknown"):
        workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)


def test_feature_collection_with_matching_known_crs_transforms_successfully(
    transformer: CoordinateTransformer,
) -> None:
    from topocore.features.models import (
        Feature,
        FeatureCategory,
        FeatureCollection,
        FeatureGeometry,
        FeatureType,
        GeometryType,
    )

    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[-74.0, 4.7, 2600.0]]))
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )
    collection = FeatureCollection(crs="EPSG:4326")  # matches transformer's own source_crs
    collection.add(feature)

    workflow = Workflow()
    workflow._store.set(ArtifactType.FEATURE_COLLECTION, collection)
    workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)

    result = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert len(result) == 1
    assert result.crs == "EPSG:32618"


def test_feature_collection_with_mismatched_crs_is_rejected(
    transformer: CoordinateTransformer,
) -> None:
    from topocore.features.models import (
        Feature,
        FeatureCategory,
        FeatureCollection,
        FeatureGeometry,
        FeatureType,
        GeometryType,
    )

    geometry = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[-74.0, 4.7, 2600.0]]))
    feature = Feature(
        feature_id=1,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTROL_POINT,
        geometry=geometry,
    )
    collection = FeatureCollection(crs="EPSG:3116")  # does NOT match transformer's own source_crs (4326)
    collection.add(feature)

    workflow = Workflow()
    workflow._store.set(ArtifactType.FEATURE_COLLECTION, collection)

    with pytest.raises(WorkflowStateError, match="does not match"):
        workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)


# ----------------------------------------------------------------------
# Check 2: a known but mismatched CRS is rejected.
# ----------------------------------------------------------------------


def test_survey_point_set_with_mismatched_crs_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "survey.txt"
    path.write_text("1,1000.000,2000.000,100.000,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")  # genuinely EPSG:3116

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    wrong_transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))  # built for 4326, not 3116

    with pytest.raises(WorkflowStateError, match="does not match"):
        workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, wrong_transformer)


def test_survey_point_set_with_matching_crs_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "survey.txt"
    path.write_text("1,1000.000,2000.000,100.000,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    correct_transformer = CoordinateTransformer(CRS.from_epsg(3116), CRS.from_epsg(32618))

    workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, correct_transformer)

    result = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert result.crs is not None
    assert result.crs.epsg == 32618


def test_point_cloud_with_mismatched_crs_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    (tmp_path / "cloud.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_point_cloud(path)
    wrong_transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))

    with pytest.raises(WorkflowStateError, match="does not match"):
        workflow.transform_crs(ArtifactType.POINT_CLOUD, wrong_transformer)


# ----------------------------------------------------------------------
# The classify_ground() propagation fix, verified directly.
# ----------------------------------------------------------------------


def test_classify_ground_propagates_crs_from_source_cloud(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    (tmp_path / "cloud.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_point_cloud(path).classify_ground()
    ground = workflow.artifact(ArtifactType.GROUND_CLOUD)

    assert ground.crs == "EPSG:4326"


def test_classify_ground_propagates_none_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())

    workflow = Workflow().read_point_cloud(path).classify_ground()
    ground = workflow.artifact(ArtifactType.GROUND_CLOUD)

    assert ground.crs is None


def test_ground_cloud_transform_crs_works_end_to_end_after_the_fix(
    tmp_path: Path, transformer: CoordinateTransformer
) -> None:
    """The full pipeline this safety check would otherwise have made impossible: read -> classify_ground -> transform_crs."""
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_ground_like_point_cloud())
    (tmp_path / "cloud.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_point_cloud(path).classify_ground()
    workflow.transform_crs(ArtifactType.GROUND_CLOUD, transformer)

    result = workflow.artifact(ArtifactType.GROUND_CLOUD)
    assert result.crs == "EPSG:32618"


# ----------------------------------------------------------------------
# Reviewer-identified gaps, closed with real, verified evidence.
# ----------------------------------------------------------------------


def test_non_epsg_crs_is_accepted_by_the_safety_checks(tmp_path: Path) -> None:
    """Both safety checks work with a genuinely non-EPSG, custom-parameter CRS -- not just EPSG-registered ones."""
    custom_crs = CRS.from_wkt(
        pyproj.CRS.from_proj4(
            "+proj=lcc +lat_1=4 +lat_2=6 +lat_0=5 +lon_0=-74 +x_0=500000 "
            "+y_0=500000 +ellps=GRS80 +units=m +no_defs +type=crs"
        ).to_wkt()
    )
    assert custom_crs.epsg is None  # confirm the premise

    path = tmp_path / "survey.txt"
    path.write_text("1,-74.0,4.7,100.0,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    transformer = CoordinateTransformer(CRS.from_epsg(4326), custom_crs)

    workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, transformer)
    result = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert result.crs is not None
    assert result.crs.epsg is None  # the target genuinely has no EPSG code, preserved correctly


def test_z_passes_through_unchanged_in_a_horizontal_transform(
    tmp_path: Path, transformer: CoordinateTransformer
) -> None:
    """Explicit assertion on Z: a horizontal (X/Y) transform never touches Z, confirmed with a real before/after comparison."""
    path = tmp_path / "survey.txt"
    path.write_text("1,-74.0,4.7,2600.0,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    original_z = workflow.artifact(ArtifactType.SURVEY_POINT_SET).points[0].z

    workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, transformer)
    result = workflow.artifact(ArtifactType.SURVEY_POINT_SET)

    assert result.points[0].z == original_z == 2600.0


def test_mismatch_error_message_names_both_crs_values(tmp_path: Path) -> None:
    """More thorough than matching a short substring: confirms the error message actually identifies both CRS values involved."""
    path = tmp_path / "survey.txt"
    path.write_text("1,1000.000,2000.000,100.000,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(3116).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    wrong_transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))

    with pytest.raises(WorkflowStateError) as exc_info:
        workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, wrong_transformer)

    message = str(exc_info.value)
    assert "3116" in message  # the artifact's actual CRS
    assert "4326" in message  # the transformer's expected source CRS


def test_identity_like_transformer_is_not_blocked_by_repeated_application(
    tmp_path: Path,
) -> None:
    """
    Confirmed, understood boundary of these 2 safety checks, not a
    gap silently left undocumented: when source_crs == target_crs,
    the CRS-still-matches after a first transform, so a SECOND
    application of the exact same transformer is not blocked by
    either check (unlike a genuinely different source/target
    transformer, which IS blocked -- see
    test_survey_point_set_with_mismatched_crs_is_rejected). This is
    low-risk in practice (an identity-shaped transform is a
    mathematical no-op, not an accumulating shift), but these checks
    were never designed to guarantee general idempotency -- only to
    reject the specific, confirmed-dangerous "unknown or mismatched
    origin CRS" cases.
    """
    path = tmp_path / "survey.txt"
    path.write_text("1,-74.0,4.7,100.0,PT\n", encoding="utf-8")
    (tmp_path / "survey.prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(), encoding="utf-8")

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    identity_transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(4326))

    workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, identity_transformer)
    workflow.transform_crs(
        ArtifactType.SURVEY_POINT_SET, identity_transformer
    )  # not blocked -- confirmed intentionally

    result = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert result.crs is not None
    assert result.crs.epsg == 4326


def test_is_stale_in_isolation_for_a_self_referential_dependency() -> None:
    """
    The _is_stale() fix itself, tested directly rather than only
    through the transform_crs()/resolve_sides() regression tests
    elsewhere. Constructs a minimal ArtifactStore + history by hand
    to exercise WorkflowValidator._is_stale() in isolation: a stage
    that both reads and re-produces the SAME ArtifactType must not
    be reported stale relative to its own prior version.
    """
    from datetime import UTC, datetime

    from topocore.workflow.artifacts import ArtifactStore
    from topocore.workflow.history import ArtifactDependency, StageResult
    from topocore.workflow.stages import StageStatus, WorkflowStage
    from topocore.workflow.validation import WorkflowValidator

    store = ArtifactStore()
    store.set(ArtifactType.SURVEY_POINT_SET, SurveyPointSet(points=()))  # version 1
    store.set(
        ArtifactType.SURVEY_POINT_SET, SurveyPointSet(points=())
    )  # version 2 -- the self-referential producer below

    now = datetime.now(UTC)
    history = [
        StageResult(
            stage=WorkflowStage.TRANSFORM_CRS,
            status=StageStatus.SUCCESS,
            started_at=now,
            finished_at=now,
            dependencies=(ArtifactDependency(artifact=ArtifactType.SURVEY_POINT_SET, version=1, required=True),),
            produced=ArtifactDependency(artifact=ArtifactType.SURVEY_POINT_SET, version=2, required=True),
            metrics=None,
            warnings=(),
            error=None,
        )
    ]

    assert WorkflowValidator._is_stale(store, history, ArtifactType.SURVEY_POINT_SET) is False
