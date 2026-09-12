"""
Tests for Workflow.transform_crs() -- the optional, chainable stage
applying an already-built CoordinateTransformer to whichever of the 4
transformable artifact types (POINT_CLOUD, GROUND_CLOUD,
SURVEY_POINT_SET, FEATURE_COLLECTION) the caller names, via
topocore.geodesy.transform's own transform_point_cloud()/
transform_survey()/transform_feature_collection() functions.

Scope, confirmed and deliberately bounded during this feature's own
design review:
  - Only wraps existing, already-verified transform_* functions --
    never constructs a CRS or CoordinateTransformer itself.
  - Only supports the 4 types those functions exist for; TIN/DTM/
    CONTOURS/CLASSIFICATION_RESULT are rejected explicitly.
  - Deliberately does NOT implement: automatic CRS detection from any
    reader, an "input/processing/output CRS" concept on Workflow
    itself, or vertical datum/geoid height correction -- all
    explicitly out of scope, registered as separate future product
    decisions, not implemented here.

A real, significant bug was found and fixed WHILE building this
stage, in shared infrastructure (WorkflowValidator._is_stale()), not
in this stage's own code: any stage that both reads and re-produces
the SAME ArtifactType (this one, and the earlier RESOLVE_SIDES)
recorded a dependency on that type's own pre-call version -- but by
the time staleness is checked, the store's current version for that
type is already the NEW one the stage itself just wrote, so the
naive comparison always found a "mismatch" and reported the
artifact stale immediately after its own successful production.
Confirmed directly: calling `workflow.artifact(FEATURE_COLLECTION)`
right after `resolve_sides()` raised `StaleArtifactError`
unconditionally, undetected until now because
`test_resolve_sides.py`'s own tests read via the private `_store.get()`
directly rather than the public `artifact()` accessor. Fixed by
skipping a dependency in `_is_stale()`'s own comparison whenever it
refers to the same ArtifactType currently being checked -- confirmed
by reasoning (not just testing) that no legitimate staleness scenario
depends on this specific self-referential comparison, since once a
newer version supersedes a self-produced one, staleness is always
re-evaluated against that newer version's own production record
instead, never revisiting the superseded one.

A later safety fix to transform_crs() itself (see
test_transform_crs_safety.py for the full detail) now requires the
artifact's own current CRS to be known and to match the
transformer's own source_crs -- the SURVEY_POINT_SET/POINT_CLOUD/
GROUND_CLOUD tests below each declare a `.prj` accordingly, not as
an incidental detail but because the stage would otherwise reject
them.

Uses the `tmp_path` fixture throughout (converted from an earlier
`tempfile.TemporaryDirectory()`-based version for consistency with
every other test module in this project), with no change in test
semantics -- same file names, same content, same assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from topocore.geodesy import CRS, CoordinateTransformer
from topocore.io.ascii.xyz.writer import XYZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.formats import SurveyFormat
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import StaleArtifactError, WorkflowStateError
from topocore.workflow.workflow import Workflow


@pytest.fixture
def transformer() -> CoordinateTransformer:
    return CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))


def _sample_point_cloud(n: int = 20, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(-74.1, -74.0, n)
    chunk[PointAttribute.Y][:] = rng.uniform(4.6, 4.7, n)
    chunk[PointAttribute.Z][:] = rng.uniform(2500, 2600, n)
    cloud.add_chunk(chunk)
    return cloud


def _sample_ground_like_point_cloud(n: int = 50, seed: int = 0) -> PointCloud:
    """
    A near-flat surface (small Z noise on an otherwise flat plane) --
    GroundManager.extract() needs real ground structure to classify
    points as ground; pure random X/Y/Z noise (as in
    _sample_point_cloud()) gets almost entirely classified as
    non-ground, confirmed directly (point_count dropped to 1),
    leaving too few points for build_tin()'s own minimum-3-point
    requirement.
    """
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(-74.1, -74.0, n)
    chunk[PointAttribute.Y][:] = rng.uniform(4.6, 4.7, n)
    chunk[PointAttribute.Z][:] = 2500.0 + rng.uniform(-0.5, 0.5, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# The core regression: artifact() must not falsely report staleness
# for a same-type read-and-rewrite stage.
# ----------------------------------------------------------------------


def test_feature_collection_accessible_via_artifact_after_transform(
    tmp_path: Path, transformer: CoordinateTransformer
) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n")
    # transform_crs() now requires FEATURE_COLLECTION's own current
    # CRS to be known and match the transformer's source_crs
    # (EPSG:4326) -- a genuine safety fix, not a relaxable test
    # inconvenience; a .prj declares it here.
    (tmp_path / "survey.prj").write_text(CRS.from_epsg(4326).to_wkt())

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
    workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)

    # This line is the regression itself: before the _is_stale() fix,
    # this raised StaleArtifactError unconditionally.
    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert len(features) == 1


def test_resolve_sides_output_also_accessible_via_artifact(tmp_path: Path) -> None:
    """The same latent bug affected the earlier RESOLVE_SIDES stage -- confirm it's fixed there too."""
    path = tmp_path / "survey.csv"
    path.write_text("1,0.0,0.0,1.0,EJE\n2,0.0,10.0,1.0,EJE\n")

    workflow = (
        Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey().resolve_sides()
    )

    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    assert len(features) == 1


def test_genuine_staleness_still_detected_after_the_fix(tmp_path: Path) -> None:
    """The _is_stale() fix must not weaken real staleness detection -- only skip the self-referential false positive."""
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_sample_ground_like_point_cloud())

    workflow = Workflow().read_point_cloud(path).classify_ground().build_tin().detect_features(strict=False)
    workflow.build_tin()  # rebuilds TIN -- FEATURE_COLLECTION's own recorded dependency is now outdated

    with pytest.raises(StaleArtifactError, match="since-superseded"):
        workflow.artifact(ArtifactType.FEATURE_COLLECTION)


# ----------------------------------------------------------------------
# Happy path -- all 4 supported artifact types.
# ----------------------------------------------------------------------


def test_transform_feature_collection(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,ARBOL\n")
    # Same requirement as above.
    (tmp_path / "survey.prj").write_text(CRS.from_epsg(4326).to_wkt())

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
    workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)

    feature = next(iter(workflow.artifact(ArtifactType.FEATURE_COLLECTION)))
    assert feature.geometry.vertices[0][0] == pytest.approx(610908.87, abs=0.1)


def test_transform_survey_point_set(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,-74.0,4.7,2600.0,PT\n")
    # transform_crs() now requires the artifact's own CRS to be known
    # and match the transformer's source_crs (EPSG:4326) -- confirmed
    # a genuine safety fix, not a relaxable test inconvenience; a
    # .prj declares it here.
    (tmp_path / "survey.prj").write_text(CRS.from_epsg(4326).to_wkt())

    workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
    workflow.transform_crs(ArtifactType.SURVEY_POINT_SET, transformer)

    point = workflow.artifact(ArtifactType.SURVEY_POINT_SET).points[0]
    assert point.x == pytest.approx(610908.87, abs=0.1)


def test_transform_point_cloud(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_sample_point_cloud())
    # Same requirement as above, for PointCloud.
    (tmp_path / "cloud.prj").write_text(CRS.from_epsg(4326).to_wkt())

    workflow = Workflow().read_point_cloud(path)
    workflow.transform_crs(ArtifactType.POINT_CLOUD, transformer)

    transformed = workflow.artifact(ArtifactType.POINT_CLOUD)
    assert transformed.metadata.crs == "EPSG:32618"


def test_transform_ground_cloud(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    path = tmp_path / "cloud.xyz"
    XYZWriter(path).write(_sample_point_cloud())
    # Same requirement -- classify_ground() carries metadata.crs
    # through from the original read (confirmed directly, a fix made
    # alongside transform_crs()'s own safety checks -- see
    # test_transform_crs_safety.py), so declaring it once on the
    # source file covers GROUND_CLOUD too.
    (tmp_path / "cloud.prj").write_text(CRS.from_epsg(4326).to_wkt())

    workflow = Workflow().read_point_cloud(path).classify_ground()
    workflow.transform_crs(ArtifactType.GROUND_CLOUD, transformer)

    transformed = workflow.artifact(ArtifactType.GROUND_CLOUD)
    assert transformed.metadata.crs == "EPSG:32618"


# ----------------------------------------------------------------------
# Rejections.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsupported_type",
    [
        ArtifactType.TIN,
        ArtifactType.DTM,
        ArtifactType.CONTOURS,
        ArtifactType.CLASSIFICATION_RESULT,
    ],
)
def test_unsupported_artifact_types_rejected(
    unsupported_type: ArtifactType, transformer: CoordinateTransformer
) -> None:
    with pytest.raises(WorkflowStateError, match="does not support"):
        Workflow().transform_crs(unsupported_type, transformer)


def test_missing_artifact_rejected(transformer: CoordinateTransformer) -> None:
    with pytest.raises(WorkflowStateError, match="requires artifact"):
        Workflow().transform_crs(ArtifactType.POINT_CLOUD, transformer)


# ----------------------------------------------------------------------
# Full pipeline chain, including RESOLVE_SIDES + TRANSFORM_CRS together.
# ----------------------------------------------------------------------


def test_full_chain_with_resolve_sides_and_transform_crs(tmp_path: Path, transformer: CoordinateTransformer) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,0.0,0.0,100.0,EJE\n2,0.0,10.0,100.0,EJE\n3,-5.0,0.0,100.0,BORDE\n4,-5.0,10.0,100.0,BORDE\n")
    # Same requirement as the other transform_crs(FEATURE_COLLECTION, ...) tests.
    (tmp_path / "survey.prj").write_text(CRS.from_epsg(4326).to_wkt())

    workflow = (
        Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey().resolve_sides()
    )
    workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)

    assert [s.stage.value for s in workflow.result.stages] == [
        "read_survey",
        "build_features_from_survey",
        "resolve_sides",
        "transform_crs",
    ]
    assert all(s.status.value == "success" for s in workflow.result.stages)

    features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
    edge = next(f for f in features if f.feature_type.value == "pavement_edge")
    assert edge.attributes["side"] == "left"
