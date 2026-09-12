"""
Tests for Workflow.transform_vertical() -- the vertical-axis
counterpart to Workflow.transform_crs(), applying a real
geoid-based VerticalTransformer (constructed internally from
source_datum/target_datum/geoid) to whichever supported artifact
type is named.

The single most important property tested here: this stage must
NEVER produce an uncorrected Z silently. This directly closes the
danger confirmed during this feature's own design review -- plain
pyproj.Transformer was confirmed to silently return an unchanged
height when its own required vertical grid was missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from topocore.geodesy import VerticalDatum
from topocore.geodesy.vertical.geoid_grid import GeoidGrid
from topocore.survey.formats import SurveyFormat
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import WorkflowExecutionError, WorkflowStateError
from topocore.workflow.workflow import Workflow


@pytest.fixture
def source_datum() -> VerticalDatum:
    return VerticalDatum(name="WGS84 ellipsoidal")


@pytest.fixture
def target_datum() -> VerticalDatum:
    return VerticalDatum(name="Synthetic", geoid_model="SYNTHETIC")


def test_transform_vertical_on_survey_point_set(
    tmp_path: Path,
    synthetic_geoid_path: Path,
    source_datum: VerticalDatum,
    target_datum: VerticalDatum,
) -> None:
    path = tmp_path / "survey.csv"
    path.write_text("1,1.0,1.0,100.0,PT\n")

    geoid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    workflow = Workflow().read_survey(str(path), format=SurveyFormat.ID_XYZ_CODE)
    workflow.transform_vertical(
        ArtifactType.SURVEY_POINT_SET,
        source_datum=source_datum,
        target_datum=target_datum,
        geoid=geoid,
    )

    result = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert result.points[0].z == 50.0
    assert [s.stage.value for s in workflow.result.stages] == [
        "read_survey",
        "transform_vertical",
    ]


def test_unsupported_artifact_type_rejected(
    synthetic_geoid_path: Path, source_datum: VerticalDatum, target_datum: VerticalDatum
) -> None:
    geoid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    with pytest.raises(WorkflowStateError, match="does not support"):
        Workflow().transform_vertical(
            ArtifactType.TIN,
            source_datum=source_datum,
            target_datum=target_datum,
            geoid=geoid,
        )


def test_missing_artifact_rejected(
    synthetic_geoid_path: Path, source_datum: VerticalDatum, target_datum: VerticalDatum
) -> None:
    geoid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    with pytest.raises(WorkflowStateError, match="requires artifact"):
        Workflow().transform_vertical(
            ArtifactType.SURVEY_POINT_SET,
            source_datum=source_datum,
            target_datum=target_datum,
            geoid=geoid,
        )


def test_point_outside_grid_fails_the_whole_stage_never_silently_unchanged(
    tmp_path: Path,
    synthetic_geoid_path: Path,
    source_datum: VerticalDatum,
    target_datum: VerticalDatum,
) -> None:
    """The core safety guarantee, verified through the full Workflow stage: never a silently-unchanged Z."""
    path = tmp_path / "survey.csv"
    path.write_text("1,100.0,100.0,100.0,PT\n")  # outside the synthetic grid's own extent

    geoid = GeoidGrid.from_geotiff(synthetic_geoid_path)
    workflow = Workflow().read_survey(str(path), format=SurveyFormat.ID_XYZ_CODE)

    with pytest.raises(WorkflowExecutionError, match="outside the geoid grid"):
        workflow.transform_vertical(
            ArtifactType.SURVEY_POINT_SET,
            source_datum=source_datum,
            target_datum=target_datum,
            geoid=geoid,
        )

    # Confirm the artifact was never silently re-produced with an
    # unchanged Z: the stage failed, so SURVEY_POINT_SET must still
    # be at its original, untransformed value.
    original = workflow.artifact(ArtifactType.SURVEY_POINT_SET)
    assert original.points[0].z == 100.0
