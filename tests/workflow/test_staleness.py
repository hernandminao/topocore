"""
Regression suite for Workflow's transitive artifact-staleness
detection (WorkflowValidator.require_current/_is_stale,
StaleArtifactError) -- PR19.

This is genuinely new functionality (validation.py's own docstring
explicitly deferred it: "Staleness checks belong to PR19, once
ArtifactDependency has real consumers... not made now"), not a bug
fix -- the underlying infrastructure (ArtifactDependency, StageResult
history, ArtifactStore.version_of) already existed and correctly
recorded what was needed; nothing was using it.

The decisive test is the transitive scenario: POINT_CLOUD v1 -> TIN
v1 -> DTM v1, then POINT_CLOUD is re-read (v2) WITHOUT rebuilding TIN
or DTM. A naive one-hop check (does DTM's recorded TIN version match
TIN's CURRENT version?) would wrongly call this fresh, since TIN's
own version number never changed. Only a transitive walk through the
production history correctly detects that TIN itself was built from
now-stale POINT_CLOUD, and DTM (built from that TIN) is therefore
stale too.
"""

from __future__ import annotations

import pytest

from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import InterpolationMethod
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import StaleArtifactError
from topocore.workflow.workflow import Workflow


@pytest.fixture
def grid() -> Grid:
    return Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=1.0)


def _bump_point_cloud_version(workflow: Workflow) -> None:
    """Simulates re-reading POINT_CLOUD without going through a full second read_point_cloud() call."""
    current = workflow._store.get(ArtifactType.POINT_CLOUD)
    workflow._store.set(ArtifactType.POINT_CLOUD, current)


def test_fresh_pipeline_has_no_staleness(xyz_file: str, grid: Grid) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(xyz_file)
    workflow.classify_ground()
    workflow.build_tin()
    workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)  # must not raise

    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_direct_dependency_staleness_detected(xyz_file: str) -> None:
    """One-hop case: build_tin() directly requires GROUND_CLOUD."""
    workflow = Workflow()
    workflow.read_point_cloud(xyz_file)
    workflow.classify_ground()

    _bump_point_cloud_version(workflow)  # POINT_CLOUD v2, GROUND_CLOUD still built from v1

    with pytest.raises(StaleArtifactError):
        workflow.build_tin()


def test_transitive_staleness_detected_two_hops(xyz_file: str, grid: Grid) -> None:
    """
    The exact regression: build_dtm() only directly requires TIN
    (never touches POINT_CLOUD), but TIN itself was transitively
    built from a now-superseded POINT_CLOUD. A one-hop check on TIN's
    own version number (unchanged, since TIN was never rebuilt) would
    miss this entirely.
    """
    workflow = Workflow()
    workflow.read_point_cloud(xyz_file)
    workflow.classify_ground()
    workflow.build_tin()

    _bump_point_cloud_version(workflow)  # POINT_CLOUD v2; TIN, GROUND_CLOUD never rebuilt

    with pytest.raises(StaleArtifactError):
        workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)


def test_rebuilding_the_stale_link_resolves_staleness(xyz_file: str, grid: Grid) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(xyz_file)
    workflow.classify_ground()
    workflow.build_tin()

    _bump_point_cloud_version(workflow)

    workflow.classify_ground()  # rebuild -- now correctly consumes POINT_CLOUD v2
    workflow.build_tin()  # rebuild -- now correctly consumes GROUND_CLOUD v2

    workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)  # must not raise anymore
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_extract_contours_also_detects_transitive_staleness(xyz_file: str) -> None:
    """Confirms the fix was applied at every required-artifact call site, not just build_dtm()."""
    workflow = Workflow()
    workflow.read_point_cloud(xyz_file)
    workflow.classify_ground()
    workflow.build_tin()

    _bump_point_cloud_version(workflow)

    with pytest.raises(StaleArtifactError):
        workflow.extract_contours(interval=1.0)


def test_stale_error_is_a_workflow_state_error(xyz_file: str) -> None:
    """
    StaleArtifactError must be a WorkflowStateError subtype (a
    pipeline-progress precondition failure), never
    WorkflowValidationError -- that class's own docstring explicitly
    reserves it for a stage call's own parameters, "never used for
    missing artifacts."
    """
    from topocore.workflow.exceptions import WorkflowStateError, WorkflowValidationError

    assert issubclass(StaleArtifactError, WorkflowStateError)
    assert not issubclass(StaleArtifactError, WorkflowValidationError)


def test_require_current_does_nothing_when_artifact_absent() -> None:
    """require_current() only checks freshness IF present -- presence is require()'s job."""
    from topocore.workflow.artifacts import ArtifactStore
    from topocore.workflow.stages import WorkflowStage
    from topocore.workflow.validation import WorkflowValidator

    store = ArtifactStore()
    WorkflowValidator.require_current(WorkflowStage.BUILD_DTM, store, [], ArtifactType.TIN)  # must not raise
