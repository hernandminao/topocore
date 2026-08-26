"""
Full end-to-end pipeline regression suite for topocore.workflow.workflow
-- PR20 coverage phase.

Complements test_reader_dispatch.py and test_staleness.py (from the
earlier reader-dispatch/staleness-detection fixes) with:

1. A complete, real 8-stage pipeline run (read_point_cloud ->
   classify_ground -> build_tin -> build_dtm -> extract_contours ->
   detect_features -> export_dxf -> export_gpkg) against real,
   synthetic survey-style data, confirming every stage succeeds and
   is correctly recorded in history.

2. Two real bugs found and fixed in this session:

   a. detect_features() previously also tracked ArtifactType.
      GROUND_CLOUD for staleness, but DetectionContext (features/
      protocols.py) has no `ground_cloud` field at all -- no
      detector reachable through this pathway can ever consume it.
      This caused a real, reproducible false-positive rejection:
      detect_features() incorrectly raised StaleArtifactError
      whenever GROUND_CLOUD alone became stale, even though every
      input it actually uses (POINT_CLOUD, and whichever of TIN/DTM/
      CLASSIFICATION_RESULT are present) was fully fresh and correct.

   b. export_dxf() called DXFExporter(**exporter_kwargs) directly,
      but DXFExporter.__init__ takes a single `context:
      ExportContext | None` parameter, not arbitrary keyword
      arguments -- confirmed directly: export_dxf(path,
      strict=False) crashed with "unexpected keyword argument
      'strict'" despite the method's own **exporter_kwargs: Any
      signature advertising pass-through configuration.
      export_gpkg() already did this correctly (builds a
      GPKGExportOptions from kwargs, then passes that single
      object) -- export_dxf() now mirrors that pattern.

3. Precondition-failure history semantics: a WorkflowStateError (or
   StaleArtifactError) raised by validation BEFORE _execute_stage is
   called never appends a FAILED entry to history -- confirmed this
   is deliberate, not a gap: the stage never actually attempted to
   run, so there is nothing to record as "failed work."

4. Out-of-order pipeline stages and mutually-exclusive READ stages
   correctly rejected with WorkflowStateError.

5. Full transitive staleness propagation through a real, deeper
   pipeline (read -> classify_ground -> build_tin -> build_dtm ->
   extract_contours), not just the 2-hop case already covered in
   test_staleness.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import InterpolationMethod
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import StaleArtifactError, WorkflowStateError
from topocore.workflow.stages import WorkflowStage
from topocore.workflow.workflow import Workflow


@pytest.fixture
def pipeline_xyz_file(tmp_path: object) -> str:
    rng = np.random.default_rng(0)
    n = 400
    xs = rng.uniform(0, 50, n)
    ys = rng.uniform(0, 50, n)
    zs = xs * 0.02 + rng.normal(0, 0.05, n)  # gentle slope + noise

    path = str(tmp_path) + "/pipeline.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.writelines(f"{x} {y} {z}\n" for x, y, z in zip(xs, ys, zs, strict=True))
    return path


@pytest.fixture
def grid() -> Grid:
    return Grid(min_x=0, min_y=0, max_x=50, max_y=50, resolution=2.0)


def _bump(workflow: Workflow, artifact_type: ArtifactType) -> None:
    current = workflow._store.get(artifact_type)
    workflow._store.set(artifact_type, current)


# ----------------------------------------------------------------------
# Full 8-stage pipeline, real data, every stage succeeds.
# ----------------------------------------------------------------------


def test_full_pipeline_all_eight_stages_succeed(pipeline_xyz_file: str, grid: Grid, tmp_path: object) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.build_tin()
    workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)
    workflow.extract_contours(interval=1.0)
    workflow.detect_features(strict=False)
    workflow.export_dxf(str(tmp_path) + "/out.dxf")  # type: ignore[operator]
    workflow.export_gpkg(str(tmp_path) + "/out.gpkg", epsg=32618)  # type: ignore[operator]

    assert len(workflow.result.stages) == 8
    assert all(stage.status.value == "success" for stage in workflow.result.stages)


def test_pipeline_produces_expected_artifacts(pipeline_xyz_file: str, grid: Grid) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.build_tin()
    workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)

    assert workflow._store.has(ArtifactType.POINT_CLOUD)
    assert workflow._store.has(ArtifactType.GROUND_CLOUD)
    assert workflow._store.has(ArtifactType.TIN)
    assert workflow._store.has(ArtifactType.DTM)


# ----------------------------------------------------------------------
# Bug 1: GROUND_CLOUD false-positive staleness rejection in detect_features().
# ----------------------------------------------------------------------


def test_detect_features_not_falsely_rejected_by_unrelated_ground_cloud_staleness(
    pipeline_xyz_file: str,
) -> None:
    """
    The exact regression: before the fix, detect_features() raised
    StaleArtifactError purely because GROUND_CLOUD became stale, even
    though DetectionContext never actually consumes GROUND_CLOUD data
    at all (confirmed: it has no such field).
    """
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()  # produces GROUND_CLOUD, consuming POINT_CLOUD v1

    _bump(workflow, ArtifactType.POINT_CLOUD)  # POINT_CLOUD v2; GROUND_CLOUD now stale, never rebuilt

    workflow.detect_features(strict=False)  # must NOT raise despite GROUND_CLOUD being stale
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_detect_features_still_rejects_genuinely_stale_point_cloud(
    pipeline_xyz_file: str,
) -> None:
    """Control case: staleness of an artifact detect_features() DOES actually consume must still be caught."""
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)

    _bump(workflow, ArtifactType.POINT_CLOUD)

    # POINT_CLOUD itself was never "rebuilt" by anything downstream, so there's
    # no staleness relative to itself here -- instead confirm TIN staleness
    # (an artifact DetectionContext DOES consume) is still correctly caught.
    workflow2 = Workflow()
    workflow2.read_point_cloud(pipeline_xyz_file)
    workflow2.classify_ground()
    workflow2.build_tin()
    _bump(workflow2, ArtifactType.POINT_CLOUD)

    with pytest.raises(StaleArtifactError):
        workflow2.detect_features(strict=False)


# ----------------------------------------------------------------------
# Bug 2: export_dxf() crashing on any non-empty exporter_kwargs.
# ----------------------------------------------------------------------


def test_export_dxf_accepts_configuration_kwargs(pipeline_xyz_file: str, tmp_path: object) -> None:
    """
    The exact regression: before the fix, this crashed with
    "DXFExporter.__init__() got an unexpected keyword argument
    'strict'".
    """
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.detect_features(strict=False)

    workflow.export_dxf(str(tmp_path) + "/configured.dxf", strict=False)  # type: ignore[operator]
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_export_dxf_still_works_with_no_kwargs(pipeline_xyz_file: str, tmp_path: object) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.detect_features(strict=False)

    workflow.export_dxf(str(tmp_path) + "/default.dxf")  # type: ignore[operator]
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_export_gpkg_already_correctly_accepted_kwargs(pipeline_xyz_file: str, tmp_path: object) -> None:
    """Confirms export_gpkg's already-correct kwargs handling (the pattern export_dxf now mirrors)."""
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.detect_features(strict=False)

    workflow.export_gpkg(str(tmp_path) + "/configured.gpkg", epsg=32618, strict=False)  # type: ignore[operator]
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


# ----------------------------------------------------------------------
# Precondition-failure history semantics.
# ----------------------------------------------------------------------


def test_missing_artifact_precondition_failure_does_not_pollute_history(
    pipeline_xyz_file: str,
) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)

    with pytest.raises(WorkflowStateError):
        workflow.build_tin()  # requires GROUND_CLOUD, not produced yet

    # Only the successful read_point_cloud stage should be in history --
    # the rejected build_tin() call never actually attempted work.
    assert len(workflow.result.stages) == 1
    assert workflow.result.stages[0].stage == WorkflowStage.READ_POINT_CLOUD


def test_stale_artifact_precondition_failure_does_not_pollute_history(
    pipeline_xyz_file: str,
) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.build_tin()

    stages_before = len(workflow.result.stages)
    _bump(workflow, ArtifactType.POINT_CLOUD)

    with pytest.raises(StaleArtifactError):
        workflow.extract_contours(interval=1.0)

    assert len(workflow.result.stages) == stages_before


# ----------------------------------------------------------------------
# Out-of-order stages and mutually-exclusive READ stages.
# ----------------------------------------------------------------------


def test_out_of_order_stage_rejected(pipeline_xyz_file: str, grid: Grid) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()

    with pytest.raises(WorkflowStateError, match="tin"):
        workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)


def test_read_point_cloud_then_read_survey_rejected(pipeline_xyz_file: str) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)

    with pytest.raises(WorkflowStateError):
        workflow.read_survey(pipeline_xyz_file)


# ----------------------------------------------------------------------
# Deeper transitive staleness propagation (3+ hops, real pipeline).
# ----------------------------------------------------------------------


def test_transitive_staleness_propagates_through_full_pipeline(pipeline_xyz_file: str, grid: Grid) -> None:
    """
    POINT_CLOUD -> GROUND_CLOUD -> TIN -> DTM/CONTOURS, all built
    fresh, then POINT_CLOUD alone is re-read. Every downstream stage
    that transitively depends on it must reject, even though none of
    the intermediate artifacts (GROUND_CLOUD, TIN) were ever
    themselves rebuilt (their own version numbers never changed).
    """
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.build_tin()
    workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)
    workflow.extract_contours(interval=1.0)

    _bump(workflow, ArtifactType.POINT_CLOUD)

    with pytest.raises(StaleArtifactError):
        workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)

    with pytest.raises(StaleArtifactError):
        workflow.extract_contours(interval=1.0)


def test_rebuilding_the_whole_chain_resolves_staleness(pipeline_xyz_file: str, grid: Grid) -> None:
    workflow = Workflow()
    workflow.read_point_cloud(pipeline_xyz_file)
    workflow.classify_ground()
    workflow.build_tin()

    _bump(workflow, ArtifactType.POINT_CLOUD)

    workflow.classify_ground()  # rebuild -- now correctly consumes POINT_CLOUD v2
    workflow.build_tin()  # rebuild -- now correctly consumes GROUND_CLOUD v2

    workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)  # must not raise anymore
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded
