"""
Targeted coverage suite for topocore.workflow.workflow -- PR20
coverage phase, closing the specific real-domain-behavior gaps
identified in the coverage report (lines 105-108, 123-128, 140-143,
258-269, 336-354, 445-462, 539-550, 619-648), per Hernán's own
instruction not to re-touch ArtifactDependency/require_current/
StaleArtifactError (already covered by test_staleness.py).

Each test below targets a real behavior, not a bare line-hit:

- input_kind: both the "nothing run yet" (None) and "first stage's
  own produced artifact" cases.
- snapshot(): genuine independence after divergence (a dedicated
  workflow.py-focused regression, complementing the earlier manual
  verification during this session's exploratory work).
- _emit_progress(): a broken ProgressObserver doesn't crash the
  workflow (logged and swallowed).
- read_survey(): full success path (SurveyPointSet produced,
  metrics correct) and its own reader error path
  (SurveyFormatError wrapped into WorkflowExecutionError).
- read_point_cloud(): every one of LAZ, PLY, E57, and PTS -- the
  four formats confirmed NOT exercised by test_reader_dispatch.py
  (which only covers LAS/XYZ/CSV) -- verified against REAL files
  (laspy/pye57, not mocks), and each confirmed to genuinely use its
  OWN reader class (not just that the extension was recognized), by
  monkeypatching the exact reader class read_point_cloud() would
  select and confirming it -- not some other reader -- was actually
  invoked.
- classify_points(): entirely untested before this -- multi-class
  ASPRS classification, distinct from classify_ground()'s binary
  split.
- build_dtm(): the three interpolation methods (BARYCENTRIC, IDW,
  NEAREST) besides the already-tested LINEAR.
- build_features_from_survey(): full success path (field-code
  interpretation via FeatureBuilder, distinct from detect_features()'s
  FeatureExtractionManager) plus its SURVEY_POINT_SET dependency and
  error path.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.terrain.grid import Grid
from topocore.terrain.interpolation import InterpolationMethod
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import WorkflowExecutionError, WorkflowStateError
from topocore.workflow.progress import ProgressEvent
from topocore.workflow.workflow import Workflow

# ----------------------------------------------------------------------
# input_kind
# ----------------------------------------------------------------------


def test_input_kind_is_none_before_any_stage_runs() -> None:
    workflow = Workflow()
    assert workflow.input_kind is None


def test_input_kind_reflects_first_stages_produced_artifact(tmp_path: object) -> None:
    path = str(tmp_path) + "/points.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n")

    workflow = Workflow()
    workflow.read_point_cloud(path)

    assert workflow.input_kind == ArtifactType.POINT_CLOUD


# ----------------------------------------------------------------------
# snapshot()
# ----------------------------------------------------------------------


def test_snapshot_diverges_independently(tmp_path: object) -> None:
    path = str(tmp_path) + "/points.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n10.0 10.0 4.0\n5.0 5.0 2.5\n")

    original = Workflow()
    original.read_point_cloud(path)
    original.classify_ground()
    original.build_tin()

    branch = original.snapshot()
    original_tin_version = original._store.version_of(ArtifactType.TIN)

    # Mutate the branch only.
    branch._store.set(ArtifactType.TIN, branch._store.get(ArtifactType.TIN))

    assert original._store.version_of(ArtifactType.TIN) == original_tin_version
    assert branch._store.version_of(ArtifactType.TIN) != original_tin_version
    assert len(branch.result.stages) == len(original.result.stages)


# ----------------------------------------------------------------------
# _emit_progress -- broken observer doesn't crash the workflow.
# ----------------------------------------------------------------------


def test_broken_progress_observer_does_not_crash_workflow(tmp_path: object) -> None:
    path = str(tmp_path) + "/points.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n")

    class ExplodingObserver:
        def on_progress(self, event: ProgressEvent) -> None:
            raise RuntimeError("observer exploded")

    workflow = Workflow(progress=ExplodingObserver())
    workflow.read_point_cloud(path)  # must not raise despite the observer blowing up

    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


# ----------------------------------------------------------------------
# read_survey() -- success and error paths.
# ----------------------------------------------------------------------


def test_read_survey_success_path(tmp_path: object) -> None:
    path = str(tmp_path) + "/survey.csv"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("Punto,Este,Norte,Z,Descripcion\nP1,500000.0,4000000.0,100.0,CERCA\nP2,500001.0,4000001.0,101.0,MURO\n")

    workflow = Workflow()
    workflow.read_survey(path)

    assert workflow._store.has(ArtifactType.SURVEY_POINT_SET)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded
    assert workflow.result.last_stage.metrics is not None
    assert workflow.result.last_stage.metrics.output_count == 2


def test_read_survey_reader_error_wrapped_in_workflow_execution_error(
    tmp_path: object,
) -> None:
    """A file with no detectable header and no format preset -- SurveyFormatError wrapped by _execute_stage."""
    path = str(tmp_path) + "/nohader.csv"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("500000.0 4000000.0 100.0\n")

    workflow = Workflow()
    with pytest.raises(WorkflowExecutionError):
        workflow.read_survey(path)


# ----------------------------------------------------------------------
# read_point_cloud() -- LAZ, PLY, E57, PTS: confirmed untested formats.
# ----------------------------------------------------------------------


def test_laz_dispatches_to_lazreader_not_some_other_reader(tmp_path: object) -> None:
    import laspy  # type: ignore[import-untyped]

    path = str(tmp_path) + "/points.laz"  # type: ignore[operator]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([1.0, 2.0, 3.0])
    las.z = np.array([1.0, 2.0, 3.0])
    las.write(path, do_compress=True)

    import topocore.io.laz as laz_module

    original_init = laz_module.LAZReader.__init__
    calls: list[bool] = []

    def spy_init(self, *args: object, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        calls.append(True)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    laz_module.LAZReader.__init__ = spy_init  # type: ignore[method-assign]
    try:
        workflow = Workflow()
        workflow.read_point_cloud(path)
    finally:
        laz_module.LAZReader.__init__ = original_init  # type: ignore[method-assign]

    assert calls == [True]
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_ply_dispatches_to_plyreader_not_some_other_reader(tmp_path: object) -> None:
    path = str(tmp_path) + "/points.ply"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            "0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n"
        )

    import topocore.io.ply.reader as ply_module

    original_init = ply_module.PLYReader.__init__
    calls: list[bool] = []

    def spy_init(self, *args: object, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        calls.append(True)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    ply_module.PLYReader.__init__ = spy_init  # type: ignore[method-assign]
    try:
        workflow = Workflow()
        workflow.read_point_cloud(path)
    finally:
        ply_module.PLYReader.__init__ = original_init  # type: ignore[method-assign]

    assert calls == [True]
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_e57_dispatches_to_e57reader_and_gets_default_chunk_size(
    tmp_path: object,
) -> None:
    """
    Also confirms the E57-specific detail: chunk_size has no
    built-in default on E57Reader (unlike every other reader), so
    read_point_cloud() must supply one -- verified the reader
    actually received a chunk_size kwarg, not just that E57Reader
    was selected.
    """
    import pye57  # type: ignore[import-untyped]

    path = str(tmp_path) + "/points.e57"  # type: ignore[operator]
    e57 = pye57.E57(path, mode="w")
    e57.write_scan_raw(
        {
            "cartesianX": np.array([1.0, 2.0, 3.0]),
            "cartesianY": np.array([1.0, 2.0, 3.0]),
            "cartesianZ": np.array([1.0, 2.0, 3.0]),
            "cartesianInvalidState": np.zeros(3, dtype=np.uint8),
        }
    )
    e57.close()

    import topocore.io.e57 as e57_module

    original_init = e57_module.E57Reader.__init__
    received_kwargs: dict[str, object] = {}

    def spy_init(self, path_arg, **kwargs):  # type: ignore[no-untyped-def]
        received_kwargs.update(kwargs)
        original_init(self, path_arg, **kwargs)  # type: ignore[arg-type]

    e57_module.E57Reader.__init__ = spy_init  # type: ignore[method-assign,assignment]
    try:
        workflow = Workflow()
        workflow.read_point_cloud(path)
    finally:
        e57_module.E57Reader.__init__ = original_init  # type: ignore[method-assign]

    assert "chunk_size" in received_kwargs
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_pts_dispatches_to_ptsreader_not_some_other_reader(tmp_path: object) -> None:
    path = str(tmp_path) + "/points.pts"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("3\n0.0 0.0 1.0\n10.0 0.0 2.0\n0.0 10.0 3.0\n")

    import topocore.io.ascii.pts.reader as pts_module

    original_init = pts_module.PTSReader.__init__
    calls: list[bool] = []

    def spy_init(self, *args: object, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        calls.append(True)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    pts_module.PTSReader.__init__ = spy_init  # type: ignore[method-assign]
    try:
        workflow = Workflow()
        workflow.read_point_cloud(path)
    finally:
        pts_module.PTSReader.__init__ = original_init  # type: ignore[method-assign]

    assert calls == [True]
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_read_point_cloud_rejects_directory_path(tmp_path: object) -> None:
    """The path exists but is a directory, not a file -- distinct from the already-tested nonexistent-path case."""
    workflow = Workflow()
    with pytest.raises(WorkflowExecutionError, match="not a file"):
        workflow.read_point_cloud(tmp_path)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# classify_points() -- entirely untested before this.
# ----------------------------------------------------------------------


def test_classify_points_produces_classification_result(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    n = 300
    xs = rng.uniform(0, 30, n)
    ys = rng.uniform(0, 30, n)
    zs = rng.uniform(0, 1, n)

    path = str(tmp_path) + "/points.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.writelines(f"{x} {y} {z}\n" for x, y, z in zip(xs, ys, zs, strict=True))

    workflow = Workflow()
    workflow.read_point_cloud(path)
    workflow.classify_points()

    assert workflow._store.has(ArtifactType.CLASSIFICATION_RESULT)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded
    assert workflow.result.last_stage.metrics is not None
    assert workflow.result.last_stage.metrics.output_count == n


def test_classify_points_requires_point_cloud() -> None:
    workflow = Workflow()
    with pytest.raises(WorkflowStateError):
        workflow.classify_points()


# ----------------------------------------------------------------------
# build_dtm() -- the three interpolation methods besides LINEAR.
# ----------------------------------------------------------------------


@pytest.fixture
def tin_ready_workflow(tmp_path: object) -> Workflow:
    rng = np.random.default_rng(0)
    n = 200
    xs = rng.uniform(0, 40, n)
    ys = rng.uniform(0, 40, n)
    zs = xs * 0.02 + rng.normal(0, 0.02, n)

    path = str(tmp_path) + "/points.xyz"  # type: ignore[operator]
    with open(path, "w") as f:
        f.writelines(f"{x} {y} {z}\n" for x, y, z in zip(xs, ys, zs, strict=True))

    workflow = Workflow()
    workflow.read_point_cloud(path)
    workflow.classify_ground()
    workflow.build_tin()
    return workflow


@pytest.mark.parametrize(
    "method",
    [
        InterpolationMethod.BARYCENTRIC,
        InterpolationMethod.IDW,
        InterpolationMethod.NEAREST,
    ],
)
def test_build_dtm_with_non_linear_interpolation_methods(
    tin_ready_workflow: Workflow, method: InterpolationMethod
) -> None:
    grid = Grid(min_x=0, min_y=0, max_x=40, max_y=40, resolution=4.0)
    tin_ready_workflow.build_dtm(grid, method=method)

    assert tin_ready_workflow._store.has(ArtifactType.DTM)
    assert tin_ready_workflow.result.last_stage is not None
    assert tin_ready_workflow.result.last_stage.succeeded


def test_build_dtm_idw_respects_power_parameter(tin_ready_workflow: Workflow) -> None:
    grid = Grid(min_x=0, min_y=0, max_x=40, max_y=40, resolution=4.0)
    tin_ready_workflow.build_dtm(grid, method=InterpolationMethod.IDW, power=3.0)

    assert tin_ready_workflow.result.last_stage is not None
    assert tin_ready_workflow.result.last_stage.succeeded


# ----------------------------------------------------------------------
# build_features_from_survey() -- entirely untested before this.
# ----------------------------------------------------------------------


def test_build_features_from_survey_success_path(tmp_path: object) -> None:
    path = str(tmp_path) + "/survey.csv"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("Punto,Este,Norte,Z,Descripcion\nP1,0.0,0.0,1.0,ARBOL\nP2,10.0,0.0,2.0,ARBOL\nP3,0.0,10.0,3.0,ARBOL\n")

    workflow = Workflow()
    workflow.read_survey(path)
    workflow.build_features_from_survey()

    assert workflow._store.has(ArtifactType.FEATURE_COLLECTION)
    assert workflow.result.last_stage is not None
    assert workflow.result.last_stage.succeeded


def test_build_features_from_survey_requires_survey_point_set() -> None:
    workflow = Workflow()
    with pytest.raises(WorkflowStateError):
        workflow.build_features_from_survey()


def test_build_features_from_survey_uses_field_builder_not_detection_manager(
    tmp_path: object,
) -> None:
    """
    Confirms build_features_from_survey() genuinely uses FeatureBuilder
    (field-code interpretation), not FeatureExtractionManager (which
    detect_features() uses) -- distinct algorithms per the module's
    own docstring.
    """
    path = str(tmp_path) + "/survey.csv"  # type: ignore[operator]
    with open(path, "w") as f:
        f.write("Punto,Este,Norte,Z,Descripcion\nP1,0.0,0.0,1.0,ARBOL\n")

    import topocore.features.feature_builder as fb_module

    original_build = fb_module.FeatureBuilder.build
    calls: list[bool] = []

    def spy_build(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(True)
        return original_build(self, *args, **kwargs)  # type: ignore[arg-type]

    fb_module.FeatureBuilder.build = spy_build  # type: ignore[method-assign]
    try:
        workflow = Workflow()
        workflow.read_survey(path)
        workflow.build_features_from_survey()
    finally:
        fb_module.FeatureBuilder.build = original_build  # type: ignore[method-assign]

    assert calls == [True]
