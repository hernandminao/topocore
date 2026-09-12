"""
Tests for Workflow.resolve_sides() -- the optional, chainable stage
wrapping topocore.features.side.SideResolver.

New capability. No existing Workflow stage's behavior changed:
RESOLVE_SIDES both reads and re-produces ArtifactType.FEATURE_COLLECTION
(does not introduce a new artifact type), following exactly the same
`_execute_stage`/`WorkflowValidator` template already used by
`build_features_from_survey()`/`detect_features()`.

Confirmed directly, before writing this suite: Workflow.
build_features_from_survey() always uses FeatureBuilder's own legacy
(non-grammar) mode -- it has no way to pass use_field_code_grammar=True
through Workflow's own public API. Every survey file below therefore
uses legacy-compatible codes (no BASE.FIGURE.COMMAND syntax), with
runs separated by an intervening point of a different code (matching
how `_group_runs()` actually splits consecutive-same-code runs) --
not the grammar-mode syntax used in `features/side/test_resolver.py`'s
own direct-SideResolver tests.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from topocore.survey.formats import SurveyFormat
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import WorkflowExecutionError, WorkflowStateError
from topocore.workflow.workflow import Workflow


def _write_survey_with_two_edges(path: str) -> None:
    """EJE + a left BORDE run + a mid EJE point (splits the runs) + a right BORDE run."""
    with open(path, "w") as f:
        f.write("1,0.0,0.0,100.0,EJE\n")
        f.write("2,0.0,10.0,100.0,EJE\n")
        f.write("3,-5.0,0.0,100.0,BORDE\n")
        f.write("4,-5.0,10.0,100.0,BORDE\n")
        f.write("5,0.0,20.0,100.0,EJE\n")
        f.write("6,5.0,0.0,100.0,BORDE\n")
        f.write("7,5.0,10.0,100.0,BORDE\n")


def test_full_chain_read_survey_build_resolve() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        _write_survey_with_two_edges(path)

        workflow = (
            Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey().resolve_sides()
        )

        assert [s.stage.value for s in workflow.result.stages] == [
            "read_survey",
            "build_features_from_survey",
            "resolve_sides",
        ]
        assert all(s.status.value == "success" for s in workflow.result.stages)

        features = workflow._store.get(ArtifactType.FEATURE_COLLECTION)
        edges = {f.feature_id: f for f in features if f.feature_type.value == "pavement_edge"}
        assert len(edges) == 2
        sides = sorted(f.attributes["side"] for f in edges.values())
        assert sides == ["left", "right"]
        assert all(f.attributes["side_method"] == "geometric" for f in edges.values())


def test_resolve_sides_forwards_kwargs_to_resolver() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        _write_survey_with_two_edges(path)

        workflow = (
            Workflow()
            .read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
            .build_features_from_survey()
            .resolve_sides(max_distance=1.0)  # real distance is ~5m -- forces UNKNOWN
        )

        features = workflow._store.get(ArtifactType.FEATURE_COLLECTION)
        edges = [f for f in features if f.feature_type.value == "pavement_edge"]
        assert all(f.attributes["side"] == "unknown" for f in edges)
        assert all(f.attributes["side_method"] == "unknown" for f in edges)


def test_resolve_sides_requires_feature_collection() -> None:
    with pytest.raises(WorkflowStateError, match="feature_collection"):
        Workflow().resolve_sides()


def test_resolve_sides_rejects_unknown_kwarg() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        with open(path, "w") as f:
            f.write("1,0.0,0.0,1.0,EJE\n2,0.0,10.0,1.0,EJE\n")

        workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()

        with pytest.raises(WorkflowExecutionError, match="unexpected keyword argument"):
            workflow.resolve_sides(not_a_real_kwarg=123)


def test_resolve_sides_on_collection_with_no_relevant_types_is_a_no_op() -> None:
    """A collection with no CENTERLINE/PAVEMENT_EDGE Features passes through with nothing modified -- not an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        with open(path, "w") as f:
            f.write("1,0.0,0.0,1.0,ARBOL\n")

        workflow = (
            Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey().resolve_sides()
        )

        features = workflow._store.get(ArtifactType.FEATURE_COLLECTION)
        assert len(features) == 1
        assert "side" not in next(iter(features)).attributes


def test_resolve_sides_after_detect_features_also_works() -> None:
    """RESOLVE_SIDES reads FEATURE_COLLECTION regardless of which stage produced it (BUILD_FEATURES_FROM_SURVEY or DETECT_FEATURES)."""
    import numpy as np
    from topocore.pointcloud.attributes import PointAttribute
    from topocore.pointcloud.chunk import Chunk
    from topocore.pointcloud.pointcloud import PointCloud

    rng = np.random.default_rng(0)
    n = 30
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        from topocore.io.ascii.xyz.writer import XYZWriter

        XYZWriter(path).write(cloud)

        workflow = Workflow().read_point_cloud(path).detect_features(strict=False).resolve_sides()

        assert workflow.result.stages[-1].stage.value == "resolve_sides"
        assert workflow.result.stages[-1].status.value == "success"
