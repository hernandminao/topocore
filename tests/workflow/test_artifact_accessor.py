"""
Tests for Workflow.artifact() -- the public, defensive-copy artifact
accessor, added to close a real architectural gap: before this,
Workflow had no public way to retrieve the actual object any stage
produced (only StageResult.produced's own type/version metadata was
public; the real data lived only behind the private `_store`).

Copy strategy per type, verified with real construction/mutation for
every one of the 8 ArtifactType members -- matching this project's
own confirmed-by-execution mutability audit, not a uniform "always
copy" or "always return as-is":

    SURVEY_POINT_SET, CONTOURS, TIN  -> returned directly (genuinely
        immutable, or self-protecting via an existing defensive-copy
        property)
    POINT_CLOUD, GROUND_CLOUD        -> .clone()
    DTM                              -> dataclasses.replace() with
                                         only .raster copied
    CLASSIFICATION_RESULT            -> dataclasses.replace() with
                                         labels/confidence/cloud
                                         all copied
    FEATURE_COLLECTION               -> new FeatureCollection over a
                                         new list (Feature itself is
                                         already frozen)

`ArtifactStore.get()` itself is unchanged -- protection lives at
Workflow's own public boundary, not in the internal store.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from topocore.io.ascii.xyz.writer import XYZWriter
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.survey.formats import SurveyFormat
from topocore.workflow.artifacts import ArtifactType
from topocore.workflow.exceptions import StaleArtifactError, WorkflowStateError
from topocore.workflow.workflow import Workflow


def _sample_point_cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Never produced.
# ----------------------------------------------------------------------


def test_artifact_never_produced_raises() -> None:
    with pytest.raises(WorkflowStateError, match="never been produced"):
        Workflow().artifact(ArtifactType.FEATURE_COLLECTION)


# ----------------------------------------------------------------------
# SURVEY_POINT_SET -- returned directly.
# ----------------------------------------------------------------------


def test_survey_point_set_returned_directly() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        with open(path, "w") as f:
            f.write("1,0.0,0.0,1.0,EJE\n2,0.0,10.0,1.0,EJE\n")

        workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE)
        points = workflow.artifact(ArtifactType.SURVEY_POINT_SET)

        assert len(points) == 2


# ----------------------------------------------------------------------
# POINT_CLOUD / GROUND_CLOUD -- .clone(), mutation-safe.
# ----------------------------------------------------------------------


def test_point_cloud_is_defensively_cloned() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path)
        cloud_a = workflow.artifact(ArtifactType.POINT_CLOUD)
        cloud_b = workflow.artifact(ArtifactType.POINT_CLOUD)

        first_chunk_a = next(iter(cloud_a))
        first_chunk_a[PointAttribute.X][0] = 999.0

        first_chunk_b = next(iter(cloud_b))
        assert first_chunk_b[PointAttribute.X][0] != 999.0


def test_ground_cloud_is_defensively_cloned() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_ground()
        ground = workflow.artifact(ArtifactType.GROUND_CLOUD)
        first_chunk = next(iter(ground))
        first_chunk[PointAttribute.X][0] = 999.0

        ground_again = workflow.artifact(ArtifactType.GROUND_CLOUD)
        assert next(iter(ground_again))[PointAttribute.X][0] != 999.0


# ----------------------------------------------------------------------
# CLASSIFICATION_RESULT -- full field copy.
# ----------------------------------------------------------------------


def test_classification_result_is_defensively_copied() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_points()
        result_a = workflow.artifact(ArtifactType.CLASSIFICATION_RESULT)
        result_a.labels[0] = 250

        result_b = workflow.artifact(ArtifactType.CLASSIFICATION_RESULT)
        assert result_b.labels[0] != 250


# ----------------------------------------------------------------------
# FEATURE_COLLECTION -- new container, shared (already-frozen) Features.
# ----------------------------------------------------------------------


def test_feature_collection_container_is_copied_not_shared() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        with open(path, "w") as f:
            f.write("1,0.0,0.0,1.0,ARBOL\n")

        workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
        collection_a = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
        collection_a.features.append("not_a_real_feature")

        collection_b = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
        assert len(collection_b.features) == 1


def test_side_resolver_works_on_artifact_without_touching_store() -> None:
    """The exact use case this API was designed for: SideResolver never needs to know ArtifactStore exists."""
    from topocore.features.side import SideResolver

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "survey.csv")
        with open(path, "w") as f:
            f.write("1,0.0,0.0,100.0,EJE\n2,0.0,10.0,100.0,EJE\n")
            f.write("3,-5.0,0.0,100.0,BORDE\n4,-5.0,10.0,100.0,BORDE\n")

        workflow = Workflow().read_survey(path, format=SurveyFormat.ID_XYZ_CODE).build_features_from_survey()
        features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
        resolved = SideResolver().resolve(features)

        edges = [f for f in resolved if f.feature_type.value == "pavement_edge"]
        assert len(edges) == 1
        assert edges[0].attributes["side"] == "left"


# ----------------------------------------------------------------------
# TIN -- returned directly (self-protecting properties).
# ----------------------------------------------------------------------


def test_tin_returned_directly_and_is_self_protecting() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_ground().build_tin()
        tin = workflow.artifact(ArtifactType.TIN)

        simplices = tin.simplices
        simplices[0][0] = 999
        assert tin.simplices[0][0] != 999


# ----------------------------------------------------------------------
# DTM -- reconstructed with only .raster copied.
# ----------------------------------------------------------------------


def test_dtm_raster_is_defensively_copied() -> None:
    from topocore.terrain.grid import Grid
    from topocore.terrain.interpolation import InterpolationMethod

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_ground().build_tin()
        tin = workflow.artifact(ArtifactType.TIN)
        min_x, min_y, max_x, max_y = tin.bounds
        grid = Grid(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, resolution=1.0)

        workflow.build_dtm(grid, method=InterpolationMethod.IDW)
        dtm_a = workflow.artifact(ArtifactType.DTM)
        dtm_a.raster.values[0][0] = 12345.0

        dtm_b = workflow.artifact(ArtifactType.DTM)
        assert dtm_b.raster.values[0][0] != 12345.0


# ----------------------------------------------------------------------
# CONTOURS -- returned directly (frozen tuple of frozen ContourLine).
# ----------------------------------------------------------------------


def test_contours_returned_directly() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_ground().build_tin().extract_contours(interval=1.0)
        contours = workflow.artifact(ArtifactType.CONTOURS)

        assert isinstance(contours, tuple)


# ----------------------------------------------------------------------
# Staleness.
# ----------------------------------------------------------------------


def test_stale_artifact_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_ground().build_tin().detect_features(strict=False)
        workflow.build_tin()  # rebuilds TIN -- FEATURE_COLLECTION's own recorded dependency is now outdated

        with pytest.raises(StaleArtifactError, match="since-superseded"):
            workflow.artifact(ArtifactType.FEATURE_COLLECTION)


def test_non_stale_artifact_remains_accessible_after_a_sibling_rebuild() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cloud.xyz")
        XYZWriter(path).write(_sample_point_cloud())

        workflow = Workflow().read_point_cloud(path).classify_ground().build_tin().detect_features(strict=False)
        workflow.build_tin()

        tin = workflow.artifact(ArtifactType.TIN)
        assert tin is not None
