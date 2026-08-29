"""
Coverage audit tests for topocore.features.buildings.roofs.RoofDetector.

PR22 coverage hardening -- the final module of this audit round.

This is the FIRST genuine, working construction of NormalResult in a
real consumer pipeline observed in this whole session: Phase 5.3's
own audit of processing/normals/models.py confirmed NormalResult(
had zero constructor calls anywhere in TopoCore's own internal code
(referenced only as a type on DetectionContext). Confirmed directly
here that RoofDetector genuinely consumes it correctly end-to-end
when a caller constructs the full detection pipeline by hand --
TopoCore's own code simply doesn't wire this up automatically yet
(no bug, just unconnected infrastructure, consistent with the
original finding).

Confirmed directly, before writing tests, that _detect()'s three
size-mismatch checks (classification, normals, extracted XYZ) are
genuinely reachable and serve a real purpose distinct from
ClassificationResult's own self-consistency validation:
ClassificationResult.__post_init__ only guarantees
labels.shape[0] == classification.cloud.point_count (its OWN cloud),
not that this matches whatever DIFFERENT cloud a caller happens to
put in DetectionContext.cloud. Verified this exact mismatch scenario
(classification built from a 10-point cloud, DetectionContext given
a different 30-point cloud) genuinely triggers the check.

_triangulate()'s three "return None" paths (fewer than 3 points,
fewer than 3 unique XY positions, QhullError from collinear XY) are
each confirmed reachable with real, hand-verified geometric inputs
(not assumed).

_group_by_orientation()'s zero-norm-normal skip is confirmed
reachable and correctly excludes that point from every group (not
silently assigning it to group 0).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.features.buildings.roofs import RoofDetector
from topocore.features.exceptions import DetectionError
from topocore.features.protocols import DetectionContext
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult
from topocore.processing.normals.models import NormalResult


def _cloud(n: int, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_normal_z": -0.1}, "min_normal_z"),
        ({"min_normal_z": 1.1}, "min_normal_z"),
        ({"orientation_angle_deg": 0}, "orientation_angle_deg"),
        ({"orientation_angle_deg": 90}, "orientation_angle_deg"),
        ({"eps": 0}, "eps must be positive"),
        ({"min_points": 0}, "min_points"),
    ],
)
def test_constructor_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(DetectionError, match=match):
        RoofDetector(**kwargs)


def test_name() -> None:
    assert RoofDetector().name() == "roofs"


# ----------------------------------------------------------------------
# End-to-end detection -- the full pipeline with real geometry.
# ----------------------------------------------------------------------


def test_detects_a_flat_horizontal_roof_plane() -> None:
    rng = np.random.default_rng(0)
    ground_n, roof_n = 50, 60
    gx, gy, gz = (
        rng.uniform(0, 20, ground_n),
        rng.uniform(0, 20, ground_n),
        np.zeros(ground_n),
    )
    rx, ry, rz = (
        rng.uniform(5, 15, roof_n),
        rng.uniform(5, 15, roof_n),
        np.full(roof_n, 5.0),
    )

    n = ground_n + roof_n
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.concatenate([gx, rx])
    chunk[PointAttribute.Y][:] = np.concatenate([gy, ry])
    chunk[PointAttribute.Z][:] = np.concatenate([gz, rz])
    cloud.add_chunk(chunk)

    labels = np.concatenate(
        [
            np.full(ground_n, PointClassification.GROUND.value, dtype=np.int64),
            np.full(roof_n, PointClassification.BUILDING.value, dtype=np.int64),
        ]
    )
    classification = ClassificationResult(labels=labels, cloud=cloud)
    normals = np.zeros((n, 3))
    normals[:, 2] = 1.0
    normal_result = NormalResult(normals=normals)

    context = DetectionContext(cloud=cloud, classification=classification, normals=normal_result)
    result = RoofDetector(min_normal_z=0.5, orientation_angle_deg=15.0, eps=2.0, min_points=10).detect(context)

    assert len(result) == 1
    feature = next(iter(result))
    assert feature.geometry.geometry_type.value == "mesh"
    assert feature.attributes["mean_normal_z"] == pytest.approx(1.0)


def test_no_building_points_returns_empty_collection() -> None:
    n = 30
    cloud = _cloud(n)
    labels = np.full(n, PointClassification.GROUND.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, cloud=cloud)
    normals = np.zeros((n, 3))
    normals[:, 2] = 1.0
    normal_result = NormalResult(normals=normals)

    context = DetectionContext(cloud=cloud, classification=classification, normals=normal_result)
    result = RoofDetector().detect(context)

    assert len(result) == 0


def test_missing_required_input_raises() -> None:
    n = 30
    cloud = _cloud(n)
    labels = np.full(n, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, cloud=cloud)

    context = DetectionContext(cloud=cloud, classification=classification)  # normals missing

    with pytest.raises(DetectionError, match="normals"):
        RoofDetector().detect(context)


def test_classification_cloud_mismatch_with_context_cloud_is_caught() -> None:
    """
    ClassificationResult's own validation only guarantees internal
    self-consistency with its OWN cloud -- it cannot catch a caller
    supplying a DIFFERENT cloud in DetectionContext.
    """
    small_cloud = _cloud(10)
    big_cloud = _cloud(30)
    labels = np.full(10, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, cloud=small_cloud)
    normals = np.zeros((30, 3))
    normals[:, 2] = 1.0
    normal_result = NormalResult(normals=normals)

    context = DetectionContext(cloud=big_cloud, classification=classification, normals=normal_result)

    with pytest.raises(DetectionError, match="Classification size does not match"):
        RoofDetector().detect(context)


def test_normals_mismatch_with_context_cloud_is_caught() -> None:
    n = 30
    cloud = _cloud(n)
    labels = np.full(n, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, cloud=cloud)
    normal_result = NormalResult(normals=np.zeros((10, 3)) + np.array([0.0, 0.0, 1.0]))

    context = DetectionContext(cloud=cloud, classification=classification, normals=normal_result)

    with pytest.raises(DetectionError, match="Normals size does not match"):
        RoofDetector().detect(context)


# ----------------------------------------------------------------------
# _triangulate() -- all 3 "return None" paths, confirmed reachable.
# ----------------------------------------------------------------------


def test_triangulate_returns_none_for_fewer_than_three_points() -> None:
    result = RoofDetector._triangulate(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert result is None


def test_triangulate_returns_none_for_fewer_than_three_unique_xy() -> None:
    """3 points but all sharing the same XY position (only Z differs) -- cannot form a 2D triangulation."""
    duplicated_xy = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    result = RoofDetector._triangulate(duplicated_xy)
    assert result is None


def test_triangulate_returns_none_for_collinear_xy() -> None:
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    result = RoofDetector._triangulate(collinear)
    assert result is None


def test_triangulate_happy_path() -> None:
    valid = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    result = RoofDetector._triangulate(valid)
    assert result is not None
    assert result.vertex_count == 4


def test_triangulate_rejects_wrong_shape() -> None:
    with pytest.raises(DetectionError, match=r"shape \(n, 3\)"):
        RoofDetector._triangulate(np.array([0.0, 0.0, 0.0]))


# ----------------------------------------------------------------------
# _group_by_orientation().
# ----------------------------------------------------------------------


def test_group_by_orientation_rejects_wrong_shape() -> None:
    with pytest.raises(DetectionError, match=r"shape \(n, 3\)"):
        RoofDetector()._group_by_orientation(np.array([0.0, 0.0, 0.0]))


def test_group_by_orientation_empty_input_returns_empty_list() -> None:
    assert RoofDetector()._group_by_orientation(np.zeros((0, 3))) == []


def test_group_by_orientation_excludes_zero_norm_normal() -> None:
    """A degenerate zero-length normal is skipped entirely, not assigned to any group."""
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    groups = RoofDetector()._group_by_orientation(normals)

    all_assigned = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
    assert 1 not in all_assigned
    assert 0 in all_assigned
    assert 2 in all_assigned
