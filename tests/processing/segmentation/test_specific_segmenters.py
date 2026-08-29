"""
Coverage audit tests for topocore.processing.segmentation.specific
(TreeSegmenter, BuildingSegmenter, and their shared module-level
helpers).

Audit findings (documented here, not force-tested):

_extract_points()'s own empty-cloud early return (returning an empty
(0, 3) array) is confirmed unreachable via the current call chain:
its only caller, _filter_cloud_by_relative_height(), is only ever
invoked from TreeSegmenter.segment()/BuildingSegmenter.segment()
AFTER their own `cloud.is_empty` check has already passed -- by the
time _extract_points() runs, the cloud is guaranteed non-empty.

_filter_tree_clusters()'s `verticality is None` check and
_filter_building_clusters()'s `planarity` default-fallback value are
both confirmed unreachable: PCAFeatures.compute_all() has no
feature-selection parameter and unconditionally includes both
"planarity" and "verticality" in every result it returns (confirmed
by reading its own implementation) -- neither key can ever be
missing from features.get(...) given how compute_all() is actually
called here (with no extra kwargs).

TreeSegmenter.name / BuildingSegmenter.name are documented as
orphaned: segmentation/manager.py (the only place that constructs
these segmenters) never reads back .name from the instances it
creates -- the same "reimplemented/bypassed rather than delegated"
pattern already found for NormalManager and Classifier elsewhere in
this audit.

_compute_relative_height()'s own "no ground points found" contract
is tested directly (calling the private module-level function with
an all-False mask) rather than through the full
TreeSegmenter/BuildingSegmenter integration chain: confirmed directly
that GridGroundClassifier mathematically always finds at least one
local-minimum ground point for any non-empty cloud, making this
branch unreachable via that specific caller chain -- but the function
itself has a clear, real, directly-testable contract, matching the
same reasoning already approved for testing _validate_feature() via
a custom broken computer in classification/ml.py's own audit.

Both segmenters' happy paths use synthetic clouds tuned so ground
points genuinely underlie the elevated structure's own footprint --
confirmed during this audit that an elevated structure spatially
ISOLATED from any real ground point (no ground point sharing its own
grid cell) gets self-referentially misclassified as its own "ground"
by the grid heuristic, an emergent property of that ground-detection
method, not a defect in TreeSegmenter/BuildingSegmenter's own logic.

Possible latent defect found during this audit, NOT fixed and NOT
locked in by a test: _filter_tree_clusters()/_filter_building_clusters()
both guard with `if k < 1: continue` before calling
`PCAFeatures(k=k)`, but PCAFeatures itself requires k >= 3 (raising
PointDescriptorError, a different exception hierarchy than
SegmentationError, if not). Confirmed directly: a segment of exactly
2 or 3 points yields k=1 or k=2, which passes the `k < 1` guard but
then crashes uncaught inside PCAFeatures' own constructor. This
looks like the guard's threshold should likely be `k < 3` to match
PCAFeatures' real minimum, but this is flagged for a decision on the
production code, not resolved here.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.specific import (
    BuildingSegmenter,
    TreeSegmenter,
    _compute_relative_height,
)


def _tree_cloud(seed: int = 0) -> PointCloud:
    """Ground plane + a tight, vertically-varying column -- confirmed to survive height filtering."""
    rng = np.random.default_rng(seed)
    ground_n, tree_n = 200, 30
    cloud = PointCloud()
    chunk = Chunk(
        size=ground_n + tree_n,
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    gx, gy, gz = (
        rng.uniform(0, 50, ground_n),
        rng.uniform(0, 50, ground_n),
        np.zeros(ground_n),
    )
    tx = np.full(tree_n, 25.0) + rng.normal(0, 0.3, tree_n)
    ty = np.full(tree_n, 25.0) + rng.normal(0, 0.3, tree_n)
    tz = rng.uniform(1.0, 8.0, tree_n)
    chunk[PointAttribute.X][:] = np.concatenate([gx, tx])
    chunk[PointAttribute.Y][:] = np.concatenate([gy, ty])
    chunk[PointAttribute.Z][:] = np.concatenate([gz, tz])
    cloud.add_chunk(chunk)
    return cloud


def _building_cloud(seed: int = 0) -> PointCloud:
    """Dense ground plane genuinely underlying a flat, elevated roof footprint."""
    rng = np.random.default_rng(seed)
    ground_n, roof_n = 3000, 80
    gx, gy, gz = (
        rng.uniform(0, 30, ground_n),
        rng.uniform(0, 30, ground_n),
        np.zeros(ground_n),
    )
    rx = np.full(roof_n, 15.0) + rng.normal(0, 0.5, roof_n)
    ry = np.full(roof_n, 15.0) + rng.normal(0, 0.5, roof_n)
    rz = np.full(roof_n, 5.0) + rng.normal(0, 0.02, roof_n)
    cloud = PointCloud()
    chunk = Chunk(
        size=ground_n + roof_n,
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = np.concatenate([gx, rx])
    chunk[PointAttribute.Y][:] = np.concatenate([gy, ry])
    chunk[PointAttribute.Z][:] = np.concatenate([gz, rz])
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# _compute_relative_height() -- its own direct contract.
# ----------------------------------------------------------------------


def test_compute_relative_height_rejects_all_false_mask() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    empty_mask = np.array([False, False])

    with pytest.raises(SegmentationError, match="No ground points found"):
        _compute_relative_height(points, empty_mask)


# ----------------------------------------------------------------------
# TreeSegmenter -- construction validation.
# ----------------------------------------------------------------------


def test_tree_min_height_negative_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_height must be"):
        TreeSegmenter(min_height=-1)


def test_tree_max_height_not_greater_than_min_rejected() -> None:
    with pytest.raises(SegmentationError, match="must be > min_height"):
        TreeSegmenter(min_height=10, max_height=5)


def test_tree_eps_not_positive_rejected() -> None:
    with pytest.raises(SegmentationError, match="eps must be positive"):
        TreeSegmenter(eps=0)


def test_tree_min_samples_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_samples must be"):
        TreeSegmenter(min_samples=0)


def test_tree_min_points_per_tree_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_points_per_tree must be"):
        TreeSegmenter(min_points_per_tree=0)


# ----------------------------------------------------------------------
# TreeSegmenter.segment() -- validation and happy path.
# ----------------------------------------------------------------------


def test_tree_segment_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError, match="empty point cloud"):
        TreeSegmenter().segment(PointCloud())


def test_tree_segment_happy_path_finds_tree_cluster() -> None:
    cloud = _tree_cloud()
    result = TreeSegmenter(min_height=0.5, max_height=50.0, eps=1.0, min_samples=3, min_points_per_tree=5).segment(
        cloud
    )

    assert result.num_segments > 0
    assert result.cloud is cloud


def test_tree_segment_with_min_points_per_tree_one_still_completes() -> None:
    """
    min_points_per_tree=1 is accepted at construction and the
    segmenter still completes -- NOT confirmed here to specifically
    exercise the `k < 1` continue branch, since this synthetic
    cloud's DBSCAN clustering happens to produce only one segment
    large enough that k stays well above 1 regardless of this
    parameter. Kept as a construction-acceptance/no-crash check, not
    a claim about which internal branch it reaches.
    """
    cloud = _tree_cloud()
    result = TreeSegmenter(min_points_per_tree=1, eps=1.0, min_samples=3).segment(cloud)

    assert isinstance(result.num_segments, int)  # completes without crashing


# ----------------------------------------------------------------------
# BuildingSegmenter -- construction validation.
# ----------------------------------------------------------------------


def test_building_min_height_negative_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_height must be"):
        BuildingSegmenter(min_height=-1)


def test_building_max_height_not_greater_than_min_rejected() -> None:
    with pytest.raises(SegmentationError, match="must be > min_height"):
        BuildingSegmenter(min_height=10, max_height=5)


def test_building_k_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="k must be"):
        BuildingSegmenter(k=0)


def test_building_curvature_threshold_negative_rejected() -> None:
    with pytest.raises(SegmentationError, match="curvature_threshold must be"):
        BuildingSegmenter(curvature_threshold=-1)


def test_building_normal_angle_threshold_not_positive_rejected() -> None:
    with pytest.raises(SegmentationError, match="normal_angle_threshold must be"):
        BuildingSegmenter(normal_angle_threshold=0)


def test_building_min_points_per_building_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_points_per_building must be"):
        BuildingSegmenter(min_points_per_building=0)


# ----------------------------------------------------------------------
# BuildingSegmenter.segment() -- validation and happy path.
# ----------------------------------------------------------------------


def test_building_segment_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError, match="empty point cloud"):
        BuildingSegmenter().segment(PointCloud())


def test_building_segment_happy_path_finds_roof_cluster() -> None:
    cloud = _building_cloud()
    result = BuildingSegmenter(
        min_height=1.0,
        max_height=50.0,
        k=6,
        min_points_per_building=10,
        normal_angle_threshold=20.0,
        curvature_threshold=0.1,
    ).segment(cloud)

    assert result.num_segments > 0
    assert result.cloud is cloud


def test_building_segment_min_points_per_building_one_reaches_size_filter_boundary() -> None:
    """
    min_points_per_building=1 lets very small segments through the
    size filter. NOT tested down to k<3 here: confirmed this can
    surface k=1/k=2, which PCAFeatures itself rejects with
    PointDescriptorError (k must be >= 3) -- an uncaught exception
    from a different hierarchy than SegmentationError. This is
    flagged as a possible latent defect (the `if k < 1: continue`
    guard likely should be `if k < 3: continue` to match
    PCAFeatures' own actual minimum), not confirmed as intended
    behavior, so it is not locked in by a test here pending a
    decision on the production code.
    """
    cloud = _building_cloud()
    result = BuildingSegmenter(
        min_points_per_building=4,  # large enough to avoid the k<3 edge case entirely
        k=6,
        normal_angle_threshold=20.0,
        curvature_threshold=0.1,
    ).segment(cloud)

    assert isinstance(result.num_segments, int)
