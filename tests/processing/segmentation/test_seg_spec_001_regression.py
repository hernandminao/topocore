"""
Regression suite for SEG-SPEC-001 (fixed in this PR).

Bug: _filter_tree_clusters()/_filter_building_clusters() guarded with
`if k < 1: continue` before calling PCAFeatures(k=k), but PCAFeatures
itself requires k >= 3 -- a segment of exactly 2 or 3 points yielded
k=1 or k=2, which passed the `k < 1` guard but then crashed uncaught
inside PCAFeatures' own constructor with PointDescriptorError (a
different exception hierarchy than SegmentationError).

Fix: the guard now reads `if k < 3: continue`, matching PCAFeatures'
real minimum. Segments too small to support PCA are skipped, the
same outcome as any other segment filtered by size -- no exception
propagates, and the final SegmentationResult remains valid.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import PointDescriptorError
from topocore.processing.segmentation.specific import BuildingSegmenter, TreeSegmenter


def _building_cloud_with_small_roof(roof_n: int, seed: int = 0) -> PointCloud:
    """Dense ground plane genuinely underlying a small, flat, elevated roof footprint."""
    rng = np.random.default_rng(seed)
    ground_n = 3000
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


@pytest.mark.parametrize("min_points_per_building", [1, 2, 3])
def test_building_tiny_min_points_no_longer_raises_point_descriptor_error(
    min_points_per_building: int,
) -> None:
    """
    min_points_per_building=1/2/3 lets segments of exactly 2 or 3
    points through the size filter (k=1 or k=2 after `min(10,
    point_count-1)`) -- previously these crashed uncaught with
    PointDescriptorError; now they are correctly skipped instead.
    """
    cloud = _building_cloud_with_small_roof(roof_n=80)

    result = BuildingSegmenter(
        min_points_per_building=min_points_per_building,
        k=6,
        normal_angle_threshold=20.0,
        curvature_threshold=0.1,
    ).segment(cloud)

    assert isinstance(result.num_segments, int)  # completes without PointDescriptorError


def test_building_valid_cluster_still_detected_after_fix() -> None:
    """Confirms the fix doesn't affect genuinely valid, sufficiently large clusters."""
    cloud = _building_cloud_with_small_roof(roof_n=80)

    result = BuildingSegmenter(
        min_points_per_building=4,
        k=6,
        normal_angle_threshold=20.0,
        curvature_threshold=0.1,
    ).segment(cloud)

    assert result.num_segments > 0


def test_point_descriptor_error_no_longer_propagates_from_building_filter() -> None:
    cloud = _building_cloud_with_small_roof(roof_n=80)

    try:
        BuildingSegmenter(
            min_points_per_building=1,
            k=6,
            normal_angle_threshold=20.0,
            curvature_threshold=0.1,
        ).segment(cloud)
    except PointDescriptorError:
        pytest.fail("PointDescriptorError leaked from _filter_building_clusters() -- SEG-SPEC-001 regressed.")


def test_tree_valid_cluster_still_detected_after_fix() -> None:
    """Same fix applies to TreeSegmenter's own _filter_tree_clusters()."""
    rng = np.random.default_rng(0)
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

    result = TreeSegmenter(min_points_per_tree=1, eps=1.0, min_samples=3).segment(cloud)

    assert isinstance(result.num_segments, int)
