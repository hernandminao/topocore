"""
Regression suite for SEG-RG-001 (fixed in this PR).

Bug: _filter_small_regions() marked regions smaller than
min_region_size as noise (-1) WITHOUT renumbering the surviving
regions to remain contiguous from 0. When the first-discovered
region (lowest curvature, assigned id 0) was filtered out as too
small but a later region (id 1) survived, the final labels contained
{-1, 1} -- not contiguous -- and SegmentationResult's own
__post_init__ validation correctly rejected this with
ValueError("Segment IDs must be contiguous...").

Fix: after marking small regions as noise, surviving region ids are
renumbered to their rank among survivors (0, 1, 2, ...), preserving
relative order. Confirmed this also resolves the same failure mode
when triggered indirectly through BuildingSegmenter/TreeSegmenter,
which use RegionGrowingSegmenter internally.
"""

from __future__ import annotations

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.segmentation.region_growing import RegionGrowingSegmenter


def test_small_region_discovered_first_no_longer_breaks_contiguity() -> None:
    """
    Exact reproduction of the original failure: a small, perfectly
    flat region (curvature ~0, discovered first, assigned id 0) is
    filtered out as too small, while a large region (discovered
    second, would have been id 1) survives. Labels must end up as
    {-1, 0} (renumbered), not {-1, 1}.
    """
    rng = np.random.default_rng(0)
    big_n, small_n = 50, 3
    cloud = PointCloud()
    chunk = Chunk(
        size=big_n + small_n,
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    sx = np.linspace(200, 200.1, small_n)
    sy = np.linspace(200, 200.1, small_n)
    sz = np.full(small_n, 100.0)
    bx = rng.uniform(0, 10, big_n)
    by = rng.uniform(0, 10, big_n)
    bz = rng.normal(0, 0.01, big_n)
    chunk[PointAttribute.X][:] = np.concatenate([sx, bx])
    chunk[PointAttribute.Y][:] = np.concatenate([sy, by])
    chunk[PointAttribute.Z][:] = np.concatenate([sz, bz])
    cloud.add_chunk(chunk)

    result = RegionGrowingSegmenter(
        k=5,
        curvature_threshold=0.5,
        normal_angle_threshold=30.0,
        min_region_size=10,
    ).segment(cloud)

    assert result.num_segments == 1
    np.testing.assert_array_equal(np.unique(result.labels), [-1, 0])


def test_multiple_small_regions_filtered_leave_contiguous_survivors() -> None:
    """Three regions: two small (discovered first and second) filtered out, one large survives -- must renumber to 0."""
    rng = np.random.default_rng(1)
    tiny_a_n, tiny_b_n, big_n = 2, 2, 50

    tiny_a_x = np.linspace(500, 500.05, tiny_a_n)
    tiny_a_y = np.linspace(500, 500.05, tiny_a_n)
    tiny_a_z = np.full(tiny_a_n, 500.0)

    tiny_b_x = np.linspace(600, 600.05, tiny_b_n)
    tiny_b_y = np.linspace(600, 600.05, tiny_b_n)
    tiny_b_z = np.full(tiny_b_n, 600.0)

    big_x = rng.uniform(0, 10, big_n)
    big_y = rng.uniform(0, 10, big_n)
    big_z = rng.normal(0, 0.01, big_n)

    cloud = PointCloud()
    chunk = Chunk(
        size=tiny_a_n + tiny_b_n + big_n,
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = np.concatenate([tiny_a_x, tiny_b_x, big_x])
    chunk[PointAttribute.Y][:] = np.concatenate([tiny_a_y, tiny_b_y, big_y])
    chunk[PointAttribute.Z][:] = np.concatenate([tiny_a_z, tiny_b_z, big_z])
    cloud.add_chunk(chunk)

    result = RegionGrowingSegmenter(
        k=5,
        curvature_threshold=0.5,
        normal_angle_threshold=30.0,
        min_region_size=10,
    ).segment(cloud)

    assert result.num_segments == 1
    np.testing.assert_array_equal(np.unique(result.labels), [-1, 0])
