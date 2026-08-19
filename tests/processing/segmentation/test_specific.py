"""
Regression suite for topocore.processing.segmentation.specific
(TreeSegmenter, BuildingSegmenter) -- PR19.

Includes a severe, real bug found and fixed in this session: both
segmenters filtered by raw, ABSOLUTE Z, despite their own docstrings
describing `min_height`/`max_height` as "above ground", and
TreeSegmenter's own class docstring explicitly listing "1. Ground
classification to separate trees from ground" as step 1 of its
algorithm -- a step that was never actually implemented anywhere in
the code. Confirmed directly: a realistic point cloud at ~1500m
absolute elevation (ordinary real-world survey/LiDAR data, not
ground-normalized) with a genuine 1-10m tall tree cluster raised
SegmentationError ("No points found above minimum height") with the
class's own documented defaults -- completely unusable on real
elevation data. Fixed by classifying ground first (GroundManager,
already audited and cache-fixed elsewhere in this session) and
filtering on each point's height above its own geometrically nearest
ground point.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.specific import BuildingSegmenter, TreeSegmenter


def _cloud(xs, ys, zs) -> PointCloud:  # type: ignore[no-untyped-def]
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def realistic_elevation_scene_with_tree() -> PointCloud:
    """
    Ground at ~1500m absolute elevation (ordinary real-world data,
    NOT ground-normalized) plus a genuine 1-10m tall tree cluster --
    the exact scenario that exposed the bug.
    """
    rng = np.random.default_rng(0)
    ground_xs = rng.uniform(0, 20, 200)
    ground_ys = rng.uniform(0, 20, 200)
    ground_zs = 1500.0 + rng.uniform(-0.1, 0.1, 200)

    tree_xs = rng.uniform(9.5, 10.5, 80)
    tree_ys = rng.uniform(9.5, 10.5, 80)
    tree_zs = 1500.0 + rng.uniform(1.0, 10.0, 80)

    xs = np.concatenate([ground_xs, tree_xs])
    ys = np.concatenate([ground_ys, tree_ys])
    zs = np.concatenate([ground_zs, tree_zs])
    return _cloud(xs, ys, zs)


# ----------------------------------------------------------------------
# TreeSegmenter -- the real bug.
# ----------------------------------------------------------------------


def test_tree_segmenter_works_at_realistic_absolute_elevation(
    realistic_elevation_scene_with_tree: PointCloud,
) -> None:
    """
    Before the fix: SegmentationError("No points found above minimum
    height") with the class's own documented default parameters.
    """
    result = TreeSegmenter(min_height=0.5, max_height=50.0, eps=1.0, min_samples=5).segment(
        realistic_elevation_scene_with_tree
    )

    assert result.num_segments >= 1
    assert result.segment_sizes.sum() > 0


def test_tree_segmenter_ground_points_excluded() -> None:
    """
    A flat ground-only cloud (no tree) at realistic elevation must
    raise -- there's nothing above the ground to segment as a tree.
    """
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 20, 100)
    ys = rng.uniform(0, 20, 100)
    zs = 1500.0 + rng.uniform(-0.1, 0.1, 100)

    with pytest.raises(SegmentationError):
        TreeSegmenter(min_height=0.5, max_height=50.0).segment(_cloud(xs, ys, zs))


def test_tree_segmenter_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError):
        TreeSegmenter().segment(PointCloud())


def test_tree_segmenter_rejects_invalid_height_range() -> None:
    with pytest.raises(SegmentationError):
        TreeSegmenter(min_height=10.0, max_height=5.0)


# ----------------------------------------------------------------------
# BuildingSegmenter -- shares the same bug/fix via _filter_cloud_by_relative_height.
# ----------------------------------------------------------------------


def test_building_segmenter_works_at_realistic_absolute_elevation() -> None:
    """
    A flat, planar "roof" cluster raised above realistic-elevation
    ground -- before the fix, this would also fail with the class's
    documented default height range (1.0-100.0), since raw elevation
    is ~1500m.

    Ground sampled densely enough that every 1x1 cell GroundManager's
    default grid classifier uses has real ground coverage -- a
    sparser sampling can leave some cells with only roof points,
    which then trivially become each cell's own "ground" reference
    (the same grid-classification density characteristic already
    established for GroundManager itself elsewhere in this session,
    not a new bug in this segmenter).
    """
    rng = np.random.default_rng(0)
    ground_xs = rng.uniform(0, 12, 2000)
    ground_ys = rng.uniform(0, 12, 2000)
    ground_zs = 1500.0 + rng.uniform(-0.1, 0.1, 2000)

    # Flat planar roof, 1.0-unit spacing, 3m above local ground.
    roof_xs, roof_ys, roof_zs = [], [], []
    for i in range(10):
        for j in range(10):
            roof_xs.append(i * 1.0)
            roof_ys.append(j * 1.0)
            roof_zs.append(1503.0)

    xs = np.concatenate([ground_xs, roof_xs])
    ys = np.concatenate([ground_ys, roof_ys])
    zs = np.concatenate([ground_zs, roof_zs])

    result = BuildingSegmenter(
        min_height=1.0,
        max_height=100.0,
        k=8,
        curvature_threshold=0.1,
        min_points_per_building=20,
    ).segment(_cloud(xs, ys, zs))

    assert result.num_segments >= 1


def test_building_segmenter_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError):
        BuildingSegmenter().segment(PointCloud())


def test_building_segmenter_rejects_invalid_height_range() -> None:
    with pytest.raises(SegmentationError):
        BuildingSegmenter(min_height=10.0, max_height=5.0)
