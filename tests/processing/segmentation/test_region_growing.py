"""
Regression suite for topocore.processing.segmentation.region_growing
-- PR19.

Includes two real bugs found and fixed in this session:

1. The neighbor search radius for growth was ``self._k * 0.1`` --
   multiplying a neighbor COUNT (meant for normal estimation) by an
   arbitrary constant, treating it as a spatial DISTANCE. Confirmed:
   a perfectly flat plane with realistic 5.0-unit point spacing (a
   trivial case that should form exactly one region, curvature=0
   everywhere) produced 0 segments -- every point was its own
   isolated "region" of 1, discarded by min_region_size. Fixed by
   deriving the radius from the cloud's actual measured point
   density (mean k-NN distance), mirroring the technique already
   used in DBSCANSegmenter/ConnectedComponentsSegmenter.

2. Normal consistency compared ``dot(n1, n2)`` directly, not
   ``abs(dot(n1, n2))``. For a perfectly VERTICAL surface, normal Z
   is exactly 0, so `orient_upward`'s flip-if-negative-Z rule never
   triggers -- each point's independent local PCA can arbitrarily
   report either sign. Confirmed: a single flat vertical wall's
   normals split ~30/26 between (0,-1,0) and (0,1,0), causing region
   growing to see a spurious ~180-degree angle between genuinely
   coplanar adjacent points and split one physical wall into two
   regions. Fixed locally (not in the shared normals module, which
   other consumers may depend on for absolute direction) by using
   ``abs(dot(...))`` -- antiparallel normals now count as consistent,
   while genuinely perpendicular surfaces (e.g. a real floor/wall
   corner) still correctly separate.
"""

from __future__ import annotations

import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.region_growing import RegionGrowingSegmenter


def _cloud(xs, ys, zs) -> PointCloud:  # type: ignore[no-untyped-def]
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def _flat_plane(spacing: float = 5.0) -> PointCloud:
    xs, ys, zs = [], [], []
    for i in range(10):
        for j in range(10):
            xs.append(i * spacing)
            ys.append(j * spacing)
            zs.append(0.0)
    return _cloud(xs, ys, zs)


def _vertical_wall(spacing: float = 5.0) -> PointCloud:
    xs, ys, zs = [], [], []
    for i in range(8):
        for k in range(1, 8):
            xs.append(i * spacing)
            ys.append(0.0)
            zs.append(k * spacing)
    return _cloud(xs, ys, zs)


# ----------------------------------------------------------------------
# Bug 1: growth radius derived from k (a count), not real density.
# ----------------------------------------------------------------------


def test_flat_plane_at_realistic_scale_forms_one_region() -> None:
    """
    The exact reproduction: before the fix, this gave 0 segments
    (every point isolated, discarded by min_region_size).
    """
    seg = RegionGrowingSegmenter(k=10, curvature_threshold=0.1, normal_angle_threshold=30.0, min_region_size=5)
    result = seg.segment(_flat_plane(spacing=5.0))

    assert result.num_segments == 1
    assert (result.labels == 0).all()
    assert result.segment_sizes[0] == 100


def test_flat_plane_at_different_scale_also_works() -> None:
    """
    Confirms the fix is genuinely scale-derived, not just tuned to
    one specific spacing value.
    """
    seg = RegionGrowingSegmenter(k=10, curvature_threshold=0.1, normal_angle_threshold=30.0, min_region_size=5)
    result = seg.segment(_flat_plane(spacing=0.02))  # much finer spacing

    assert result.num_segments == 1
    assert result.segment_sizes[0] == 100


# ----------------------------------------------------------------------
# Bug 2: normal sign ambiguity on vertical surfaces.
# ----------------------------------------------------------------------


def test_vertical_wall_alone_forms_one_region() -> None:
    """
    The exact reproduction: before the fix, a single flat vertical
    wall split into 2 regions along a normal-sign boundary.
    """
    seg = RegionGrowingSegmenter(k=8, curvature_threshold=0.1, normal_angle_threshold=20.0, min_region_size=5)
    result = seg.segment(_vertical_wall())

    assert result.num_segments == 1
    assert result.segment_sizes[0] == 56


def test_perpendicular_surfaces_still_separate_correctly() -> None:
    """
    Confirms the abs(dot(...)) fix didn't break genuine perpendicular
    separation -- a real floor/wall corner must still yield 2
    distinct regions (not merge into 1), even though some points
    exactly at the shared edge may be reasonably excluded as
    ambiguous (their local neighborhood spans both surfaces).
    """
    xs, ys, zs = [], [], []
    for i in range(8):
        for j in range(8):
            xs.append(i * 5.0)
            ys.append(j * 5.0)
            zs.append(0.0)
    for i in range(8):
        for k in range(1, 8):
            xs.append(i * 5.0)
            ys.append(0.0)
            zs.append(k * 5.0)

    seg = RegionGrowingSegmenter(k=8, curvature_threshold=0.1, normal_angle_threshold=20.0, min_region_size=5)
    result = seg.segment(_cloud(xs, ys, zs))

    assert result.num_segments == 2  # floor and wall, not merged into 1, not split into 3+


# ----------------------------------------------------------------------
# Validation, unaffected by the fixes.
# ----------------------------------------------------------------------


def test_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError):
        RegionGrowingSegmenter().segment(PointCloud())


def test_rejects_k_below_three() -> None:
    with pytest.raises(SegmentationError):
        RegionGrowingSegmenter(k=2)


def test_rejects_normal_angle_threshold_out_of_range() -> None:
    with pytest.raises(SegmentationError):
        RegionGrowingSegmenter(normal_angle_threshold=91.0)


def test_rejects_max_region_size_below_min() -> None:
    with pytest.raises(SegmentationError):
        RegionGrowingSegmenter(min_region_size=100, max_region_size=10)
