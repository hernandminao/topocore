"""
Regression suite for PointCloud.version -- PR21.3.1.

version is a mutation counter incremented by exactly the 3 methods
that change which points a PointCloud holds (add_chunk, remove_chunk,
clear), introduced to close a real, demonstrated cache-invalidation
bug in NormalManager (see test_normal_manager_version_invalidation.py):
id(cloud) alone cannot detect a PointCloud mutated in place, since
Python object identity never changes when an object is mutated
rather than replaced.
"""

from __future__ import annotations

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


def _make_chunk(n: int = 3) -> Chunk:
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Y][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    return chunk


def test_version_starts_at_zero() -> None:
    assert PointCloud().version == 0


def test_add_chunk_increments_version() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_make_chunk())
    assert cloud.version == 1
    cloud.add_chunk(_make_chunk())
    assert cloud.version == 2


def test_remove_chunk_increments_version() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_make_chunk())
    version_after_add = cloud.version
    cloud.remove_chunk(0)
    assert cloud.version == version_after_add + 1


def test_clear_increments_version() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_make_chunk())
    version_before_clear = cloud.version
    cloud.clear()
    assert cloud.version == version_before_clear + 1


def test_update_bounds_does_not_change_version() -> None:
    """update_bounds() derives a summary from existing points -- it doesn't change which points the cloud holds."""
    cloud = PointCloud()
    cloud.add_chunk(_make_chunk())
    version_before = cloud.version
    cloud.update_bounds()
    assert cloud.version == version_before


def test_crs_setter_does_not_change_version() -> None:
    """CRS is a label on the coordinate system, not a change to the point geometry itself."""
    cloud = PointCloud()
    cloud.add_chunk(_make_chunk())
    version_before = cloud.version
    cloud.crs = "EPSG:32618"
    assert cloud.version == version_before


def test_clone_starts_at_version_zero() -> None:
    """A clone is a genuinely new, independent object (own id()) -- it doesn't need to inherit the source's version."""
    cloud = PointCloud()
    cloud.add_chunk(_make_chunk())
    cloud.add_chunk(_make_chunk())
    assert cloud.version == 2

    cloned = cloud.clone()
    assert cloned.version == 0
