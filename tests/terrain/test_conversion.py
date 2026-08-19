"""
Regression suite for topocore.terrain.conversion.pointcloud_to_points
-- PR19.
"""

from __future__ import annotations

import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.terrain.conversion import pointcloud_to_points
from topocore.terrain.exceptions import ConversionError


def _cloud_with(xs: list[float], ys: list[float], zs: list[float]) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def test_converts_single_chunk_preserving_values() -> None:
    cloud = _cloud_with([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    points = pointcloud_to_points(cloud)

    assert len(points) == 2
    assert points[0].x == 1.0
    assert points[0].y == 3.0
    assert points[0].z == 5.0
    assert points[1].x == 2.0


def test_preserves_chunk_order_then_within_chunk_order() -> None:
    cloud = PointCloud()
    chunk1 = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk1[PointAttribute.X][:] = [1.0, 2.0]
    chunk1[PointAttribute.Y][:] = [0.0, 0.0]
    chunk1[PointAttribute.Z][:] = [0.0, 0.0]
    chunk2 = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk2[PointAttribute.X][:] = [3.0, 4.0]
    chunk2[PointAttribute.Y][:] = [0.0, 0.0]
    chunk2[PointAttribute.Z][:] = [0.0, 0.0]
    cloud.add_chunk(chunk1)
    cloud.add_chunk(chunk2)

    points = pointcloud_to_points(cloud)
    assert [p.x for p in points] == [1.0, 2.0, 3.0, 4.0]


def test_rejects_empty_pointcloud() -> None:
    with pytest.raises(ConversionError):
        pointcloud_to_points(PointCloud())
