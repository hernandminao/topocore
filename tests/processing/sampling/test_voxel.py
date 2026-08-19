"""
Regression suite for topocore.processing.sampling.voxel -- PR19.

Verified against known geometry: exact voxel-index boundaries
(including negative coordinates), exact centroid, and closest-to-
center selection. No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.voxel import VoxelSampler, _voxel_indices


def _cloud(xs: list[float], ys: list[float], zs: list[float]) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def test_voxel_indices_at_exact_boundaries_and_negative_coordinates() -> None:
    x = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    i, _, _ = _voxel_indices(x, np.zeros_like(x), np.zeros_like(x), voxel_size=1.0)
    np.testing.assert_array_equal(i, [-2, -1, -1, 0, 0, 1, 1])


def test_voxel_indices_rejects_nonpositive_size() -> None:
    x = np.array([0.0])
    with pytest.raises(SamplingError):
        _voxel_indices(x, x, x, voxel_size=0.0)


def test_centroid_method_gives_exact_mean() -> None:
    cloud = _cloud([0.1, 0.2, 0.9], [0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    result = VoxelSampler(voxel_size=10.0, method="centroid").sample(cloud)
    point = next(iter(result))

    assert point[PointAttribute.X][0] == pytest.approx(np.mean([0.1, 0.2, 0.9]))
    assert point[PointAttribute.Y][0] == pytest.approx(np.mean([0.1, 0.5, 0.9]))
    assert point[PointAttribute.Z][0] == pytest.approx(np.mean([0.1, 0.5, 0.9]))


def test_closest_method_picks_point_nearest_voxel_center() -> None:
    cloud = _cloud([0.1, 0.2, 0.9], [0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    result = VoxelSampler(voxel_size=10.0, method="closest").sample(cloud)
    point = next(iter(result))

    # Voxel center is (5,5,5); (0.9,0.9,0.9) is closest among the 3.
    assert point[PointAttribute.X][0] == pytest.approx(0.9)


def test_all_method_returns_cloud_unchanged() -> None:
    cloud = _cloud([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    result = VoxelSampler(voxel_size=10.0, method="all").sample(cloud)
    assert result.point_count == 3


def test_multiple_voxels_produce_multiple_output_points() -> None:
    # Two points far apart -> 2 separate voxels.
    cloud = _cloud([0.0, 100.0], [0.0, 0.0], [0.0, 0.0])
    result = VoxelSampler(voxel_size=1.0, method="centroid").sample(cloud)
    assert result.point_count == 2


def test_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError):
        VoxelSampler(voxel_size=1.0).sample(PointCloud())


def test_rejects_nonpositive_voxel_size() -> None:
    with pytest.raises(SamplingError):
        VoxelSampler(voxel_size=0.0)


def test_rejects_unsupported_method() -> None:
    with pytest.raises(SamplingError):
        VoxelSampler(voxel_size=1.0, method="bogus")  # type: ignore[arg-type]
