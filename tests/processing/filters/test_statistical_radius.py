"""
Regression suite for topocore.processing.filters.statistical and
.radius -- PR19.

Both verified against a clear, unambiguous outlier scenario (a dense
100-point cluster plus 2 far-away isolated points) -- 100% of the
cluster kept, 100% of the outliers removed, by both filters
independently. No bugs found.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError
from topocore.processing.filters.radius import RadiusOutlierFilter
from topocore.processing.filters.statistical import StatisticalOutlierFilter


@pytest.fixture
def cluster_with_outliers() -> PointCloud:
    rng = np.random.default_rng(0)
    cluster_xs = rng.uniform(0, 1, 100)
    cluster_ys = rng.uniform(0, 1, 100)
    outlier_xs = np.array([50.0, -50.0])
    outlier_ys = np.array([50.0, -50.0])

    xs = np.concatenate([cluster_xs, outlier_xs])
    ys = np.concatenate([cluster_ys, outlier_ys])
    zs = np.zeros_like(xs)

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# StatisticalOutlierFilter
# ----------------------------------------------------------------------


def test_sor_removes_isolated_outliers_keeps_cluster(
    cluster_with_outliers: PointCloud,
) -> None:
    mask = StatisticalOutlierFilter(k=8, std_ratio=1.0).mask(cluster_with_outliers)

    assert mask[:100].all()  # entire cluster kept
    assert not mask[100:].any()  # both outliers removed


def test_sor_min_points_fallback_returns_all_when_strict_is_false() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=5, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.0, 1.0, 2.0, 100.0, -100.0]
    chunk[PointAttribute.Y][:] = [0.0, 0.0, 0.0, 0.0, 0.0]
    chunk[PointAttribute.Z][:] = [0.0, 0.0, 0.0, 0.0, 0.0]
    cloud.add_chunk(chunk)

    # Extremely strict std_ratio could filter below min_points --
    # falls back to keeping everything rather than erroring.
    mask = StatisticalOutlierFilter(k=2, std_ratio=0.001, min_points=5).mask(cloud)
    assert mask.all()


def test_sor_strict_raises_instead_of_fallback() -> None:
    cloud = PointCloud()
    chunk = Chunk(size=5, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.0, 1.0, 2.0, 100.0, -100.0]
    chunk[PointAttribute.Y][:] = [0.0, 0.0, 0.0, 0.0, 0.0]
    chunk[PointAttribute.Z][:] = [0.0, 0.0, 0.0, 0.0, 0.0]
    cloud.add_chunk(chunk)

    with pytest.raises(FilterError):
        StatisticalOutlierFilter(k=2, std_ratio=0.001, min_points=5, strict=True).mask(cloud)


def test_sor_rejects_empty_cloud() -> None:
    with pytest.raises(FilterError):
        StatisticalOutlierFilter().mask(PointCloud())


def test_sor_rejects_nonpositive_std_ratio() -> None:
    with pytest.raises(FilterError):
        StatisticalOutlierFilter(std_ratio=0.0)


# ----------------------------------------------------------------------
# RadiusOutlierFilter
# ----------------------------------------------------------------------


def test_ror_removes_isolated_outliers_keeps_cluster(
    cluster_with_outliers: PointCloud,
) -> None:
    mask = RadiusOutlierFilter(radius=0.5, min_neighbors=3).mask(cluster_with_outliers)

    assert mask[:100].all()
    assert not mask[100:].any()


def test_ror_include_self_affects_neighbor_count() -> None:
    # A single isolated point: with include_self=True, it counts
    # itself (1 neighbor); with include_self=False, 0 neighbors.
    cloud = PointCloud()
    chunk = Chunk(size=1, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.0]
    chunk[PointAttribute.Y][:] = [0.0]
    chunk[PointAttribute.Z][:] = [0.0]
    cloud.add_chunk(chunk)

    mask_with_self = RadiusOutlierFilter(radius=1.0, min_neighbors=1, include_self=True, min_points=0).mask(cloud)
    mask_without_self = RadiusOutlierFilter(radius=1.0, min_neighbors=1, include_self=False, min_points=0).mask(cloud)

    assert mask_with_self[0]
    assert not mask_without_self[0]


def test_ror_rejects_empty_cloud() -> None:
    with pytest.raises(FilterError):
        RadiusOutlierFilter().mask(PointCloud())


def test_ror_rejects_nonpositive_radius() -> None:
    with pytest.raises(FilterError):
        RadiusOutlierFilter(radius=0.0)
