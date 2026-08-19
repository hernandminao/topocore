"""
Regression suite for topocore.processing.filters.manager.FilterManager
-- PR19.

Its cache key already correctly derives id(current) fresh inside the
apply() loop (never stored/frozen on the manager) -- the pattern used
as the reference fix for topocore.processing.features.manager.
FeatureManager and topocore.processing.normals.manager.NormalManager
elsewhere in this session. No bugs found here.
"""

from __future__ import annotations

import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError
from topocore.processing.filters.manager import FilterManager
from topocore.processing.filters.pass_through import Axis, PassThroughFilter


def _cloud(zs: list[float]) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=len(zs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [float(i) for i in range(len(zs))]
    chunk[PointAttribute.Y][:] = [0.0] * len(zs)
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def test_chains_filters_in_order() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=100.0)
    manager.add_pass_through(Axis.Z, min_value=10.0, max_value=100.0)

    cloud = _cloud([5.0, 15.0, 50.0, -5.0])
    result = manager.apply(cloud)

    # First filter keeps [5,15,50] (drops -5); second keeps [15,50]
    # (drops 5) -- chained, not independent.
    assert result.point_count == 2


def test_empty_filter_list_returns_cloud_unchanged() -> None:
    manager = FilterManager()
    cloud = _cloud([1.0, 2.0])
    assert manager.apply(cloud) is cloud


def test_different_clouds_do_not_collide_in_cache() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=10.0)

    cloud_a = _cloud([5.0, 50.0])
    cloud_b = _cloud([100.0, 200.0])  # both would be dropped by the filter

    result_a = manager.apply(cloud_a)
    result_b = manager.apply(cloud_b)

    assert result_a.point_count == 1  # only 5.0 survives
    assert result_b.point_count == 0  # both dropped


def test_statistics_tracked_by_default() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=10.0)

    manager.apply(_cloud([5.0, 50.0]))
    stats = manager.statistics()

    assert stats is not None
    assert len(stats) == 1
    assert stats[0]["points_before"] == 2
    assert stats[0]["points_after"] == 1


def test_statistics_none_when_disabled() -> None:
    manager = FilterManager(track_stats=False)
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=10.0)
    manager.apply(_cloud([5.0]))
    assert manager.statistics() is None


def test_apply_masks_returns_one_mask_per_filter() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=100.0)
    manager.add_pass_through(Axis.Z, min_value=10.0, max_value=100.0)

    masks = manager.apply_masks(_cloud([5.0, 15.0, 50.0]))
    assert len(masks) == 2


def test_clear_resets_filters_cache_and_stats() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=10.0)
    manager.apply(_cloud([5.0]))

    manager.clear()

    assert manager.filter_count == 0
    assert manager.statistics() is None


def test_method_chaining_returns_self() -> None:
    manager = FilterManager()
    result = manager.add_pass_through(Axis.Z, min_value=0.0, max_value=10.0)
    assert result is manager


def test_filter_failure_wrapped_in_filter_error() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0))

    with pytest.raises(FilterError):
        manager.apply(PointCloud())  # empty cloud -> PassThroughFilter raises internally


def test_len_and_iteration() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.X, min_value=0.0, max_value=1.0)
    manager.add_pass_through(Axis.Y, min_value=0.0, max_value=1.0)

    assert len(manager) == 2
    assert len(list(manager)) == 2
    assert manager[0].name().startswith("pass_through")
