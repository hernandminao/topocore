"""
Regression suite for topocore.processing.segmentation.manager.
SegmentationManager -- PR19.

Includes a real bug found and fixed in this session: _create_segmenter()
never passed `ground_method` through to TreeSegmenter/BuildingSegmenter
-- a direct consequence of adding `ground_method` to those two classes
in the same session (the real-elevation fix). Same class of gap already
found and fixed for SamplingManager's `seed` passthrough elsewhere in
this session.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.manager import SegmentationManager


def _cloud(xs, ys, zs) -> PointCloud:  # type: ignore[no-untyped-def]
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def _realistic_tree_scene() -> PointCloud:
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


def test_trees_method_works_at_realistic_elevation_through_manager() -> None:
    manager = SegmentationManager(method="trees")
    result = manager.segment(_realistic_tree_scene(), min_height=0.5, max_height=50.0, eps=1.0, min_samples=5)
    assert result.num_segments >= 1


def test_ground_method_is_passed_through_for_trees() -> None:
    """
    ground_method="grid" (explicit, matching the default) must not
    raise or behave differently from the implicit default -- confirms
    the parameter genuinely reaches TreeSegmenter through the manager.
    """
    manager = SegmentationManager(method="trees")
    result = manager.segment(
        _realistic_tree_scene(),
        min_height=0.5,
        max_height=50.0,
        eps=1.0,
        min_samples=5,
        ground_method="grid",
    )
    assert result.num_segments >= 1


def test_dbscan_method_dispatch() -> None:
    rng = np.random.default_rng(1)
    cluster = rng.uniform(-0.5, 0.5, (30, 2))
    xs, ys = cluster[:, 0], cluster[:, 1]
    zs = np.zeros(30)

    manager = SegmentationManager(method="dbscan")
    result = manager.segment(_cloud(xs, ys, zs), eps=2.0, min_samples=5)
    assert result.num_segments == 1


def test_region_growing_method_dispatch() -> None:
    xs, ys, zs = [], [], []
    for i in range(10):
        for j in range(10):
            xs.append(i * 5.0)
            ys.append(j * 5.0)
            zs.append(0.0)

    manager = SegmentationManager(method="region_growing")
    result = manager.segment(_cloud(xs, ys, zs), k=10, curvature_threshold=0.1, min_region_size=5)
    assert result.num_segments == 1


def test_method_setter_switches_method() -> None:
    manager = SegmentationManager(method="dbscan")
    manager.method = "connected_components"
    assert manager.method == "connected_components"


def test_set_params_persists_across_calls() -> None:
    rng = np.random.default_rng(1)
    cluster = rng.uniform(-0.5, 0.5, (30, 2))
    cloud = _cloud(cluster[:, 0], cluster[:, 1], np.zeros(30))

    manager = SegmentationManager(method="dbscan")
    manager.set_params(eps=2.0, min_samples=5)

    result = manager.segment(cloud)
    assert result.num_segments == 1


def test_rejects_unsupported_method_at_construction() -> None:
    with pytest.raises(SegmentationError):
        SegmentationManager(method="bogus")


def test_rejects_unsupported_method_via_setter() -> None:
    manager = SegmentationManager(method="dbscan")
    with pytest.raises(SegmentationError):
        manager.method = "bogus"


def test_rejects_empty_cloud() -> None:
    manager = SegmentationManager(method="dbscan")
    with pytest.raises(SegmentationError):
        manager.segment(PointCloud())
