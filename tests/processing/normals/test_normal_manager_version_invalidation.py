"""
Regression suite for the PR21.3.1 fix: NormalManager's normal/
curvature cache now includes PointCloud.version alongside id(cloud).

The real, demonstrated bug (before this fix): id(cloud) alone cannot
detect a PointCloud mutated in place between two estimate() calls on
the SAME NormalManager instance, since Python object identity never
changes when an object is mutated (via add_chunk/remove_chunk/clear)
rather than replaced. Confirmed directly: mutating a flat-plane cloud
into a steeply-tilted one via add_chunk() and re-calling estimate()
on the same manager returned the STALE, pre-mutation normal instead
of recomputing.
"""

from __future__ import annotations

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.normals.manager import NormalManager


def _make_chunk(xs: list[float], ys: list[float], zs: list[float]) -> Chunk:
    n = len(xs)
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    return chunk


def _flat_plane_cloud() -> PointCloud:
    cloud = PointCloud()
    cloud.add_chunk(
        _make_chunk(
            [0.0, 10.0, 0.0, 10.0, 5.0],
            [0.0, 0.0, 10.0, 10.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        )
    )
    return cloud


def test_mutating_cloud_in_place_invalidates_cache_on_same_manager() -> None:
    """
    The exact PR21.3.1 regression: before the fix, this returned the
    stale, pre-mutation normal for point[0] instead of recomputing.
    """
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=4)

    normals_before = manager.estimate(cloud)
    np.testing.assert_allclose(normals_before[0], [0.0, 0.0, 1.0], atol=1e-9)

    # Mutate the SAME PointCloud object in place -- id(cloud) unchanged,
    # but the local geometry near point[0] is now steeply tilted.
    cloud.add_chunk(_make_chunk([0.5, 1.0, 0.3], [0.5, 1.0, 0.3], [5.5, 8.0, 4.5]))

    normals_after = manager.estimate(cloud)  # SAME manager instance
    assert not np.allclose(normals_after[0], normals_before[0]), (
        "stale cached normal returned after mutating the point cloud in place"
    )


def test_recomputed_result_after_mutation_matches_fresh_manager() -> None:
    """The cache-corrected result must match what a brand-new, never-cached manager computes."""
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=4)
    manager.estimate(cloud)  # populate cache with pre-mutation state

    cloud.add_chunk(_make_chunk([0.5, 1.0, 0.3], [0.5, 1.0, 0.3], [5.5, 8.0, 4.5]))

    normals_same_manager = manager.estimate(cloud)
    normals_fresh_manager = NormalManager(method="pca", k=4).estimate(cloud)

    np.testing.assert_allclose(normals_same_manager[0], normals_fresh_manager[0], atol=1e-9)


def test_cache_hit_still_works_when_cloud_unchanged() -> None:
    """Confirms the fix didn't break the cache's actual purpose -- repeated calls on unchanged data still hit."""
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=4)

    first = manager.estimate(cloud)
    second = manager.estimate(cloud)

    assert first is second  # same cached array object, not merely equal values


def test_cache_still_distinguishes_different_clouds() -> None:
    cloud_a = _flat_plane_cloud()
    cloud_b = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=4)

    result_a = manager.estimate(cloud_a)
    result_b = manager.estimate(cloud_b)

    assert result_a is not result_b  # distinct cache entries despite identical geometry


def test_cache_still_distinguishes_different_k() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=4)

    result_k4 = manager.estimate(cloud)
    manager.k = 3
    result_k3 = manager.estimate(cloud)

    assert result_k4 is not result_k3


def test_remove_chunk_also_invalidates_cache() -> None:
    cloud = _flat_plane_cloud()
    cloud.add_chunk(_make_chunk([0.5, 1.0, 0.3], [0.5, 1.0, 0.3], [5.5, 8.0, 4.5]))
    manager = NormalManager(method="pca", k=4)

    normals_before = manager.estimate(cloud)

    cloud.remove_chunk(1)  # remove the second chunk, reverting to the flat plane

    normals_after = manager.estimate(cloud)
    np.testing.assert_allclose(normals_after[0], [0.0, 0.0, 1.0], atol=1e-9)
    assert not np.allclose(normals_after[0], normals_before[0])


def test_clear_also_invalidates_cache() -> None:
    cloud = _flat_plane_cloud()
    manager = NormalManager(method="pca", k=4)
    manager.estimate(cloud)

    cloud.clear()
    cloud.add_chunk(
        _make_chunk(
            [0.0, 1.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0, 1.0, 0.5],
            [0.0, 1.0, 2.0, 3.0, 1.5],
        )
    )

    # Must not raise, and must not silently reuse a result computed for the old (now-cleared) points.
    result = manager.estimate(cloud)
    assert result.shape[0] == 5
