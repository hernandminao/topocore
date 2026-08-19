"""
Regression suite for topocore.processing.normals.manager.NormalManager
-- PR19.

NormalManager's cache existed (LRUCache, _cache_key()) but was never
actually wired into estimate()/estimate_at()/estimate_curvature()/
estimate_both() -- harmless (always recomputed correctly) but
completely inert, contradicting the class's own docstring ("...with
automatic method selection and caching"). While designing the fix,
auditing topocore.processing.features.manager.FeatureManager (a
"working" reference) revealed a SEVERE, separate bug there: id(cloud)
frozen at construction, causing silently stale cross-cloud results
(see test_manager.py in that package). This module's cache key was
redesigned from the start to avoid that mistake -- id(cloud) is
computed fresh every call, and viewpoint is keyed by its actual (x,
y, z) value, not merely whether one was given (an earlier version of
THIS module's own _cache_key(), never wired up, only stored
`viewpoint is not None` -- caught during this audit, before caching
was ever connected, not a bug that ever shipped live).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.normals.manager import NormalManager


def _grid_cloud(z_fn) -> PointCloud:  # type: ignore[no-untyped-def]
    xs, ys, zs = [], [], []
    for i in range(5):
        for j in range(5):
            xs.append(float(i))
            ys.append(float(j))
            zs.append(z_fn(i, j))
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def flat_cloud() -> PointCloud:
    return _grid_cloud(lambda i, j: 0.0)


@pytest.fixture
def tilted_cloud() -> PointCloud:
    return _grid_cloud(lambda i, j: float(i))  # z = x


# ----------------------------------------------------------------------
# Different clouds must never collide in the cache.
# ----------------------------------------------------------------------


def test_different_clouds_give_different_normals(flat_cloud: PointCloud, tilted_cloud: PointCloud) -> None:
    manager = NormalManager(k=9)

    flat_normal = manager.estimate(flat_cloud)[12]
    tilted_normal = manager.estimate(tilted_cloud)[12]

    assert not np.allclose(flat_normal, tilted_normal)


def test_alternating_clouds_never_return_stale_results(flat_cloud: PointCloud, tilted_cloud: PointCloud) -> None:
    manager = NormalManager(k=9)

    for _ in range(3):
        assert np.allclose(manager.estimate(flat_cloud)[12], [0.0, 0.0, 1.0])
        assert not np.allclose(manager.estimate(tilted_cloud)[12], [0.0, 0.0, 1.0])


# ----------------------------------------------------------------------
# Different viewpoints must never collide (the bug caught before it
# was ever wired up).
# ----------------------------------------------------------------------


def test_different_viewpoints_give_different_normals(flat_cloud: PointCloud) -> None:
    manager = NormalManager(k=9)

    below = np.array([2.0, 2.0, -10.0])
    above = np.array([2.0, 2.0, 10.0])

    normal_below = manager.estimate(flat_cloud, viewpoint=below)[12]
    normal_above = manager.estimate(flat_cloud, viewpoint=above)[12]

    assert not np.allclose(normal_below, normal_above)
    assert normal_below[2] < 0.0
    assert normal_above[2] > 0.0


# ----------------------------------------------------------------------
# Different sigma (weighted_pca) must never collide.
# ----------------------------------------------------------------------


def test_different_sigma_can_give_different_results() -> None:
    """
    An irregular (non-grid) cloud where sigma genuinely changes the
    weighting balance enough to produce a measurably different
    curvature at a corner point.
    """
    xs = [0.0, 1.0, 2.0, 0.0, 2.0, 0.5, 1.5, 0.0, 1.0, 2.0]
    ys = [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 2.0]
    zs = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0]  # one bump in the middle
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    manager = NormalManager(method="weighted_pca", k=6)
    curvature_tight = manager.estimate_curvature(cloud, sigma=0.3)
    curvature_wide = manager.estimate_curvature(cloud, sigma=5.0)

    assert not np.allclose(curvature_tight, curvature_wide)


# ----------------------------------------------------------------------
# Cache genuinely caches.
# ----------------------------------------------------------------------


def test_repeated_call_same_cloud_returns_identical_object(
    flat_cloud: PointCloud,
) -> None:
    manager = NormalManager(k=9)
    first = manager.estimate(flat_cloud)
    second = manager.estimate(flat_cloud)
    assert first is second


def test_estimate_and_estimate_both_share_one_cache_entry(
    flat_cloud: PointCloud,
) -> None:
    """
    estimate() and estimate_both() for the same cloud/params must
    hit the SAME underlying computation -- confirms the shared
    _estimate_both_cached() helper, not two independent caches.
    """
    manager = NormalManager(k=9)
    normals_only = manager.estimate(flat_cloud)
    normals_both, _ = manager.estimate_both(flat_cloud)

    assert normals_only is normals_both


def test_estimate_at_uses_same_cache_as_estimate(flat_cloud: PointCloud) -> None:
    manager = NormalManager(k=9)
    full = manager.estimate(flat_cloud)
    subset = manager.estimate_at(flat_cloud, indices=np.array([12]))

    np.testing.assert_array_equal(subset[0], full[12])


def test_different_k_does_not_collide(flat_cloud: PointCloud) -> None:
    manager = NormalManager(k=5)
    result_k5 = manager.estimate(flat_cloud)
    result_k9 = manager.estimate(flat_cloud, k=9)

    # Both should still be flat-plane normals (both correct), but
    # verifies the call with a different k doesn't just silently
    # reuse k=5's cache entry -- checked via cache identity, not
    # value (both would coincidentally be [0,0,1] on a flat plane).
    assert result_k5 is not result_k9


def test_clear_cache_forces_recomputation(flat_cloud: PointCloud) -> None:
    manager = NormalManager(k=9)
    first = manager.estimate(flat_cloud)
    manager.clear_cache()
    second = manager.estimate(flat_cloud)

    assert first is not second
    np.testing.assert_array_equal(first, second)  # still numerically identical


def test_changing_k_property_clears_cache(flat_cloud: PointCloud) -> None:
    manager = NormalManager(k=9)
    first = manager.estimate(flat_cloud)
    manager.k = 7
    manager.k = 9  # back to original value
    second = manager.estimate(flat_cloud)

    assert first is not second  # cache was cleared by the k= setter
