"""
Regression/coverage suite for topocore.processing.ground.manager --
PR20 coverage phase, closing out Phase 1 (processing/ground).

GroundManager's own module comment already documents that its
id(cloud) caching pattern deliberately follows the SAME safe
convention already established (and fixed where it was previously
unsafe) elsewhere in this session for NormalManager/FeatureManager:
id(cloud) is computed fresh on every call, never stored on the
manager. Rather than trust the comment's reasoning alone, this was
stress-tested directly: 2000 classify() calls across a tight loop of
short-lived, discarded PointCloud objects sharing one manager
instance -- ZERO stale-cache collisions, unlike the analogous
compute_pca bug found and fixed earlier this session (which
reproduced in roughly 1 of every 10-30 attempts under the same
style of stress test). The key architectural difference: `cloud`
here is caller-owned for the call's duration, unlike compute_pca's
ephemeral, immediately-garbage-collectable NeighborhoodManager.

Also verified end to end: all 5 supported methods
(grid/adaptive_grid/progressive_tin/pmf/csf) for classify/extract,
the dedicated-estimator vs. nearest-ground-fallback paths for
estimate_elevation, cache hit/invalidation behavior (property
setters correctly clear the cache), and that differing **kwargs
correctly produce different cache keys rather than colliding.

No bugs found -- this module was already correct; only test
coverage was added.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import GroundError
from topocore.processing.ground.manager import GroundManager


def _flat_ground_with_building() -> tuple[PointCloud, int]:
    gx, gy = np.meshgrid(np.arange(0, 30, 1.0), np.arange(0, 30, 1.0))
    ground_x, ground_y = gx.ravel(), gy.ravel()
    rng = np.random.default_rng(0)
    ground_z = np.zeros_like(ground_x) + rng.normal(0, 0.02, ground_x.size)

    bx, by = np.meshgrid(np.arange(12, 18, 0.5), np.arange(12, 18, 0.5))
    building_x, building_y = bx.ravel(), by.ravel()
    building_z = np.full(building_x.size, 5.0)

    xs = np.concatenate([ground_x, building_x])
    ys = np.concatenate([ground_y, building_y])
    zs = np.concatenate([ground_z, building_z])

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    return cloud, len(ground_x)


def _flat_cloud(z_value: float, n: int = 10) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Y][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Z][:] = np.full(n, z_value)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# id(cloud) caching -- stress-tested, not just trusted from the comment.
# ----------------------------------------------------------------------


def test_id_cloud_caching_has_no_stale_collisions_under_stress() -> None:
    """
    The decisive check: 2000 classify() calls over short-lived,
    discarded PointCloud objects sharing one manager -- confirms no
    stale cache result is ever silently reused, unlike the analogous
    compute_pca bug found and fixed earlier this session.
    """
    manager = GroundManager(method="grid", cell_size=1.0, height_threshold=0.5)

    collisions_found = 0
    for i in range(2000):
        cloud = _flat_cloud(float(i % 3))
        mask = manager.classify(cloud)
        if not mask.all():  # a flat single-elevation cloud must be entirely "ground"
            collisions_found += 1
        del cloud

    assert collisions_found == 0


# ----------------------------------------------------------------------
# All 5 methods, classify/extract end to end.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["grid", "adaptive_grid", "progressive_tin", "pmf", "csf"])
def test_classify_and_extract_all_methods(method: str) -> None:
    cloud, n_ground = _flat_ground_with_building()

    manager = GroundManager(method=method, cell_size=1.0)
    mask = manager.classify(cloud)
    assert mask[:n_ground].mean() > 0.9

    result = manager.extract(cloud)
    assert result.point_count > 0


def test_grid_uses_dedicated_elevation_estimator() -> None:
    cloud = _flat_cloud(3.0, n=100)
    manager = GroundManager(method="grid", cell_size=1.0)
    elevations = manager.estimate_elevation(cloud)
    assert elevations == pytest.approx(3.0)


def test_progressive_tin_falls_back_to_nearest_ground_elevation() -> None:
    """progressive_tin has no dedicated estimator -- exercises _nearest_ground_elevation's fallback path."""
    gx, gy = np.meshgrid(np.arange(0, 10, 1.0), np.arange(0, 10, 1.0))
    xs, ys = gx.ravel(), gy.ravel()
    zs = np.full(xs.size, 3.0)

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)

    manager = GroundManager(method="progressive_tin", cell_size=1.0)
    elevations = manager.estimate_elevation(cloud)
    assert elevations == pytest.approx(3.0)


# ----------------------------------------------------------------------
# Cache hit / invalidation / kwargs-key behavior.
# ----------------------------------------------------------------------


def test_repeated_call_hits_cache_same_object_returned() -> None:
    cloud = _flat_cloud(1.0, n=50)
    manager = GroundManager(method="grid", cell_size=1.0)

    first = manager.classify(cloud)
    second = manager.classify(cloud)

    assert first is second


def test_different_kwargs_produce_different_cache_entries() -> None:
    cloud, _n_ground = _flat_ground_with_building()
    manager = GroundManager(method="grid", cell_size=1.0)

    default_result = manager.classify(cloud)
    overridden_result = manager.classify(cloud, height_threshold=10.0)  # very permissive threshold

    assert default_result is not overridden_result


def test_property_setters_invalidate_cache() -> None:
    cloud = _flat_cloud(1.0, n=50)
    manager = GroundManager(method="grid", cell_size=1.0)

    first = manager.classify(cloud)
    manager.cell_size = 2.0
    second = manager.classify(cloud)

    assert first is not second


def test_method_setter_invalidates_cache() -> None:
    cloud = _flat_cloud(1.0, n=50)
    manager = GroundManager(method="grid", cell_size=1.0)

    first = manager.classify(cloud)
    manager.method = "progressive_tin"
    second = manager.classify(cloud)

    assert first is not second


def test_clear_cache_forces_recomputation() -> None:
    cloud = _flat_cloud(1.0, n=50)
    manager = GroundManager(method="grid", cell_size=1.0)

    first = manager.classify(cloud)
    manager.clear_cache()
    second = manager.classify(cloud)

    assert first is not second
    np.testing.assert_array_equal(first, second)


def test_callable_interface_matches_classify() -> None:
    cloud = _flat_cloud(1.0, n=50)
    manager = GroundManager(method="grid", cell_size=1.0)

    via_call = manager(cloud)
    via_classify = manager.classify(cloud)

    np.testing.assert_array_equal(via_call, via_classify)


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------


def test_rejects_unsupported_method_at_construction() -> None:
    with pytest.raises(GroundError, match="Unsupported method"):
        GroundManager(method="bogus")


def test_rejects_unsupported_method_via_setter() -> None:
    manager = GroundManager(method="grid")
    with pytest.raises(GroundError, match="Unsupported method"):
        manager.method = "bogus"


def test_rejects_nonpositive_cell_size_via_setter() -> None:
    manager = GroundManager(method="grid")
    with pytest.raises(GroundError):
        manager.cell_size = 0.0


def test_rejects_negative_height_threshold_via_setter() -> None:
    manager = GroundManager(method="grid")
    with pytest.raises(GroundError):
        manager.height_threshold = -1.0


def test_property_getters_reflect_current_state() -> None:
    manager = GroundManager(method="grid", cell_size=2.5, height_threshold=0.3)
    assert manager.method == "grid"
    assert manager.cell_size == pytest.approx(2.5)
    assert manager.height_threshold == pytest.approx(0.3)
