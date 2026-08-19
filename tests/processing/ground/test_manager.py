"""
Regression suite for topocore.processing.ground.manager.GroundManager
-- PR19.

Same dead-cache pattern as NormalManager (found in the same audit
pass), now fixed for classify() specifically -- the only method the
originally-declared cache type (BoolArray1D) was ever designed to
hold. extract()/estimate_elevation()'s dedicated extractor/estimator
paths remain uncached (a scoping decision, documented in the source
-- see CacheKey's docstring in manager.py), not attempted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.ground.manager import GroundManager


def _grid_cloud(z_fn) -> PointCloud:  # type: ignore[no-untyped-def]
    xs, ys, zs = [], [], []
    for i in range(10):
        for j in range(10):
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
def raised_cloud() -> PointCloud:
    return _grid_cloud(lambda i, j: 50.0)


# ----------------------------------------------------------------------
# Different clouds must never collide.
# ----------------------------------------------------------------------


def test_different_clouds_get_independent_results(flat_cloud: PointCloud, raised_cloud: PointCloud) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)

    mask_flat = manager.classify(flat_cloud)
    mask_raised = manager.classify(raised_cloud)

    # Both internally flat -> both fully classified as ground,
    # independently -- the point is that computing the SECOND does
    # not silently reuse the FIRST cloud's cached mask object.
    assert mask_flat is not mask_raised
    assert mask_flat.all()
    assert mask_raised.all()


def test_alternating_clouds_never_stale(flat_cloud: PointCloud, raised_cloud: PointCloud) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)

    for _ in range(3):
        assert manager.classify(flat_cloud).all()
        assert manager.classify(raised_cloud).all()


# ----------------------------------------------------------------------
# Per-call kwargs overrides must not collide with default params.
# ----------------------------------------------------------------------


def test_kwargs_override_does_not_collide_with_default(flat_cloud: PointCloud) -> None:
    manager = GroundManager(method="grid", cell_size=1.0, height_threshold=0.5)

    default_result = manager.classify(flat_cloud)
    override_result = manager.classify(flat_cloud, height_threshold=0.01)

    assert default_result is not override_result  # distinct cache entries


def test_repeated_call_same_params_returns_identical_object(
    flat_cloud: PointCloud,
) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)

    first = manager.classify(flat_cloud)
    second = manager.classify(flat_cloud)

    assert first is second


def test_repeated_kwargs_override_also_hits_cache(flat_cloud: PointCloud) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)

    first = manager.classify(flat_cloud, height_threshold=0.3)
    second = manager.classify(flat_cloud, height_threshold=0.3)

    assert first is second


# ----------------------------------------------------------------------
# Different method must not collide.
# ----------------------------------------------------------------------


def test_different_method_does_not_collide(flat_cloud: PointCloud) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)
    grid_result = manager.classify(flat_cloud)

    manager.method = "progressive_tin"
    tin_result = manager.classify(flat_cloud)

    assert grid_result is not tin_result


# ----------------------------------------------------------------------
# clear_cache() / property setters invalidate correctly.
# ----------------------------------------------------------------------


def test_clear_cache_forces_recomputation(flat_cloud: PointCloud) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)
    first = manager.classify(flat_cloud)
    manager.clear_cache()
    second = manager.classify(flat_cloud)

    assert first is not second
    np.testing.assert_array_equal(first, second)


def test_changing_cell_size_property_invalidates_old_entries(
    flat_cloud: PointCloud,
) -> None:
    manager = GroundManager(method="grid", cell_size=1.0)
    first = manager.classify(flat_cloud)

    manager.cell_size = 2.0
    manager.cell_size = 1.0  # back to original
    second = manager.classify(flat_cloud)

    assert first is not second  # cache was cleared by the setter
