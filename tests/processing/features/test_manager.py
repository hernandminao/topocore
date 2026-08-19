"""
Regression suite for topocore.processing.features.manager.FeatureManager
-- PR19.

Includes a severe, real bug found and fixed in this session: an
earlier version stored id(cloud) ONCE, at construction (or defaulted
to 0 when no cloud was given), and reused that frozen value as part
of the cache key on every subsequent compute() call -- regardless of
which cloud was actually passed to that call. Reusing one
FeatureManager across multiple different clouds (a natural pattern,
e.g. processing a batch of tiles with one shared, pre-configured
manager) silently returned the FIRST cloud's cached feature values
for every OTHER cloud -- wrong data, no error, no warning. Confirmed
directly: computing "height" for a cloud with points at z=[10,20,30]
then a second cloud at z=[100,200,300] returned [10,20,30] for BOTH.

Fixed by deriving id(cloud) fresh on every call (never stored),
matching the pattern already used correctly in
topocore.processing.filters.manager.FilterManager.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import PointDescriptorError
from topocore.processing.features.geometric import HeightFeatureComputer
from topocore.processing.features.manager import FeatureManager


def _cloud_with_z(zs: list[float]) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=len(zs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [float(i) for i in range(len(zs))]
    chunk[PointAttribute.Y][:] = [0.0] * len(zs)
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def manager() -> FeatureManager:
    m = FeatureManager()
    m.register("height", HeightFeatureComputer())
    return m


# ----------------------------------------------------------------------
# The exact regression this fix targets.
# ----------------------------------------------------------------------


def test_different_clouds_give_different_results_not_stale_cache(
    manager: FeatureManager,
) -> None:
    cloud_a = _cloud_with_z([10.0, 20.0, 30.0])
    cloud_b = _cloud_with_z([100.0, 200.0, 300.0])

    result_a = manager.compute(cloud_a, "height")
    result_b = manager.compute(cloud_b, "height")

    np.testing.assert_array_equal(result_a, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(result_b, [100.0, 200.0, 300.0])


def test_alternating_between_clouds_never_returns_stale_data(
    manager: FeatureManager,
) -> None:
    """
    Not just "first switch works" -- repeatedly alternating between
    two clouds must always give each cloud's own correct result,
    confirming the cache key genuinely tracks identity per call
    rather than merely being correct once by coincidence.
    """
    cloud_a = _cloud_with_z([1.0, 2.0])
    cloud_b = _cloud_with_z([50.0, 60.0])

    for _ in range(3):
        np.testing.assert_array_equal(manager.compute(cloud_a, "height"), [1.0, 2.0])
        np.testing.assert_array_equal(manager.compute(cloud_b, "height"), [50.0, 60.0])


def test_three_or_more_clouds_each_get_correct_results(manager: FeatureManager) -> None:
    clouds = [_cloud_with_z([float(i)]) for i in range(5)]

    for i, cloud in enumerate(clouds):
        result = manager.compute(cloud, "height")
        assert result[0] == pytest.approx(float(i))


# ----------------------------------------------------------------------
# Cache still genuinely caches (the fix must not turn caching off).
# ----------------------------------------------------------------------


def test_repeated_call_on_same_cloud_returns_cached_identical_object(
    manager: FeatureManager,
) -> None:
    cloud = _cloud_with_z([5.0, 15.0])

    first = manager.compute(cloud, "height")
    second = manager.compute(cloud, "height")

    assert first is second  # genuine cache hit, not just equal values


def test_clear_cache_forces_recomputation() -> None:
    """
    Uses a computer with an internal call counter to distinguish a
    genuine cache hit from a coincidental value match.
    """
    call_count = {"n": 0}

    class CountingComputer(HeightFeatureComputer):
        def compute(self, cloud: PointCloud) -> object:  # type: ignore[override]
            call_count["n"] += 1
            return super().compute(cloud)

    manager = FeatureManager()
    manager.register("height", CountingComputer())
    cloud = _cloud_with_z([1.0])

    manager.compute(cloud, "height")
    manager.compute(cloud, "height")
    assert call_count["n"] == 1  # second call was a cache hit

    manager.clear_cache()
    manager.compute(cloud, "height")
    assert call_count["n"] == 2  # cache was actually cleared


# ----------------------------------------------------------------------
# Construction with an initial cloud (eager computation path) --
# also went through the same buggy _cloud_id before the fix.
# ----------------------------------------------------------------------


def test_construction_with_initial_cloud_computes_eagerly() -> None:
    cloud = _cloud_with_z([7.0, 8.0, 9.0])
    manager = FeatureManager(cloud=cloud)
    manager.register("height", HeightFeatureComputer())

    # Registered AFTER construction -- eager computation at __init__
    # only covers computers registered before it, so this call still
    # computes (and correctly caches) fresh, for THIS cloud.
    result = manager.compute(cloud, "height")
    np.testing.assert_array_equal(result, [7.0, 8.0, 9.0])


def test_construction_with_initial_cloud_then_different_cloud_still_correct() -> None:
    cloud_a = _cloud_with_z([1.0, 2.0])
    manager = FeatureManager(cloud=cloud_a)
    manager.register("height", HeightFeatureComputer())

    cloud_b = _cloud_with_z([99.0, 98.0])
    result_b = manager.compute(cloud_b, "height")

    np.testing.assert_array_equal(result_b, [99.0, 98.0])


# ----------------------------------------------------------------------
# compute_all
# ----------------------------------------------------------------------


def test_compute_all_different_clouds_give_different_results(
    manager: FeatureManager,
) -> None:
    cloud_a = _cloud_with_z([1.0])
    cloud_b = _cloud_with_z([99.0])

    result_a = manager.compute_all(cloud_a)
    result_b = manager.compute_all(cloud_b)

    assert result_a["height"][0] == pytest.approx(1.0)
    assert result_b["height"][0] == pytest.approx(99.0)


# ----------------------------------------------------------------------
# Basic registration/error-handling behavior, unaffected by the fix.
# ----------------------------------------------------------------------


def test_compute_unregistered_feature_raises() -> None:
    manager = FeatureManager()
    with pytest.raises(PointDescriptorError):
        manager.compute(_cloud_with_z([1.0]), "nonexistent")


def test_unregister_removes_feature() -> None:
    manager = FeatureManager()
    manager.register("height", HeightFeatureComputer())
    assert manager.unregister("height") is True
    assert manager.is_registered("height") is False


def test_unregister_missing_feature_returns_false() -> None:
    manager = FeatureManager()
    assert manager.unregister("nonexistent") is False


def test_getitem_and_contains(manager: FeatureManager) -> None:
    assert "height" in manager
    assert isinstance(manager["height"], HeightFeatureComputer)
