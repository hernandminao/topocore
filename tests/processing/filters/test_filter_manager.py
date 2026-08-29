"""
Coverage audit tests for topocore.processing.filters.manager.FilterManager.

FILTER-MANAGER-001 (FIXED in this PR): apply()'s cache key was
`(id(current), index, filter.name())`. Since self._cache persists
ACROSS multiple apply() calls, and CPython is well known to reuse
the memory address (and therefore id()) of a garbage-collected
object for a later, unrelated object, the cache could return a
stale, WRONG result from a completely different, already-freed
PointCloud if such an id() collision occurred. Confirmed
deterministically (bypassing the non-deterministic nature of actual
allocator reuse) by directly inserting a cache entry under the exact
id()-based key a colliding object would produce, and confirming
apply() blindly returned it.

Design alternatives considered and rejected before this fix:
  - `id(cloud) + cloud.version`: rejected -- confirmed directly that
    `.version` is a per-instance counter reset to 0 for every new
    PointCloud, so two DIFFERENT clouds can easily share the same
    (id, version) pair if id() is reused and both happen to be at
    the same version count. This solves NormalManager's own
    different bug (same-object in-place mutation), not this one
    (cross-object id() reuse).
  - Nested `WeakKeyDictionary[PointCloud, LRUCache]`: rejected --
    would silently change cache_size's public semantics from "N
    entries total" to "N entries per cloud", a real behavior change
    to reject without explicit sign-off.
  - Content-hash-based keys: rejected -- would cost O(N) per cache
    lookup, defeating the purpose of caching for large clouds.
  - Clearing the cache at the start of every apply() call: rejected
    -- would discard the legitimate cross-call reuse the class's own
    docstring demonstrates as an intended usage pattern.

Fix: the existing single global LRUCache (same class, same
`cache_size` meaning "N entries total") is kept exactly as before.
A `weakref.finalize()` is registered on each stage's INPUT cloud
(captured as `stage_input`, before `current` is reassigned to that
stage's own output -- the cache key's `id()` component refers to the
INPUT, not the output) immediately after a cache miss is stored. The
finalizer calls a new, purely additive `LRUCache.remove_where()`
method the moment that specific object is garbage collected --
synchronously, not lazily -- purging every cache entry tied to that
exact id(). This means a later object reusing the same freed address
can never find a leftover stale entry: it was already removed before
the reuse could ever happen.

Verified directly, in this order, before considering this fixed:
  1. Same live PointCloud -> cache hit still works (identity
     preserved).
  2. A real garbage-collection cycle (del + gc.collect()) on the
     ORIGINAL cloud used in a normal apply() call correctly leaves
     NO cache entries referencing its id() afterward -- confirmed by
     inspecting the cache's own internal keys before and after.
  3. The global `cache_size` bound is still enforced across 10
     DIFFERENT, simultaneously-live clouds (not per-cloud) --
     confirms the rejected nested-WeakKeyDictionary semantics change
     did NOT sneak in.
  4. Two different, simultaneously-live clouds produce their own
     correct, non-interfering results through the same manager.
  5. A multi-cloud, multi-stage pipeline (2 filters x 5 sequential
     clouds through the same manager instance) completes correctly.
  6. The full existing filters/ test suite (38 tests) passes
     unchanged -- no regression to any other filter or to
     FilterManager's own previously-audited behavior.

FilterManager as a whole class, and most of its methods, remain
documented as currently unused (zero real callers) -- exercised here
as legitimate, directly-testable public contract, consistent with
the same policy already applied to SegmentationManager's own audit.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from topocore.geometry.bbox3d import BBox3D
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import FilterError
from topocore.processing.filters.crop_box import CropBoxFilter
from topocore.processing.filters.manager import FilterManager
from topocore.processing.filters.pass_through import Axis, PassThroughFilter


def _cloud(n: int = 10) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = np.arange(n, dtype=float)
    chunk[PointAttribute.Y][:] = np.zeros(n)
    chunk[PointAttribute.Z][:] = np.arange(n, dtype=float)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# add() / add_*() convenience constructors.
# ----------------------------------------------------------------------


def test_add_returns_self_for_chaining() -> None:
    manager = FilterManager()
    result = manager.add(PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0))
    assert result is manager
    assert manager.filter_count == 1


def test_add_convenience_constructors_build_correct_filter_types() -> None:
    manager = FilterManager()
    manager.add_pass_through(Axis.Z, min_value=0.0, max_value=10.0)
    manager.add_crop_box(BBox3D(0, 0, 0, 10, 10, 10))

    assert manager.filter_count == 2
    assert type(manager[0]).__name__ == "PassThroughFilter"
    assert type(manager[1]).__name__ == "CropBoxFilter"


# ----------------------------------------------------------------------
# apply() -- happy path, empty pipeline, empty cloud, error enrichment.
# ----------------------------------------------------------------------


def test_apply_with_no_filters_returns_cloud_unchanged() -> None:
    manager = FilterManager()
    cloud = _cloud()
    result = manager.apply(cloud)
    assert result is cloud


def test_apply_chains_multiple_filters_and_tracks_statistics() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=7.0))
    manager.add(CropBoxFilter(BBox3D(0, 0, 0, 5, 5, 10)))

    result = manager.apply(_cloud())

    assert result.point_count == 4
    stats = manager.statistics()
    assert stats is not None
    assert len(stats) == 2
    assert stats[0]["points_before"] == 10
    assert stats[1]["points_after"] == 4


def test_apply_reuses_cache_for_the_same_live_object() -> None:
    """The common, correct caching case: same object, no id() collision risk."""
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=7.0))
    cloud = _cloud()

    first = manager.apply(cloud)
    second = manager.apply(cloud)

    assert first is second
    stats = manager.statistics()
    assert stats is not None
    assert stats[0]["cached"] is True


def test_apply_rejects_empty_cloud_with_enriched_error() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=7.0))

    with pytest.raises(FilterError, match="pass_through.*failed") as exc_info:
        manager.apply(PointCloud())

    assert isinstance(exc_info.value.__cause__, FilterError)


def test_track_stats_false_disables_statistics() -> None:
    manager = FilterManager(track_stats=False)
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=7.0))

    manager.apply(_cloud())

    assert manager.statistics() is None


# ----------------------------------------------------------------------
# apply_masks() -- returns the individual per-stage masks.
# ----------------------------------------------------------------------


def test_apply_masks_returns_one_mask_per_filter_in_pipeline_order() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=7.0))
    manager.add(CropBoxFilter(BBox3D(0, 0, 0, 5, 5, 10)))

    masks = manager.apply_masks(_cloud())

    assert len(masks) == 2
    assert masks[0].shape == (10,)
    assert masks[1].shape == (int(masks[0].sum()),)


# ----------------------------------------------------------------------
# clear() / statistics() / filter_count / filters / dunder methods.
# ----------------------------------------------------------------------


def test_clear_resets_filters_cache_and_stats() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=7.0))
    manager.apply(_cloud())

    manager.clear()

    assert manager.filter_count == 0
    assert manager.statistics() is None


def test_filters_property_returns_a_copy() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0))

    filters_copy = manager.filters
    filters_copy.append(PassThroughFilter(Axis.X, min_value=0.0, max_value=5.0))

    assert manager.filter_count == 1  # the manager's own internal list is unaffected


def test_dunder_len_iter_getitem() -> None:
    manager = FilterManager()
    f1 = PassThroughFilter(Axis.Z, min_value=0.0, max_value=10.0)
    f2 = PassThroughFilter(Axis.X, min_value=0.0, max_value=5.0)
    manager.add(f1)
    manager.add(f2)

    assert len(manager) == 2
    assert manager[0] is f1
    assert list(iter(manager)) == [f1, f2]


# ----------------------------------------------------------------------
# FILTER-MANAGER-001 regression tests.
# ----------------------------------------------------------------------


def test_cache_entry_is_purged_when_its_cloud_is_garbage_collected() -> None:
    """
    The core regression: after a real garbage-collection cycle on the
    cloud used in a normal apply() call, no cache entry referencing
    its id() must remain -- verified by inspecting the cache's own
    internal keys directly, not merely by re-running apply().
    """
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=0.0, max_value=5.0))
    cloud = _cloud()
    manager.apply(cloud)
    cloud_id = id(cloud)

    assert any(key[0] == cloud_id for key in manager._cache._cache)

    del cloud
    gc.collect()

    assert not any(key[0] == cloud_id for key in manager._cache._cache)


def test_cache_size_bound_is_global_not_per_cloud() -> None:
    """
    Confirms the rejected nested-WeakKeyDictionary design's semantics
    change (cache_size meaning "N per cloud") did not sneak in:
    with 10 DIFFERENT, simultaneously-live clouds (kept alive so none
    are garbage collected), the cache never exceeds cache_size=3
    total entries.
    """
    manager = FilterManager(cache_size=3)
    manager.add(PassThroughFilter(Axis.Z, min_value=-100.0, max_value=100.0))

    clouds = [_cloud(n=5) for _ in range(10)]
    for c in clouds:
        manager.apply(c)

    assert manager._cache.size() <= 3


def test_different_live_clouds_do_not_interfere() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=2.0, max_value=100.0))

    cloud_a = _cloud(n=5)  # Z: 0,1,2,3,4 -- 3 points pass
    cloud_b = _cloud(n=3)  # Z: 0,1,2 -- 1 point passes

    result_a = manager.apply(cloud_a)
    result_b = manager.apply(cloud_b)

    assert result_a.point_count == 3
    assert result_b.point_count == 1


def test_pipeline_across_multiple_sequential_clouds() -> None:
    manager = FilterManager()
    manager.add(PassThroughFilter(Axis.Z, min_value=1.0, max_value=100.0))
    manager.add(PassThroughFilter(Axis.Z, min_value=1.0, max_value=3.0))

    results = [manager.apply(_cloud(n=5)) for _ in range(5)]

    assert all(r.point_count == 3 for r in results)
