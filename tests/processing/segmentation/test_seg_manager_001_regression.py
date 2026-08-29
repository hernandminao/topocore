"""
Regression suite for SEG-MANAGER-001 (fixed in this PR).

Bug: DBSCANSegmenter's own `cache_neighbors` constructor parameter
was never read by SegmentationManager._create_segmenter()'s "dbscan"
dispatch branch. Confirmed directly before the fix:
`SegmentationManager(method="dbscan", cache_neighbors=False)`
silently constructed a DBSCANSegmenter with cache_neighbors=True
(its own class default) regardless of what the caller explicitly
passed -- a parameter-propagation defect, not merely a coverage gap.

Fix: the "dbscan" branch now reads `cache_neighbors` from `p` (the
merged params dict), defaulting to True -- unchanged from
DBSCANSegmenter's own existing default, preserving current behavior
for callers who don't specify it.

Tests below check the ACTUAL VALUE received by the constructed
DBSCANSegmenter instance (`segmenter._cache_neighbors`), not merely
that segment() completes without error -- this is what would have
caught the original regression, since segment() completed
successfully either way (cache_neighbors only affects an internal
performance optimization, not the observable segmentation result,
confirmed directly: identical labels/num_segments with either value).
"""

from __future__ import annotations

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.segmentation.dbscan import DBSCANSegmenter
from topocore.processing.segmentation.manager import SegmentationManager


def _cloud(n: int = 60, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


def test_cache_neighbors_omitted_preserves_existing_default() -> None:
    manager = SegmentationManager(method="dbscan")
    segmenter = manager._create_segmenter()

    assert isinstance(segmenter, DBSCANSegmenter)
    assert segmenter._cache_neighbors is True


def test_cache_neighbors_true_is_respected() -> None:
    manager = SegmentationManager(method="dbscan", cache_neighbors=True)
    segmenter = manager._create_segmenter()

    assert isinstance(segmenter, DBSCANSegmenter)
    assert segmenter._cache_neighbors is True


def test_cache_neighbors_false_is_respected() -> None:
    """The exact regression: False was previously silently discarded, always yielding True."""
    manager = SegmentationManager(method="dbscan", cache_neighbors=False)
    segmenter = manager._create_segmenter()

    assert isinstance(segmenter, DBSCANSegmenter)
    assert segmenter._cache_neighbors is False


def test_cache_neighbors_propagates_via_set_params() -> None:
    manager = SegmentationManager(method="dbscan")
    manager.set_params(cache_neighbors=False)
    segmenter = manager._create_segmenter()

    assert isinstance(segmenter, DBSCANSegmenter)
    assert segmenter._cache_neighbors is False


def test_segment_produces_identical_result_regardless_of_cache_neighbors() -> None:
    """cache_neighbors is an internal performance optimization -- it must not affect the observable result."""
    cloud = _cloud()

    result_cached = SegmentationManager(method="dbscan", eps=1.0, min_samples=3, cache_neighbors=True).segment(cloud)
    result_uncached = SegmentationManager(method="dbscan", eps=1.0, min_samples=3, cache_neighbors=False).segment(cloud)

    assert result_cached.num_segments == result_uncached.num_segments
    np.testing.assert_array_equal(result_cached.labels, result_uncached.labels)
