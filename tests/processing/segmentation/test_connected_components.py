"""
Coverage audit tests for topocore.processing.segmentation.connected_components.ConnectedComponentsSegmenter.

Confirmed via direct execution: unlike RegionGrowingSegmenter
(SEG-RG-001), cluster_id here is only incremented AFTER confirming a
component meets min_points -- filtering happens synchronously per
component, before moving to the next one, not as a separate
post-processing pass over all components at the end. Verified
directly with a small component (discovered first, filtered as
noise without incrementing cluster_id) followed by a large one (which
correctly receives id=0, not 1): SegmentationResult accepted the
result without any contiguity error. No SEG-RG-001-style defect
found here.

use_adaptive_threshold=True is tested directly even though zero real
callers currently set it -- a legitimate, unguarded public
constructor parameter, same reasoning as DBSCANSegmenter's own
use_adaptive_eps audit.

_compute_threshold_values()'s `n <= 1` branch is confirmed reachable
independently of any manager (checked before `manager` is touched at
all) -- unlike region_growing.py's analogous check, there is no
PCA/normals precondition gating this method that would exclude small
n beforehand.

Two branches confirmed unreachable or very hard to trigger:
  - `k == 0`: mathematically unreachable given `n <= 1` already
    returns early, guaranteeing `k = min(5, n-1) >= min(5, 1) = 1`.
  - `global_mean <= 0.0`: confirmed a near-duplicate (1e-10)
    perturbation still yields a nonzero mean distance, matching the
    same finding already made for DBSCANSegmenter's analogous check.

The `segment_sizes.dtype != np.int64` defensive conversion in
_build_result() is not tested: confirmed directly that
np.bincount()'s own default output dtype on this platform is already
int64, making the conversion branch a platform-variance safeguard
rather than something reachable here.

The `name` property is documented as orphaned -- zero external callers.

Minor code-cleanliness note (not a functional issue, not addressed
here): `from collections import deque` is imported both at module
level (line 26) and redundantly again inside cluster()'s own method
body (lines 102-103).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.connected_components import (
    ConnectedComponentsSegmenter,
)

# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_non_positive_distance_threshold_rejected() -> None:
    with pytest.raises(SegmentationError, match="distance_threshold must be positive"):
        ConnectedComponentsSegmenter(distance_threshold=0.0)


def test_min_points_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_points must be"):
        ConnectedComponentsSegmenter(min_points=0)


# ----------------------------------------------------------------------
# segment() -- empty cloud.
# ----------------------------------------------------------------------


def test_segment_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError, match="empty point cloud"):
        ConnectedComponentsSegmenter().segment(PointCloud())


# ----------------------------------------------------------------------
# Happy path -- confirms ID contiguity is safe when a small,
# first-discovered component is filtered out.
# ----------------------------------------------------------------------


def test_small_component_discovered_first_does_not_break_contiguity() -> None:
    rng = np.random.default_rng(0)
    small = rng.normal([0, 0, 0], 0.05, (3, 3))  # discovered first, below min_points
    big = rng.normal([100, 100, 100], 0.3, (30, 3))  # discovered second
    points = np.vstack([small, big])

    seg = ConnectedComponentsSegmenter(distance_threshold=1.0, min_points=10)
    labels, num_clusters = seg.cluster(points)

    assert num_clusters == 1
    np.testing.assert_array_equal(np.unique(labels), [-1, 0])


def test_segment_builds_valid_segmentation_result() -> None:
    rng = np.random.default_rng(0)
    small = rng.normal([0, 0, 0], 0.05, (3, 3))
    big = rng.normal([100, 100, 100], 0.3, (30, 3))
    points = np.vstack([small, big])

    cloud = PointCloud()
    chunk = Chunk(
        size=len(points),
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = points[:, 0]
    chunk[PointAttribute.Y][:] = points[:, 1]
    chunk[PointAttribute.Z][:] = points[:, 2]
    cloud.add_chunk(chunk)

    result = ConnectedComponentsSegmenter(distance_threshold=1.0, min_points=10).segment(cloud)

    assert result.num_segments == 1


def test_all_components_too_small_gives_zero_clusters() -> None:
    rng = np.random.default_rng(0)
    scattered = rng.uniform(0, 10000, (10, 3))  # too far apart to connect

    labels, num_clusters = ConnectedComponentsSegmenter(distance_threshold=1.0, min_points=5).cluster(scattered)

    assert num_clusters == 0
    np.testing.assert_array_equal(labels, -1)


# ----------------------------------------------------------------------
# use_adaptive_threshold -- legitimate public config, tested directly.
# ----------------------------------------------------------------------


def test_adaptive_threshold_runs_without_error() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, (50, 3))

    labels, num_clusters = ConnectedComponentsSegmenter(distance_threshold=1.0, use_adaptive_threshold=True).cluster(
        points
    )

    assert len(labels) == 50
    assert num_clusters >= 0


# ----------------------------------------------------------------------
# _compute_threshold_values() -- n <= 1, confirmed reachable without a manager.
# ----------------------------------------------------------------------


def test_compute_threshold_values_single_point_returns_constant() -> None:
    seg = ConnectedComponentsSegmenter(distance_threshold=1.0, use_adaptive_threshold=True)

    values = seg._compute_threshold_values(np.array([[0.0, 0.0, 0.0]]), None)  # type: ignore[arg-type]

    np.testing.assert_array_equal(values, [1.0])
