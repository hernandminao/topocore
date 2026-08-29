"""
Coverage audit tests for topocore.processing.segmentation.dbscan.DBSCANSegmenter.

Confirmed via direct execution: unlike RegionGrowingSegmenter
(SEG-RG-001), DBSCAN's cluster_id counter is never filtered/renumbered
after assignment -- every id from 0 to cluster_id-1 is guaranteed to
have been genuinely created via _expand_cluster(), with no
post-processing step that could remove one and leave a gap. Verified
directly with a 2-cluster + noise scenario: SegmentationResult
accepted the result without any contiguity error. No SEG-RG-001-style
defect found here.

use_adaptive_eps=True is tested directly even though zero real
callers currently set it -- it is a legitimate, unguarded public
constructor parameter (not bypassed by any alternative
implementation), matching the same reasoning as build_cloud()'s own
audit: testable regardless of what current callers happen to use.

Two branches confirmed unreachable or very hard to trigger,
documented rather than force-tested:
  - _compute_eps_values()'s `global_mean_distance <= 0.0` check:
    confirmed that genuinely coincident points make knn_many() itself
    raise NeighborError first (the KDTree can't find a query point
    among its own de-duplicated neighbors), and a near-duplicate
    perturbation (1e-10) still yields a nonzero mean distance -- this
    branch requires an exact zero that the upstream KDTree call
    itself prevents from being reached.
  - _find_nearest_cluster_label()'s NOISE fallback: initially assumed
    reachable via an "isolated" noise point, but confirmed directly
    this assumption was wrong -- the search has NO distance limit, so
    it will find and return an arbitrarily distant clustered point's
    label rather than falling back to NOISE, as long as any cluster
    exists anywhere in the full point set. The fallback is only
    reachable when NO cluster exists in the set at all -- covered
    below by the all-noise + include_noise=False test.

The `name` property is documented as orphaned -- zero external callers.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SegmentationError
from topocore.processing.segmentation.dbscan import DBSCANSegmenter

# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_non_positive_eps_rejected() -> None:
    with pytest.raises(SegmentationError, match="eps must be positive"):
        DBSCANSegmenter(eps=0.0)


def test_min_samples_less_than_one_rejected() -> None:
    with pytest.raises(SegmentationError, match="min_samples must be"):
        DBSCANSegmenter(min_samples=0)


# ----------------------------------------------------------------------
# cluster() / segment() -- empty input.
# ----------------------------------------------------------------------


def test_cluster_empty_array_returns_zero_clusters() -> None:
    labels, num_clusters = DBSCANSegmenter().cluster(np.empty((0, 3)))

    assert num_clusters == 0
    assert len(labels) == 0


def test_segment_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError, match="empty point cloud"):
        DBSCANSegmenter().segment(PointCloud())


# ----------------------------------------------------------------------
# Happy path -- 2 clusters + noise, confirming ID contiguity is safe.
# ----------------------------------------------------------------------


def test_two_clusters_and_noise_produce_contiguous_ids() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal([0, 0, 0], 0.3, (30, 3))
    cluster_b = rng.normal([100, 100, 100], 0.3, (30, 3))
    noise = rng.uniform(200, 210, (5, 3))
    points = np.vstack([cluster_a, cluster_b, noise])

    seg = DBSCANSegmenter(eps=1.0, min_samples=5, include_noise=True)
    labels, num_clusters = seg.cluster(points)

    assert num_clusters == 2
    np.testing.assert_array_equal(np.unique(labels), [-1, 0, 1])


def test_segment_builds_valid_segmentation_result() -> None:
    """Confirms SegmentationResult's own contiguity validation accepts DBSCAN's output without error."""
    rng = np.random.default_rng(0)
    cluster_a = rng.normal([0, 0, 0], 0.3, (30, 3))
    cluster_b = rng.normal([100, 100, 100], 0.3, (30, 3))
    points = np.vstack([cluster_a, cluster_b])

    cloud = PointCloud()
    chunk = Chunk(
        size=len(points),
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = points[:, 0]
    chunk[PointAttribute.Y][:] = points[:, 1]
    chunk[PointAttribute.Z][:] = points[:, 2]
    cloud.add_chunk(chunk)

    result = DBSCANSegmenter(eps=1.0, min_samples=5).segment(cloud)

    assert result.num_segments == 2


# ----------------------------------------------------------------------
# cache_neighbors and use_adaptive_eps -- both legitimate public config options.
# ----------------------------------------------------------------------


def test_cache_neighbors_disabled_still_produces_correct_clusters() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal([0, 0, 0], 0.3, (30, 3))

    with_cache = DBSCANSegmenter(eps=1.0, min_samples=5, cache_neighbors=True).cluster(points)
    without_cache = DBSCANSegmenter(eps=1.0, min_samples=5, cache_neighbors=False).cluster(points)

    np.testing.assert_array_equal(with_cache[0], without_cache[0])
    assert with_cache[1] == without_cache[1]


def test_adaptive_eps_produces_valid_clustering() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, (50, 3))

    labels, num_clusters = DBSCANSegmenter(eps=1.0, min_samples=3, use_adaptive_eps=True).cluster(points)

    assert len(labels) == 50
    assert num_clusters >= 0


# ----------------------------------------------------------------------
# include_noise=False -- reassignment happy path and the genuine
# all-noise NOISE-fallback case.
# ----------------------------------------------------------------------


def test_include_noise_false_reassigns_noise_to_nearest_cluster() -> None:
    rng = np.random.default_rng(0)
    cluster_points = rng.normal([0, 0, 0], 0.1, (20, 3))
    noise_point = np.array([[5.0, 5.0, 5.0]])
    points = np.vstack([cluster_points, noise_point])

    seg = DBSCANSegmenter(eps=1.0, min_samples=5, include_noise=False)
    labels, _ = seg.cluster(points)

    assert (labels == -1).sum() == 0  # no noise labels remain
    assert labels[-1] == 0  # reassigned to the only existing cluster


def test_include_noise_false_with_no_clusters_falls_back_to_noise() -> None:
    """When no cluster exists anywhere in the set, the NOISE fallback in _find_nearest_cluster_label must fire."""
    rng = np.random.default_rng(0)
    scattered = rng.uniform(0, 10000, (10, 3))  # far too dispersed for any core point to form

    seg = DBSCANSegmenter(eps=1.0, min_samples=5, include_noise=False)
    labels, num_clusters = seg.cluster(scattered)

    assert num_clusters == 0
    np.testing.assert_array_equal(labels, -1)
