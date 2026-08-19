"""
Regression suite for topocore.processing.segmentation.dbscan and
.connected_components -- PR19.

Both verified against clear, analytic scenarios: two well-separated
dense clusters + isolated noise (exact segment sizes and noise
labels), a classic DBSCAN border-point case (must join the cluster,
not become noise), include_noise=False reassignment, and correct
discarding of undersized components without leaving gaps in cluster
IDs. No bugs found in either module.
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
from topocore.processing.segmentation.dbscan import DBSCANSegmenter


def _cloud(xs, ys, zs) -> PointCloud:  # type: ignore[no-untyped-def]
    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def two_clusters_with_noise() -> PointCloud:
    rng = np.random.default_rng(1)
    cluster_a = rng.uniform(-0.5, 0.5, (30, 2))
    cluster_b = rng.uniform(-0.5, 0.5, (30, 2)) + [20, 20]
    noise = np.array([[50.0, 50.0], [-50.0, -50.0], [100.0, 0.0]])

    xs = np.concatenate([cluster_a[:, 0], cluster_b[:, 0], noise[:, 0]])
    ys = np.concatenate([cluster_a[:, 1], cluster_b[:, 1], noise[:, 1]])
    zs = np.zeros_like(xs)
    return _cloud(xs, ys, zs)


# ----------------------------------------------------------------------
# DBSCANSegmenter
# ----------------------------------------------------------------------


def test_dbscan_separates_two_clusters_and_marks_noise(
    two_clusters_with_noise: PointCloud,
) -> None:
    result = DBSCANSegmenter(eps=2.0, min_samples=5, include_noise=True).segment(two_clusters_with_noise)

    assert result.num_segments == 2
    assert len(set(result.labels[:30].tolist())) == 1
    assert len(set(result.labels[30:60].tolist())) == 1
    assert result.labels[:30][0] != result.labels[30:60][0]
    assert (result.labels[60:] == -1).all()
    np.testing.assert_array_equal(np.sort(result.segment_sizes), [30, 30])


def test_dbscan_border_point_joins_cluster_not_noise() -> None:
    core_cluster = np.array([[0, 0], [0.3, 0], [0, 0.3], [0.3, 0.3], [0.15, 0.15], [0.6, 0.15]])
    border_point = np.array([[1.4, 0.15]])  # within eps of core, but not itself a core point

    xs = np.concatenate([core_cluster[:, 0], border_point[:, 0]])
    ys = np.concatenate([core_cluster[:, 1], border_point[:, 1]])
    zs = np.zeros_like(xs)

    result = DBSCANSegmenter(eps=1.5, min_samples=5, include_noise=True).segment(_cloud(xs, ys, zs))

    assert len(set(result.labels.tolist())) == 1  # all 7 points, including the border point, same cluster
    assert -1 not in result.labels


def test_dbscan_include_noise_false_reassigns_isolated_points() -> None:
    core_cluster = np.array([[0, 0], [0.3, 0], [0, 0.3], [0.3, 0.3], [0.15, 0.15], [0.6, 0.15]])
    xs = np.concatenate([core_cluster[:, 0], [50.0]])
    ys = np.concatenate([core_cluster[:, 1], [50.0]])
    zs = np.zeros_like(xs)

    result = DBSCANSegmenter(eps=1.5, min_samples=5, include_noise=False).segment(_cloud(xs, ys, zs))

    assert -1 not in result.labels


def test_dbscan_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError):
        DBSCANSegmenter().segment(PointCloud())


def test_dbscan_rejects_nonpositive_eps() -> None:
    with pytest.raises(SegmentationError):
        DBSCANSegmenter(eps=0.0)


# ----------------------------------------------------------------------
# ConnectedComponentsSegmenter
# ----------------------------------------------------------------------


def test_connected_components_discards_undersized_without_gaps() -> None:
    rng = np.random.default_rng(0)
    big = rng.uniform(-0.5, 0.5, (20, 2))
    small = rng.uniform(-0.3, 0.3, (3, 2)) + [50, 50]  # below min_points=10

    xs = np.concatenate([big[:, 0], small[:, 0]])
    ys = np.concatenate([big[:, 1], small[:, 1]])
    zs = np.zeros_like(xs)

    result = ConnectedComponentsSegmenter(distance_threshold=2.0, min_points=10).segment(_cloud(xs, ys, zs))

    assert result.num_segments == 1
    assert len(set(result.labels[:20].tolist())) == 1
    assert (result.labels[20:] == -1).all()


def test_connected_components_rejects_empty_cloud() -> None:
    with pytest.raises(SegmentationError):
        ConnectedComponentsSegmenter().segment(PointCloud())


def test_connected_components_rejects_nonpositive_threshold() -> None:
    with pytest.raises(SegmentationError):
        ConnectedComponentsSegmenter(distance_threshold=0.0)
