"""
Regression suite for PR21.8: NeighborhoodManager.query_points_many()
and RelativeHeightFeatureComputer's use of it, replacing a genuine
per-point Python loop (one query_point() call per point).

Found during the PR21.7.9 audit of features/geometric.py: the ground
reference structure itself (a NeighborhoodManager built from all
ground points) is genuinely global -- correctly excluded from
PR21.7's chunked-accumulator work -- but the per-point query LOOP
that consults it was a separate, real inefficiency, of the same
class PR21.4 already fixed for local_density(): N individual
single-point KDTree queries instead of one batched call.

Confirmed via direct benchmarking: a ~29x-36x speedup at realistic
sizes (2,000-20,000 points), with numerically identical results to
the original per-point loop (verified directly, for both k=1 and
k=3, before this suite was written).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.features.geometric import RelativeHeightFeatureComputer
from topocore.processing.neighbors.kdtree import KDTreeNeighborSearch
from topocore.processing.neighbors.manager import NeighborhoodManager


@pytest.fixture
def points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 100, (500, 3))


# ----------------------------------------------------------------------
# query_points_many() -- matches query_point() called individually.
# ----------------------------------------------------------------------


def test_query_points_many_matches_individual_query_point_calls_k1(
    points: np.ndarray,
) -> None:
    search = KDTreeNeighborSearch(points)
    rng = np.random.default_rng(1)
    query_points = rng.uniform(0, 100, (20, 3))

    indices_batch, distances_batch = search.query_points_many(query_points, k=1)

    for i in range(20):
        indices_single, distances_single = search.query_point(
            float(query_points[i, 0]),
            float(query_points[i, 1]),
            float(query_points[i, 2]),
            k=1,
        )
        assert indices_single[0] == indices_batch[i, 0]
        assert distances_single[0] == pytest.approx(distances_batch[i, 0])


def test_query_points_many_matches_individual_query_point_calls_k5(
    points: np.ndarray,
) -> None:
    search = KDTreeNeighborSearch(points)
    rng = np.random.default_rng(2)
    query_points = rng.uniform(0, 100, (20, 3))

    indices_batch, distances_batch = search.query_points_many(query_points, k=5)

    for i in range(20):
        indices_single, distances_single = search.query_point(
            float(query_points[i, 0]),
            float(query_points[i, 1]),
            float(query_points[i, 2]),
            k=5,
        )
        np.testing.assert_array_equal(indices_single, indices_batch[i])
        np.testing.assert_allclose(distances_single, distances_batch[i])


def test_query_points_many_shapes(points: np.ndarray) -> None:
    search = KDTreeNeighborSearch(points)
    query_points = np.random.default_rng(3).uniform(0, 100, (7, 3))

    indices_k1, distances_k1 = search.query_points_many(query_points, k=1)
    assert indices_k1.shape == (7, 1)
    assert distances_k1.shape == (7, 1)

    indices_k4, distances_k4 = search.query_points_many(query_points, k=4)
    assert indices_k4.shape == (7, 4)
    assert distances_k4.shape == (7, 4)


def test_neighborhood_manager_query_points_many_uses_default_k(
    points: np.ndarray,
) -> None:
    from topocore.processing.config import NeighborConfig

    manager = NeighborhoodManager.from_array(points, NeighborConfig(knn_default=3))
    query_points = np.random.default_rng(4).uniform(0, 100, (5, 3))

    indices, _distances = manager.query_points_many(query_points)

    assert indices.shape == (5, 3)


def test_query_points_many_rejects_wrong_shape(points: np.ndarray) -> None:
    from topocore.processing.exceptions import NeighborError

    search = KDTreeNeighborSearch(points)
    with pytest.raises(NeighborError):
        search.query_points_many(np.random.default_rng(0).uniform(0, 100, (5, 2)), k=1)


# ----------------------------------------------------------------------
# RelativeHeightFeatureComputer -- matches the pre-PR21.8 per-point loop.
# ----------------------------------------------------------------------


def _make_cloud(n: int, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(
        size=n,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.CLASSIFICATION,
        ],
    )
    chunk[PointAttribute.X][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 100, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.CLASSIFICATION][:] = rng.integers(0, 5, n)
    cloud.add_chunk(chunk)
    return cloud


def _reference_relative_height(cloud: PointCloud, ground_class: int, k: int) -> np.ndarray:
    """The exact pre-PR21.8 per-point-loop algorithm, reimplemented here as an independent reference."""
    (chunk,) = list(cloud)
    x = chunk[PointAttribute.X]
    y = chunk[PointAttribute.Y]
    z = chunk[PointAttribute.Z]
    cls = chunk[PointAttribute.CLASSIFICATION]
    ground_mask = cls == ground_class
    ground_points = np.column_stack((x[ground_mask], y[ground_mask], z[ground_mask]))
    manager = NeighborhoodManager.from_array(ground_points)

    ground_z = np.empty_like(z)
    for i in range(len(z)):
        indices, _ = manager.query_point(float(x[i]), float(y[i]), float(z[i]), k=k)
        if k == 1:
            ground_z[i] = ground_points[indices[0], 2]
        else:
            ground_z[i] = np.mean(ground_points[indices, 2])

    return z - ground_z


def test_relative_height_matches_reference_k1() -> None:
    cloud = _make_cloud(300)
    reference = _reference_relative_height(cloud, ground_class=2, k=1)
    actual = RelativeHeightFeatureComputer(ground_class=2, k=1).compute(cloud)

    np.testing.assert_allclose(actual, reference)


def test_relative_height_matches_reference_k3() -> None:
    cloud = _make_cloud(300, seed=1)
    reference = _reference_relative_height(cloud, ground_class=2, k=3)
    actual = RelativeHeightFeatureComputer(ground_class=2, k=3).compute(cloud)

    np.testing.assert_allclose(actual, reference)


def test_relative_height_no_ground_points_still_raises() -> None:
    from topocore.processing.exceptions import PointDescriptorError

    cloud = PointCloud()
    chunk = Chunk(
        size=5,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.CLASSIFICATION,
        ],
    )
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    chunk[PointAttribute.Y][:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    chunk[PointAttribute.Z][:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    chunk[PointAttribute.CLASSIFICATION][:] = [
        1,
        1,
        1,
        1,
        1,
    ]  # no ground_class=2 points
    cloud.add_chunk(chunk)

    with pytest.raises(PointDescriptorError, match="No ground points"):
        RelativeHeightFeatureComputer(ground_class=2, k=1).compute(cloud)
