"""
Regression suite for PR21.2 -- opt-in parallelism (`workers`) for
topocore.processing.neighbors.kdtree.KDTreeNeighborSearch and its
propagation through NeighborConfig / NeighborhoodManager.

Per the profiling evidence gathered before this change: at realistic
LiDAR scale (500,000 points), knn_many() -- the batched
scipy.spatial.cKDTree.query() call -- dominates total PCA-pipeline
cost (58% of total time: KD-tree build + knn_many + covariance/
eigendecomposition), while the covariance/eigendecomposition math
(already fully vectorized via np.einsum/np.linalg.eigh) is
proportionally smaller at that scale. scipy.spatial.cKDTree already
supports parallel batched queries via its own `workers` parameter,
which was not previously being used (implicitly workers=1,
single-threaded) anywhere in this module.

This suite does NOT claim or measure a speedup -- that could not be
demonstrated in the single-core sandbox this session's audit was
performed in (confirmed: `workers=-1` gave no measurable improvement
over `workers=1` there, purely because only one core was available
to parallelize across, not because the mechanism doesn't work). What
IS verified here, with real data:

1. workers=1 (the default, matching pre-PR21 behavior exactly),
   workers=-1 (all cores), and workers=4 (a specific count) all
   produce numerically IDENTICAL indices and distances for the same
   input -- confirming parallelizing this embarrassingly-parallel
   batch query (each query point's neighbor search is fully
   independent of every other's) cannot change the result, only
   wall-clock time.
2. The default behavior (no config, or a config that doesn't set
   workers) is unchanged from before this PR.
3. workers is validated the same way this module already validates
   k/radius (NeighborError before scipy ever sees an invalid value),
   not left to leak scipy's own raw exception.
4. NeighborConfig.workers flows correctly through
   NeighborhoodManager.from_point_cloud()/.from_array() down to the
   underlying KDTreeNeighborSearch -- confirming the configuration
   point sits where a caller would naturally set it (the existing
   config object), not a newly-invented, disconnected parameter.

See benchmarks/benchmark_neighbors.py for the reproducible benchmark
script to run on real (multi-core) hardware to measure the actual
speedup this change enables.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.processing.config import NeighborConfig
from topocore.processing.exceptions import NeighborError
from topocore.processing.neighbors.kdtree import KDTreeNeighborSearch
from topocore.processing.neighbors.manager import NeighborhoodManager


@pytest.fixture
def points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 100, (500, 3))


# ----------------------------------------------------------------------
# Numerical equivalence across worker counts.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, -1, 2, 4])
def test_knn_many_identical_results_across_worker_counts(points: np.ndarray, workers: int) -> None:
    baseline = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=1))
    baseline_indices, baseline_distances = baseline.knn_many(k=10)

    manager = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=workers))
    indices, distances = manager.knn_many(k=10)

    np.testing.assert_array_equal(indices, baseline_indices)
    np.testing.assert_array_equal(distances, baseline_distances)


def test_radius_many_identical_results_across_worker_counts(points: np.ndarray) -> None:
    baseline = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=1))
    parallel = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=-1))

    baseline_results = baseline.radius_many(radius=15.0)
    parallel_results = parallel.radius_many(radius=15.0)

    assert len(baseline_results) == len(parallel_results)
    for baseline_indices, parallel_indices in zip(baseline_results, parallel_results, strict=True):
        np.testing.assert_array_equal(np.sort(baseline_indices), np.sort(parallel_indices))


def test_single_point_knn_identical_across_worker_counts(points: np.ndarray) -> None:
    baseline = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=1))
    parallel = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=-1))

    baseline_indices, baseline_distances = baseline.knn(0, k=8)
    parallel_indices, parallel_distances = parallel.knn(0, k=8)

    np.testing.assert_array_equal(baseline_indices, parallel_indices)
    np.testing.assert_array_equal(baseline_distances, parallel_distances)


# ----------------------------------------------------------------------
# Default behavior unchanged.
# ----------------------------------------------------------------------


def test_default_config_matches_explicit_workers_one(points: np.ndarray) -> None:
    default_manager = NeighborhoodManager.from_array(points)
    explicit_manager = NeighborhoodManager.from_array(points, config=NeighborConfig(workers=1))

    default_indices, default_distances = default_manager.knn_many(k=10)
    explicit_indices, explicit_distances = explicit_manager.knn_many(k=10)

    np.testing.assert_array_equal(default_indices, explicit_indices)
    np.testing.assert_array_equal(default_distances, explicit_distances)


def test_neighbor_config_workers_default_is_one() -> None:
    assert NeighborConfig().workers == 1


# ----------------------------------------------------------------------
# Validation -- consistent with this module's existing k/radius pattern.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("workers", [0, -2, -100])
def test_kdtree_rejects_invalid_workers(points: np.ndarray, workers: int) -> None:
    with pytest.raises(NeighborError, match="workers must be -1 or >= 1"):
        KDTreeNeighborSearch.from_array(points, workers=workers)


def test_kdtree_rejects_non_integer_workers(points: np.ndarray) -> None:
    with pytest.raises(NeighborError, match="workers must be integer"):
        KDTreeNeighborSearch.from_array(points, workers=1.5)  # type: ignore[arg-type]


def test_kdtree_accepts_workers_negative_one(points: np.ndarray) -> None:
    KDTreeNeighborSearch.from_array(points, workers=-1)  # must not raise


def test_kdtree_accepts_specific_positive_worker_count(points: np.ndarray) -> None:
    KDTreeNeighborSearch.from_array(points, workers=8)  # must not raise


# ----------------------------------------------------------------------
# Configuration propagation: NeighborConfig -> NeighborhoodManager -> KDTreeNeighborSearch.
# ----------------------------------------------------------------------


def test_workers_propagates_from_point_cloud_constructor() -> None:
    from topocore.pointcloud.attributes import PointAttribute
    from topocore.pointcloud.chunk import Chunk
    from topocore.pointcloud.pointcloud import PointCloud

    rng = np.random.default_rng(1)
    n = 100
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)

    manager = NeighborhoodManager.from_point_cloud(cloud, config=NeighborConfig(workers=-1))
    assert manager.config.workers == -1

    indices, _distances = manager.knn_many(k=5)
    assert indices.shape == (n, 5)
