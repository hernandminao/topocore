"""
Regression suite for PR21.4: NeighborhoodManager.local_density_many(),
added after profiling RuleBasedClassifier.classify() found a genuine
Python-level hot loop -- one local_density(i, radius=...) call per
point -- accounting for 36% of total classify() time on a
100,000-point cloud. This suite verifies the batched replacement
gives numerically IDENTICAL results to the original per-point loop,
plus its cache-population contract (a later single-point
local_density() call for an index already covered by
local_density_many() must correctly hit the cache).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.processing.neighbors.manager import NeighborhoodManager


@pytest.fixture
def points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 100, (2000, 3))


def test_matches_per_point_loop_exactly(points: np.ndarray) -> None:
    """The decisive check: the exact regression this method was introduced to fix the cost of."""
    manager_loop = NeighborhoodManager.from_array(points)
    density_loop = np.array([manager_loop.local_density(i, radius=1.5) for i in range(len(points))])

    manager_batch = NeighborhoodManager.from_array(points)
    density_batch = manager_batch.local_density_many(radius=1.5)

    np.testing.assert_array_equal(density_loop, density_batch)


def test_populates_the_shared_density_cache(points: np.ndarray) -> None:
    """A later single-point local_density() call for an already-covered index must hit the cache, not recompute."""
    manager = NeighborhoodManager.from_array(points)
    batch_result = manager.local_density_many(radius=1.5)

    single_call_result = manager.local_density(100, radius=1.5)

    assert single_call_result == batch_result[100]
    assert (100, 1.5) in manager._density_cache


def test_subset_of_indices_matches_full_batch(points: np.ndarray) -> None:
    manager = NeighborhoodManager.from_array(points)
    full_batch = manager.local_density_many(radius=1.5)

    subset_indices = np.array([5, 10, 15, 200], dtype=np.int64)
    manager_subset = NeighborhoodManager.from_array(points)
    subset_result = manager_subset.local_density_many(subset_indices, radius=1.5)

    np.testing.assert_array_equal(subset_result, full_batch[subset_indices])


def test_default_radius_matches_config(points: np.ndarray) -> None:
    manager = NeighborhoodManager.from_array(points)
    default_result = manager.local_density_many()
    explicit_result = manager.local_density_many(radius=manager.config.radius_default)

    np.testing.assert_array_equal(default_result, explicit_result)


def test_returns_one_value_per_query_index(points: np.ndarray) -> None:
    manager = NeighborhoodManager.from_array(points)
    result = manager.local_density_many(radius=2.0)
    assert result.shape == (len(points),)
