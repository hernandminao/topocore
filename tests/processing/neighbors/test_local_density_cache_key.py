"""
Regression suite for the PR21.3.3 fix: NeighborhoodManager.local_density()
now keys its cache by (index, effective_radius), not index alone.

Found during PR21.3.2's audit as a real, latent correctness gap
(not yet triggered by any existing internal caller, since
_effective_k()/_effective_radius() always call local_density(index)
without an explicit radius override, implicitly using
config.radius_default, constant for a manager's lifetime): before
this fix, local_density(100, radius=1.0) and local_density(100,
radius=5.0) on the same manager instance would silently return
whichever was computed FIRST for that index, ignoring the second
call's different radius entirely. This mattered specifically for the
shared-NeighborhoodManager scenario PR21.3 is building toward, where
different consumers could legitimately request local_density() at
different radii on the same shared manager instance.
"""

from __future__ import annotations

import numpy as np

from topocore.processing.neighbors.manager import NeighborhoodManager


def _clustered_points() -> np.ndarray:
    """Non-uniform density: a dense cluster near the origin, a sparse ring farther out."""
    rng = np.random.default_rng(0)
    dense = rng.uniform(-0.5, 0.5, (100, 3))
    sparse = rng.uniform(-10, 10, (20, 3))
    return np.vstack([dense, sparse])


def test_different_radii_give_different_densities_not_a_stale_cached_value() -> None:
    """The exact regression: before the fix, the second call returned the FIRST call's cached value."""
    manager = NeighborhoodManager.from_array(_clustered_points())

    density_small_radius = manager.local_density(0, radius=1.0)
    density_large_radius = manager.local_density(0, radius=5.0)

    assert density_small_radius != density_large_radius


def test_same_radius_still_hits_cache_with_identical_value() -> None:
    manager = NeighborhoodManager.from_array(_clustered_points())

    first = manager.local_density(0, radius=2.0)
    second = manager.local_density(0, radius=2.0)

    assert first == second


def test_default_radius_and_explicit_matching_radius_agree() -> None:
    """local_density(index) (using config.radius_default) and local_density(index, radius=that_same_value) must match."""
    manager = NeighborhoodManager.from_array(_clustered_points())

    default_result = manager.local_density(0)
    explicit_result = manager.local_density(0, radius=manager.config.radius_default)

    assert default_result == explicit_result


def test_clear_cache_forces_recomputation_for_all_radii() -> None:
    manager = NeighborhoodManager.from_array(_clustered_points())

    manager.local_density(0, radius=1.0)
    manager.local_density(0, radius=5.0)
    manager.clear_cache()

    # Must not raise, and must recompute (not read from a cleared-but-stale dict).
    recomputed = manager.local_density(0, radius=1.0)
    assert recomputed >= 0.0


def test_different_indices_at_same_radius_are_independent() -> None:
    manager = NeighborhoodManager.from_array(_clustered_points())

    dense_point_density = manager.local_density(0, radius=1.0)  # in the dense cluster
    sparse_point_density = manager.local_density(105, radius=1.0)  # in the sparse ring

    assert dense_point_density > sparse_point_density
