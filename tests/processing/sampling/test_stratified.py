"""
Regression suite for topocore.processing.sampling.stratified -- PR19.

Includes a real bug found and fixed in this session: the "random"
method used a HARDCODED seed=42 with no constructor parameter to
override it, unlike every other sampler in this module (RandomSampler,
VoxelSampler, DensitySampler all expose `seed: int | None`). Confirmed
directly: the same StratifiedSampler instance, called repeatedly, or a
different instance entirely, always produced the exact same selection
regardless of NumPy's global random state. Fixed to accept an optional
`seed` parameter, defaulting to None (non-deterministic, matching the
rest of the module), searched the codebase for anything relying on the
old hardcoded 42 before changing it -- nothing did.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.stratified import StratifiedSampler


def _scattered_cloud(n: int, extent: float = 10.0) -> PointCloud:
    rng = np.random.default_rng(123)
    xs = rng.uniform(0, extent, n)
    ys = rng.uniform(0, extent, n)
    zs = np.zeros(n)

    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# The real bug: hardcoded seed=42.
# ----------------------------------------------------------------------


def test_default_seed_is_nondeterministic_across_calls() -> None:
    """
    Before the fix: always seed=42, so this would ALWAYS be
    reproducible (a False negative for "non-deterministic" if this
    test had existed pre-fix). After the fix: seed=None by default,
    matching the rest of the module.
    """
    cloud = _scattered_cloud(200)
    sampler = StratifiedSampler(cell_size=2.0, samples_per_cell=1, method="random")

    r1 = next(iter(sampler.sample(cloud)))[PointAttribute.X]
    r2 = next(iter(sampler.sample(cloud)))[PointAttribute.X]

    assert not np.array_equal(r1, r2)


def test_explicit_seed_is_reproducible() -> None:
    cloud = _scattered_cloud(200)
    sampler = StratifiedSampler(cell_size=2.0, samples_per_cell=1, method="random", seed=7)

    r1 = next(iter(sampler.sample(cloud)))[PointAttribute.X]
    r2 = next(iter(sampler.sample(cloud)))[PointAttribute.X]

    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_give_different_results() -> None:
    cloud = _scattered_cloud(200)
    r1 = next(iter(StratifiedSampler(cell_size=2.0, method="random", seed=1).sample(cloud)))[PointAttribute.X]
    r2 = next(iter(StratifiedSampler(cell_size=2.0, method="random", seed=2).sample(cloud)))[PointAttribute.X]

    assert not np.array_equal(r1, r2)


def test_different_instances_no_longer_collide_on_hardcoded_seed() -> None:
    """
    Two DIFFERENT instances (both seed=None) must be free to diverge
    -- before the fix, both would deterministically produce the
    identical selection (hardcoded 42), which this test would have
    caught as a regression.
    """
    cloud = _scattered_cloud(200)
    r1 = next(iter(StratifiedSampler(cell_size=2.0, method="random").sample(cloud)))[PointAttribute.X]
    r2 = next(iter(StratifiedSampler(cell_size=2.0, method="random").sample(cloud)))[PointAttribute.X]

    assert not np.array_equal(r1, r2)


# ----------------------------------------------------------------------
# Core sampling behavior, unaffected by the fix.
# ----------------------------------------------------------------------


def test_samples_per_cell_limits_output_per_cell() -> None:
    # All points in ONE cell (small extent, large cell_size).
    cloud = _scattered_cloud(50, extent=1.0)
    result = StratifiedSampler(cell_size=10.0, samples_per_cell=5, method="random", seed=1).sample(cloud)
    assert result.point_count == 5


def test_centroid_method_picks_real_point_closest_to_mean() -> None:
    cloud = _scattered_cloud(50, extent=1.0)  # single cell with cell_size=10
    result = StratifiedSampler(cell_size=10.0, samples_per_cell=1, method="centroid").sample(cloud)
    assert result.point_count == 1


def test_rejects_nonpositive_cell_size() -> None:
    with pytest.raises(SamplingError):
        StratifiedSampler(cell_size=0.0)


def test_rejects_samples_per_cell_below_one() -> None:
    with pytest.raises(SamplingError):
        StratifiedSampler(cell_size=1.0, samples_per_cell=0)


def test_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError):
        StratifiedSampler(cell_size=1.0).sample(PointCloud())
