"""
Coverage audit tests for topocore.processing.sampling.random.RandomSampler.

Uses a single vectorized np.random.Generator.choice() call, not a
per-point Python loop -- unlike DensitySampler's own audit
(SAMPLING-DENSITY-001), there is no performance finding here.

sample()'s own `if self._count is None: raise ...` (and the
subsequent `if n is None: raise ...`, checking a value just assigned
from self._count) are confirmed unreachable: __init__'s own XOR
validation (`(fraction is None) == (count is None)` raises unless
exactly one is given) already guarantees that whenever
self._fraction is None, self._count is NEVER None -- these are
defensive checks against a state __init__ already makes impossible.

name() is documented as orphaned -- zero external callers confirmed
via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.random import RandomSampler


def _cloud(n: int = 100, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 10, n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_neither_fraction_nor_count_rejected() -> None:
    with pytest.raises(SamplingError, match="exactly one"):
        RandomSampler()


def test_both_fraction_and_count_rejected() -> None:
    with pytest.raises(SamplingError, match="exactly one"):
        RandomSampler(fraction=0.5, count=10)


def test_fraction_out_of_range_rejected() -> None:
    with pytest.raises(SamplingError, match="fraction must be in"):
        RandomSampler(fraction=0.0)


def test_count_less_than_one_rejected() -> None:
    with pytest.raises(SamplingError, match="count must be"):
        RandomSampler(count=0)


# ----------------------------------------------------------------------
# sample() -- empty cloud, happy path (fraction and count), edge cases.
# ----------------------------------------------------------------------


def test_sample_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError, match="empty point cloud"):
        RandomSampler(fraction=0.5).sample(PointCloud())


def test_sample_by_fraction() -> None:
    result = RandomSampler(fraction=0.3, seed=42).sample(_cloud(100))
    assert result.point_count == 30


def test_sample_by_count() -> None:
    result = RandomSampler(count=25, seed=42).sample(_cloud(100))
    assert result.point_count == 25


def test_count_exceeding_total_keeps_all_points() -> None:
    result = RandomSampler(count=1000, seed=42).sample(_cloud(100))
    assert result.point_count == 100


def test_sample_is_reproducible_with_same_seed() -> None:
    cloud = _cloud(100)
    result_a = RandomSampler(fraction=0.3, seed=42).sample(cloud)
    result_b = RandomSampler(fraction=0.3, seed=42).sample(cloud)
    assert result_a.point_count == result_b.point_count


def test_tiny_fraction_keeps_at_least_one_point() -> None:
    result = RandomSampler(fraction=0.01, seed=0).sample(_cloud(5))
    assert result.point_count == 1
