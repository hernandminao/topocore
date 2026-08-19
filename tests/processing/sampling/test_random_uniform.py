"""
Regression suite for topocore.processing.sampling.random and
.uniform -- PR19.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.random import RandomSampler
from topocore.processing.sampling.uniform import UniformSampler


def _cloud(n: int) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [float(i) for i in range(n)]
    chunk[PointAttribute.Y][:] = [0.0] * n
    chunk[PointAttribute.Z][:] = [0.0] * n
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# RandomSampler
# ----------------------------------------------------------------------


def test_random_fraction_keeps_expected_count() -> None:
    result = RandomSampler(fraction=0.5, seed=1).sample(_cloud(100))
    assert result.point_count == 50


def test_random_count_keeps_exact_count() -> None:
    result = RandomSampler(count=10, seed=1).sample(_cloud(100))
    assert result.point_count == 10


def test_random_count_exceeding_total_keeps_all() -> None:
    result = RandomSampler(count=1000, seed=1).sample(_cloud(100))
    assert result.point_count == 100


def test_random_same_seed_reproducible() -> None:
    cloud = _cloud(100)
    r1 = RandomSampler(fraction=0.3, seed=5).sample(cloud)
    r2 = RandomSampler(fraction=0.3, seed=5).sample(cloud)
    np.testing.assert_array_equal(next(iter(r1))[PointAttribute.X], next(iter(r2))[PointAttribute.X])


def test_random_different_seed_differs() -> None:
    cloud = _cloud(100)
    r1 = RandomSampler(fraction=0.3, seed=5).sample(cloud)
    r2 = RandomSampler(fraction=0.3, seed=6).sample(cloud)
    assert not np.array_equal(next(iter(r1))[PointAttribute.X], next(iter(r2))[PointAttribute.X])


def test_random_rejects_both_fraction_and_count() -> None:
    with pytest.raises(SamplingError):
        RandomSampler(fraction=0.5, count=10)


def test_random_rejects_neither_fraction_nor_count() -> None:
    with pytest.raises(SamplingError):
        RandomSampler()


def test_random_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError):
        RandomSampler(fraction=0.5).sample(PointCloud())


# ----------------------------------------------------------------------
# UniformSampler
# ----------------------------------------------------------------------


def test_uniform_keeps_every_nth_point() -> None:
    result = UniformSampler(step=10).sample(_cloud(100))
    result_x = next(iter(result))[PointAttribute.X]
    np.testing.assert_array_equal(result_x, [float(i) for i in range(0, 100, 10)])


def test_uniform_offset_shifts_selection() -> None:
    result = UniformSampler(step=10, offset=3).sample(_cloud(100))
    result_x = next(iter(result))[PointAttribute.X]
    np.testing.assert_array_equal(result_x, [float(i) for i in range(3, 100, 10)])


def test_uniform_rejects_step_below_one() -> None:
    with pytest.raises(SamplingError):
        UniformSampler(step=0)


def test_uniform_rejects_offset_out_of_range() -> None:
    with pytest.raises(SamplingError):
        UniformSampler(step=5, offset=5)
