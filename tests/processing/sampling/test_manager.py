"""
Regression suite for topocore.processing.sampling.manager.SamplingManager
-- PR19.

Includes a real bug found and fixed in this session:
_create_sampler() never passed `seed` through to VoxelSampler
(pre-existing) or StratifiedSampler (a direct consequence of adding
`seed` to StratifiedSampler in this same session) -- both accept a
`seed` parameter directly, but a caller going through SamplingManager
had it silently dropped. Confirmed directly: seed=7 via the manager
did not make voxel/stratified "random" sampling reproducible before
the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.manager import SamplingManager


def _cloud(n: int) -> PointCloud:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = np.zeros(n)

    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# The real bug: seed silently dropped for voxel/stratified.
# ----------------------------------------------------------------------


def test_voxel_seed_is_respected_through_manager() -> None:
    manager = SamplingManager(method="voxel")
    cloud = _cloud(50)

    r1 = next(iter(manager.sample(cloud, voxel_size=5.0, method="random", seed=7)))[PointAttribute.X]
    r2 = next(iter(manager.sample(cloud, voxel_size=5.0, method="random", seed=7)))[PointAttribute.X]

    np.testing.assert_array_equal(r1, r2)


def test_voxel_different_seed_gives_different_result_through_manager() -> None:
    manager = SamplingManager(method="voxel")
    cloud = _cloud(50)

    r1 = next(iter(manager.sample(cloud, voxel_size=5.0, method="random", seed=7)))[PointAttribute.X]
    r2 = next(iter(manager.sample(cloud, voxel_size=5.0, method="random", seed=99)))[PointAttribute.X]

    assert not np.array_equal(r1, r2)


def test_stratified_seed_is_respected_through_manager() -> None:
    manager = SamplingManager(method="stratified")
    cloud = _cloud(50)

    r1 = next(iter(manager.sample(cloud, cell_size=5.0, method="random", seed=7)))[PointAttribute.X]
    r2 = next(iter(manager.sample(cloud, cell_size=5.0, method="random", seed=7)))[PointAttribute.X]

    np.testing.assert_array_equal(r1, r2)


# ----------------------------------------------------------------------
# Method selection / dispatch, unaffected by the fix.
# ----------------------------------------------------------------------


def test_random_method_dispatch() -> None:
    manager = SamplingManager(method="random")
    result = manager.sample(_cloud(100), fraction=0.2, seed=1)
    assert result.point_count == 20


def test_uniform_method_dispatch() -> None:
    manager = SamplingManager(method="uniform")
    result = manager.sample(_cloud(100), step=10)
    assert result.point_count == 10


def test_method_setter_switches_method() -> None:
    manager = SamplingManager(method="random")
    manager.method = "uniform"
    result = manager.sample(_cloud(100), step=20)
    assert result.point_count == 5


def test_set_params_persists_across_calls() -> None:
    manager = SamplingManager(method="random")
    manager.set_params(fraction=0.1, seed=1)

    result = manager.sample(_cloud(100))
    assert result.point_count == 10


def test_rejects_unsupported_method_at_construction() -> None:
    with pytest.raises(SamplingError):
        SamplingManager(method="bogus")


def test_rejects_unsupported_method_via_setter() -> None:
    manager = SamplingManager(method="random")
    with pytest.raises(SamplingError):
        manager.method = "bogus"
