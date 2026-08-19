"""
Regression suite for topocore.processing.sampling.density -- PR19.

Verified: a dense cluster and a sparse cluster with target_density
calibrated between their actual measured densities -- the dense
cluster retains a meaningfully smaller fraction than the sparse one,
matching the class's own documented intent.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.density import DensitySampler


@pytest.fixture
def dense_and_sparse_cloud() -> PointCloud:
    rng = np.random.default_rng(0)
    dense_xs = rng.uniform(0, 1, 500)
    dense_ys = rng.uniform(0, 1, 500)
    sparse_xs = rng.uniform(100, 150, 50)
    sparse_ys = rng.uniform(100, 150, 50)

    xs = np.concatenate([dense_xs, sparse_xs])
    ys = np.concatenate([dense_ys, sparse_ys])
    zs = np.zeros_like(xs)

    cloud = PointCloud()
    chunk = Chunk(size=len(xs), attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud.add_chunk(chunk)
    return cloud


def test_dense_region_sampled_more_aggressively_than_sparse(
    dense_and_sparse_cloud: PointCloud,
) -> None:
    # target_density calibrated (session audit) between the measured
    # densities of the two clusters (~14.9 dense, ~0.033 sparse).
    sampler = DensitySampler(target_density=7.5, radius=2.0, min_fraction=0.05, max_fraction=1.0, seed=42)
    result = sampler.sample(dense_and_sparse_cloud)
    result_x = next(iter(result))[PointAttribute.X]

    dense_kept = int(np.sum(result_x < 10))
    sparse_kept = int(np.sum(result_x > 50))

    dense_fraction = dense_kept / 500
    sparse_fraction = sparse_kept / 50

    assert sparse_fraction > dense_fraction
    assert sparse_fraction == pytest.approx(1.0)  # sparse: always below target -> max_fraction


def test_min_fraction_and_max_fraction_bound_the_result(
    dense_and_sparse_cloud: PointCloud,
) -> None:
    # A target_density much LOWER than either cluster's actual density
    # (~14.9 dense, ~0.033 sparse) pushes density_ratio = target/density
    # toward zero for both -> clipped up to min_fraction for both.
    sampler = DensitySampler(target_density=0.001, radius=2.0, min_fraction=0.2, max_fraction=0.9, seed=1)
    result = sampler.sample(dense_and_sparse_cloud)
    # ~20% of 550 points (allow slack for the random Bernoulli-style draw).
    assert 0.1 * 550 < result.point_count < 0.35 * 550


def test_rejects_nonpositive_target_density() -> None:
    with pytest.raises(SamplingError):
        DensitySampler(target_density=0.0)


def test_rejects_nonpositive_radius() -> None:
    with pytest.raises(SamplingError):
        DensitySampler(target_density=1.0, radius=0.0)


def test_rejects_min_fraction_greater_than_max_fraction() -> None:
    with pytest.raises(SamplingError):
        DensitySampler(target_density=1.0, min_fraction=0.9, max_fraction=0.1)


def test_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError):
        DensitySampler(target_density=1.0).sample(PointCloud())
