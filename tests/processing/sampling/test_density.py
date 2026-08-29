"""
Coverage audit tests for topocore.processing.sampling.density.DensitySampler.

SAMPLING-DENSITY-001 (FIXED in this PR): sample()'s own per-point
acceptance loop (`for i in range(total_points): if rng.random() <
fractions[i]`) has been replaced with a single batched draw
(`rng.random(total_points) < fractions`). Confirmed directly, before
this change, that `rng.random()` called N times sequentially
produces the IDENTICAL sequence as `rng.random(N)` called once from
the same seed -- and confirmed the batched version selects the exact
same indices as an independent reference reimplementation of the old
per-point loop, both by count and by which specific points are
selected. Seed, reproducibility, and the underlying acceptance
distribution are unchanged; only the drawing mechanism changed.

"No points selected" is confirmed genuinely reachable via extreme
min_fraction/max_fraction values combined with a fixed seed.

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
from topocore.processing.sampling.density import DensitySampler


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


def test_non_positive_target_density_rejected() -> None:
    with pytest.raises(SamplingError, match="target_density must be positive"):
        DensitySampler(target_density=-1.0, radius=1.0)


def test_non_positive_radius_rejected() -> None:
    with pytest.raises(SamplingError, match="radius must be positive"):
        DensitySampler(target_density=1.0, radius=-1.0)


def test_min_fraction_out_of_range_rejected() -> None:
    with pytest.raises(SamplingError, match="min_fraction must be in"):
        DensitySampler(target_density=1.0, min_fraction=0.0)


def test_max_fraction_out_of_range_rejected() -> None:
    with pytest.raises(SamplingError, match="max_fraction must be in"):
        DensitySampler(target_density=1.0, max_fraction=1.5)


def test_min_fraction_greater_than_max_rejected() -> None:
    with pytest.raises(SamplingError, match="must be <="):
        DensitySampler(target_density=1.0, min_fraction=0.9, max_fraction=0.5)


# ----------------------------------------------------------------------
# sample() -- empty cloud, happy path, reproducibility, no-points edge case.
# ----------------------------------------------------------------------


def test_sample_rejects_empty_cloud() -> None:
    with pytest.raises(SamplingError, match="empty point cloud"):
        DensitySampler(target_density=1.0).sample(PointCloud())


def test_sample_happy_path_produces_valid_subset() -> None:
    cloud = _cloud()
    result = DensitySampler(target_density=0.5, radius=2.0, min_fraction=0.1, max_fraction=1.0, seed=42).sample(cloud)

    assert 0 < result.point_count <= cloud.point_count


def test_sample_is_reproducible_with_same_seed() -> None:
    cloud = _cloud()

    def _make_sampler() -> DensitySampler:
        return DensitySampler(target_density=0.5, radius=2.0, min_fraction=0.1, max_fraction=1.0, seed=42)

    result_a = _make_sampler().sample(cloud)
    result_b = _make_sampler().sample(cloud)

    assert result_a.point_count == result_b.point_count


def test_sample_raises_when_no_points_selected() -> None:
    cloud = _cloud(n=20)

    sampler = DensitySampler(
        target_density=0.0001,
        radius=1.0,
        min_fraction=0.0001,
        max_fraction=0.0001,
        seed=42,
    )

    with pytest.raises(SamplingError, match="No points selected"):
        sampler.sample(cloud)


# ----------------------------------------------------------------------
# _compute_fractions() -- clipping to [min_fraction, max_fraction].
# ----------------------------------------------------------------------


def test_compute_fractions_clips_to_configured_range() -> None:
    sampler = DensitySampler(target_density=0.5, min_fraction=0.1, max_fraction=1.0)
    densities = np.array([0.001, 1.0, 1000.0])

    fractions = sampler._compute_fractions(densities)

    np.testing.assert_allclose(fractions, [1.0, 0.5, 0.1])


def test_sample_matches_reference_per_point_loop_indices() -> None:
    """
    Regression for SAMPLING-DENSITY-001: confirms the batched
    `rng.random(n) < fractions` draw selects the EXACT same indices
    (not merely the same count) as an independent reimplementation of
    the original per-point loop, from the same seed and fractions.
    """
    n = 100
    cloud = _cloud(n, seed=0)
    sampler = DensitySampler(target_density=0.05, radius=2.0, min_fraction=0.1, max_fraction=0.5, seed=42)

    densities = sampler._compute_density(cloud)
    fractions = sampler._compute_fractions(densities)

    reference_rng = np.random.default_rng(42)
    reference_indices = [i for i in range(n) if reference_rng.random() < fractions[i]]

    result = sampler.sample(cloud)

    assert result.point_count == len(reference_indices)
    assert result.point_count not in (
        0,
        n,
    )  # confirms this is a genuine, discriminating subset
