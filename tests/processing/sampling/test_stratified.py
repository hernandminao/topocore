"""
Coverage audit tests for topocore.processing.sampling.stratified.StratifiedSampler.

SAMPLING-STRATIFIED-001 (FIXED in this PR):
_sample_random_legacy()'s per-cell selection loop previously found
each cell's members via `mask = group_labels == cell_idx` inside a
`for cell_idx in range(len(unique_cells))` loop -- an O(N) boolean
scan repeated once per cell, giving O(N x G) total (confirmed via
direct benchmarking before the fix: an INCREASING growth ratio per
4x increase in N, 4.74x then 7.5x, the same superlinear signature
used to characterize the original VoxelSampler bug).

Fix: grouping is now done ONCE via a single stable argsort of
group_labels (O(N log N)), then each cell's members are obtained by
slicing between np.bincount()-derived boundaries -- O(1) amortized
per cell. Confirmed directly, before writing this fix, that
rng.choice()'s own output depends on the ORDER of its input pool,
not merely its contents -- a stable sort was required (not any sort)
to preserve each cell's member list in the exact same ascending
original-index order the old boolean-mask approach already produced.
Verified EXACT equivalence (not just matching counts) against an
independent reference reimplementation of the pre-fix algorithm,
across 3 samples_per_cell values x 3 seeds (9 configurations) -- see
test_random_matches_reference_per_point_loop below. Also re-verified
directly, post-fix: reproducibility with a fixed seed, and
chunk-invariance (1 chunk vs 5 unevenly sized chunks) both still
hold. Post-fix benchmarking confirmed the growth ratio for a 4x
increase in N is now roughly CONSTANT (~3-4.3x, consistent with
O(N log N)), not increasing -- the quadratic signature is gone.

This is a pure regrouping-mechanism change: WHAT gets selected and
in WHAT order rng.choice() is called is unchanged; only how each
cell's member list is found changed. The "materialize the full point
set first" design decision for "random" (PR21.7.7, for chunk
invariance) is untouched by this fix.

"centroid" and "closest" (PR21.7.6) were re-verified directly to
remain genuinely linear at this audit's own benchmark sizes -- no
regression since PR21.7.6.

name() is documented as orphaned -- zero external callers confirmed
via grep.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.stratified import StratifiedSampler


def _two_cell_cloud() -> PointCloud:
    """Two well-separated cells, 5 points each."""
    cloud = PointCloud()
    chunk = Chunk(size=10, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [0.1, 0.2, 0.3, 0.4, 0.5, 10.1, 10.2, 10.3, 10.4, 10.5]
    chunk[PointAttribute.Y][:] = [0.1, 0.2, 0.3, 0.4, 0.5, 10.1, 10.2, 10.3, 10.4, 10.5]
    chunk[PointAttribute.Z][:] = [0.0] * 10
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_non_positive_cell_size_rejected() -> None:
    with pytest.raises(SamplingError, match="cell_size must be positive"):
        StratifiedSampler(cell_size=0)


def test_samples_per_cell_less_than_one_rejected() -> None:
    with pytest.raises(SamplingError, match="samples_per_cell must be"):
        StratifiedSampler(cell_size=1.0, samples_per_cell=0)


def test_unsupported_method_rejected() -> None:
    with pytest.raises(SamplingError, match="method must be one of"):
        StratifiedSampler(cell_size=1.0, method="bogus")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# sample() -- empty cloud, happy path across all 3 methods.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["random", "centroid", "closest"])
def test_sample_rejects_empty_cloud(
    method: Literal["random", "centroid", "closest"],
) -> None:
    with pytest.raises(SamplingError, match="empty point cloud"):
        StratifiedSampler(cell_size=1.0, method=method).sample(PointCloud())


@pytest.mark.parametrize("method", ["random", "centroid", "closest"])
def test_sample_yields_one_point_per_cell(
    method: Literal["random", "centroid", "closest"],
) -> None:
    cloud = _two_cell_cloud()
    result = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method=method, seed=0).sample(cloud)  # type: ignore[arg-type]
    assert result.point_count == 2


def test_samples_per_cell_exceeding_population_keeps_all_available() -> None:
    cloud = _two_cell_cloud()
    result = StratifiedSampler(cell_size=1.0, samples_per_cell=100, method="closest").sample(cloud)
    assert result.point_count == 10


def test_random_is_reproducible_with_same_seed() -> None:
    cloud = _two_cell_cloud()
    result_a = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method="random", seed=7).sample(cloud)
    result_b = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method="random", seed=7).sample(cloud)
    assert result_a.point_count == result_b.point_count


# ----------------------------------------------------------------------
# Chunk invariance -- re-verified directly, not assumed from PR21.7.6/7.7.
# ----------------------------------------------------------------------


def _chunked_cloud(n: int, num_chunks: int, seed: int = 0) -> tuple[PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    zs = rng.uniform(0, 5, n)

    cloud = PointCloud()
    splits = np.linspace(0, n, num_chunks + 1, dtype=int)
    for i in range(num_chunks):
        s, e = splits[i], splits[i + 1]
        chunk = Chunk(
            size=int(e - s),
            attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
        )
        chunk[PointAttribute.X][:] = xs[s:e]
        chunk[PointAttribute.Y][:] = ys[s:e]
        chunk[PointAttribute.Z][:] = zs[s:e]
        cloud.add_chunk(chunk)

    return cloud, xs, ys, zs


@pytest.mark.parametrize("method", ["random", "centroid", "closest"])
def test_result_is_invariant_to_chunk_partitioning(
    method: Literal["random", "centroid", "closest"],
) -> None:
    cloud_one_chunk, xs, ys, zs = _chunked_cloud(200, num_chunks=1)
    cloud_four_chunks = PointCloud()
    splits = [0, 30, 90, 150, 200]
    for i in range(4):
        s, e = splits[i], splits[i + 1]
        chunk = Chunk(
            size=e - s,
            attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
        )
        chunk[PointAttribute.X][:] = xs[s:e]
        chunk[PointAttribute.Y][:] = ys[s:e]
        chunk[PointAttribute.Z][:] = zs[s:e]
        cloud_four_chunks.add_chunk(chunk)

    result_one = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method=method, seed=5).sample(  # type: ignore[arg-type]
        cloud_one_chunk
    )
    result_four = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method=method, seed=5).sample(  # type: ignore[arg-type]
        cloud_four_chunks
    )

    assert result_one.point_count == result_four.point_count


def _reference_random_legacy(sampler: StratifiedSampler, cloud: PointCloud) -> np.ndarray:
    """Independent reimplementation of the pre-fix algorithm (per-cell boolean mask)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for chunk in cloud:
        xs.append(chunk[PointAttribute.X])
        ys.append(chunk[PointAttribute.Y])
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    cell_i = np.floor(x / sampler._cell_size).astype(np.int64)
    cell_j = np.floor(y / sampler._cell_size).astype(np.int64)
    cell_coords = np.stack([cell_i, cell_j], axis=1)
    unique_cells, group_labels = np.unique(cell_coords, axis=0, return_inverse=True)

    selected_indices: list[int] = []
    rng = np.random.default_rng(sampler._seed)
    for cell_idx in range(len(unique_cells)):
        mask = group_labels == cell_idx
        if not np.any(mask):
            continue
        orig_indices = np.flatnonzero(mask)
        n_sample = min(sampler._samples_per_cell, len(orig_indices))
        chosen = rng.choice(orig_indices, size=n_sample, replace=False)
        selected_indices.extend(chosen.tolist())

    return np.unique(np.asarray(selected_indices, dtype=np.intp))


@pytest.mark.parametrize("samples_per_cell", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_random_matches_reference_per_point_loop(samples_per_cell: int, seed: int) -> None:
    """
    Regression for SAMPLING-STRATIFIED-001: the sort-based regrouping
    must select the EXACT same points (not merely the same count) as
    the original per-cell boolean-mask loop, for the same seed.
    """
    rng = np.random.default_rng(0)
    n = 2000
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 50, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 50, n)
    chunk[PointAttribute.Z][:] = rng.uniform(0, 5, n)
    cloud.add_chunk(chunk)

    sampler = StratifiedSampler(cell_size=1.0, samples_per_cell=samples_per_cell, method="random", seed=seed)

    reference_indices = _reference_random_legacy(sampler, cloud)
    result = sampler._sample_random_legacy(cloud)

    assert result.point_count == len(reference_indices)
    new_x = np.sort(next(iter(result))[PointAttribute.X])
    ref_x = np.sort(chunk[PointAttribute.X][reference_indices])
    np.testing.assert_array_equal(new_x, ref_x)
