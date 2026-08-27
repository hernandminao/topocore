"""
Regression suite for PR21.7.7: VoxelSampler and StratifiedSampler's
"random" method's documented contract -- see both classes' own
docstrings for the full rationale.

For method="random", sampling is performed over the complete logical
point set. The result is reproducible for a given seed and
INDEPENDENT of the physical chunk partitioning. This was confirmed to
already be the real, existing behavior BEFORE PR21.7.7 (an emergent
property of "random" still materializing the whole cloud, unlike
"centroid"/"closest" which were rewritten to a chunk-wise accumulator
in PR21.7.5/7.6) -- but it was previously undocumented and untested.
This suite formalizes it as a permanent, explicit guarantee: "random"
is deliberately NOT chunk-optimized, because a genuinely streaming
implementation (e.g. reservoir sampling) would necessarily make its
RNG consumption -- and therefore its selected points -- depend on how
the cloud happens to be chunked, silently breaking this real (if
previously undocumented) guarantee. This is a deliberate compatibility
decision, not unaddressed technical debt.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.sampling.stratified import StratifiedSampler
from topocore.processing.sampling.voxel import VoxelSampler


def _make_cloud(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, chunk_sizes: list[int]) -> PointCloud:
    cloud = PointCloud()
    offset = 0
    for size in chunk_sizes:
        chunk = Chunk(size=size, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
        chunk[PointAttribute.X][:] = xs[offset : offset + size]
        chunk[PointAttribute.Y][:] = ys[offset : offset + size]
        chunk[PointAttribute.Z][:] = zs[offset : offset + size]
        cloud.add_chunk(chunk)
        offset += size
    return cloud


def _sorted_x(cloud: PointCloud) -> list[float]:
    xs: list[float] = []
    for chunk in cloud:
        xs.extend(chunk[PointAttribute.X].tolist())
    return sorted(xs)


@pytest.fixture
def coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 200
    return rng.uniform(0, 50, n), rng.uniform(0, 50, n), rng.uniform(0, 5, n)


# ----------------------------------------------------------------------
# VoxelSampler "random" -- chunking invariance.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [200],  # 1 chunk
        [100, 100],  # 2 equal chunks
        [50, 50, 50, 50],  # 4 equal chunks
        [30, 90, 20, 60],  # uneven chunks
    ],
    ids=["1_chunk", "2_equal_chunks", "4_equal_chunks", "uneven_chunks"],
)
def test_voxel_random_selects_identical_points_regardless_of_chunking(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    chunk_sizes: list[int],
) -> None:
    xs, ys, zs = coordinates
    reference_cloud = _make_cloud(xs, ys, zs, [200])
    reference = _sorted_x(VoxelSampler(voxel_size=5.0, method="random", seed=42).sample(reference_cloud))

    cloud = _make_cloud(xs, ys, zs, chunk_sizes)
    result = _sorted_x(VoxelSampler(voxel_size=5.0, method="random", seed=42).sample(cloud))

    assert result == reference


def test_voxel_random_different_seeds_give_different_results(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Sanity check: the invariance above isn't because seed is ignored -- different seeds do change the outcome."""
    xs, ys, zs = coordinates
    cloud = _make_cloud(xs, ys, zs, [200])

    result_a = _sorted_x(VoxelSampler(voxel_size=5.0, method="random", seed=1).sample(cloud))
    result_b = _sorted_x(VoxelSampler(voxel_size=5.0, method="random", seed=2).sample(cloud))

    assert result_a != result_b


# ----------------------------------------------------------------------
# StratifiedSampler "random" -- chunking invariance.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [200],
        [100, 100],
        [50, 50, 50, 50],
        [30, 90, 20, 60],
    ],
    ids=["1_chunk", "2_equal_chunks", "4_equal_chunks", "uneven_chunks"],
)
def test_stratified_random_selects_identical_points_regardless_of_chunking(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    chunk_sizes: list[int],
) -> None:
    xs, ys, zs = coordinates
    reference_cloud = _make_cloud(xs, ys, zs, [200])
    reference = _sorted_x(StratifiedSampler(cell_size=5.0, method="random", seed=42).sample(reference_cloud))

    cloud = _make_cloud(xs, ys, zs, chunk_sizes)
    result = _sorted_x(StratifiedSampler(cell_size=5.0, method="random", seed=42).sample(cloud))

    assert result == reference


def test_stratified_random_different_seeds_give_different_results(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xs, ys, zs = coordinates
    cloud = _make_cloud(xs, ys, zs, [200])

    result_a = _sorted_x(StratifiedSampler(cell_size=5.0, method="random", seed=1).sample(cloud))
    result_b = _sorted_x(StratifiedSampler(cell_size=5.0, method="random", seed=2).sample(cloud))

    assert result_a != result_b


def test_stratified_random_multi_sample_per_cell_also_chunk_invariant(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Confirms the invariance holds with samples_per_cell > 1 too, not just the default of 1."""
    xs, ys, zs = coordinates
    reference_cloud = _make_cloud(xs, ys, zs, [200])
    reference = _sorted_x(
        StratifiedSampler(cell_size=5.0, samples_per_cell=3, method="random", seed=7).sample(reference_cloud)
    )

    cloud = _make_cloud(xs, ys, zs, [30, 90, 20, 60])
    result = _sorted_x(StratifiedSampler(cell_size=5.0, samples_per_cell=3, method="random", seed=7).sample(cloud))

    assert result == reference
