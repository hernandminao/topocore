"""
Regression suite for PR21.7.5: VoxelSampler's "centroid" and
"closest" methods, rewritten to use a chunk-wise accumulator instead
of concatenating every chunk's X/Y/Z into one global array and then
looping `for i in range(n_groups): mask = inverse == i`.

That loop was a genuine, severe, PRE-EXISTING complexity bug --
confirmed via direct benchmarking to be effectively quadratic for
realistic data (n_groups scales with N for typical point clouds):
20,000 points took 1.267s; 100,000 points took 28.66s (a ~22.6x
increase for a 5x increase in N -- consistent with O(N^2), not
O(N x G) with G roughly constant). This was independent of the
chunking/memory concern PR21.7 was originally scoped around, and is
fixed by the SAME accumulator design: a single vectorized pass per
chunk (np.unique + np.add.at + np.bincount for centroid; a per-point
distance comparison against a running best-per-voxel for closest),
merged into a global dict keyed by voxel coordinate.

Measured improvement (see benchmarks/benchmark_voxel_sampler.py):
100,000 points, centroid: 28.66s -> 0.317s (~90x). 100,000 points,
closest: 25.6s -> 0.127s (~200x). Honest trade-off noted: the
dict-based accumulator carries more per-voxel-group memory overhead
than the old single-merged-array approach, so peak memory at these
same sizes is HIGHER, not lower, than before this change -- this PR
fixes the catastrophic time complexity, not memory; see PR21.7.5's
own audit notes for why memory was a secondary concern for this
specific bug.

The decisive correctness property verified throughout: a single
voxel whose member points are split across MULTIPLE chunks, in
various arrangements, must produce the exact same result as if all
points had been in one chunk -- and "closest"'s tie-breaking (ties
broken by GLOBAL point index order, matching the pre-PR21.7.5
np.argmin "first occurrence wins" behavior) must not depend on chunk
boundaries either.

"random" and "all" are explicitly NOT touched by the accumulator
work: "random" would require reservoir sampling (a genuine
algorithmic change affecting RNG-reproducibility semantics, not
addressed here); "all" already returned the exact same PointCloud
instance unmodified (confirmed directly before this change: `result
is cloud` was True) -- its dispatch was simply moved earlier to skip
now-wasted work, changing nothing observable.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.voxel import VoxelSampler


def _chunk(xs: list[float], ys: list[float], zs: list[float]) -> Chunk:
    n = len(xs)
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    return chunk


def _single_chunk_xyz(cloud: PointCloud) -> np.ndarray:
    (chunk,) = list(cloud)
    return np.column_stack([chunk[PointAttribute.X], chunk[PointAttribute.Y], chunk[PointAttribute.Z]])


# ----------------------------------------------------------------------
# Centroid -- cross-chunk-boundary merging, the decisive property.
# ----------------------------------------------------------------------


def test_centroid_merges_a_single_voxel_split_across_three_chunks() -> None:
    """5 points, all in the SAME voxel, spread across 3 chunks (2+1+2) -- must merge into 1 centroid."""
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 2.0], [1.0, 2.0], [1.0, 2.0]))
    cloud.add_chunk(_chunk([3.0], [3.0], [3.0]))
    cloud.add_chunk(_chunk([4.0, 5.0], [4.0, 5.0], [4.0, 5.0]))

    result = VoxelSampler(voxel_size=10.0, method="centroid").sample(cloud)

    assert result.point_count == 1
    xyz = _single_chunk_xyz(result)
    np.testing.assert_allclose(xyz[0], [3.0, 3.0, 3.0])  # mean of 1..5


def test_centroid_merges_three_voxels_arbitrarily_split_across_chunks() -> None:
    """
    Voxels A, B, C distributed as chunk1->{A,B}, chunk2->{B,C},
    chunk3->{A,C} -- confirming arbitrary cross-chunk membership
    patterns merge correctly, not just a single voxel split evenly.
    """
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0], [1.0, 11.0], [1.0, 11.0]))  # A, B
    cloud.add_chunk(_chunk([12.0, 21.0], [12.0, 21.0], [12.0, 21.0]))  # B, C
    cloud.add_chunk(_chunk([2.0, 22.0], [2.0, 22.0], [2.0, 22.0]))  # A, C

    result = VoxelSampler(voxel_size=10.0, method="centroid").sample(cloud)

    assert result.point_count == 3
    xs = sorted(_single_chunk_xyz(result)[:, 0])
    np.testing.assert_allclose(xs, [1.5, 11.5, 21.5])  # (1+2)/2, (11+12)/2, (21+22)/2


def test_centroid_matches_reference_algorithm_on_random_data() -> None:
    """Cross-check against the pre-PR21.7.5 algorithm's own helper functions, at a size small enough to compare."""
    from topocore.processing.sampling.voxel import (
        _compute_voxel_centroid,
        _voxel_indices,
    )

    rng = np.random.default_rng(0)
    n = 2000
    xs, ys, zs = rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(0, 20, n)

    cloud = PointCloud()
    for start, end in ((0, 700), (700, 1300), (1300, 2000)):
        cloud.add_chunk(_chunk(list(xs[start:end]), list(ys[start:end]), list(zs[start:end])))

    voxel_size = 5.0
    new_result = VoxelSampler(voxel_size=voxel_size, method="centroid").sample(cloud)
    new_centroids = _single_chunk_xyz(new_result)

    points = np.column_stack([xs, ys, zs])
    voxel_i, voxel_j, voxel_k = _voxel_indices(points[:, 0], points[:, 1], points[:, 2], voxel_size)
    coords = np.column_stack((voxel_i, voxel_j, voxel_k))
    _unique_groups, labels = np.unique(coords, axis=0, return_inverse=True)
    old_centroids, _counts = _compute_voxel_centroid(points, labels)

    new_sorted = new_centroids[np.lexsort(new_centroids.T[::-1])]
    old_sorted = old_centroids[np.lexsort(old_centroids.T[::-1])]
    np.testing.assert_allclose(new_sorted, old_sorted)


# ----------------------------------------------------------------------
# Closest -- cross-chunk merging and strict tie-breaking.
# ----------------------------------------------------------------------


def test_closest_merges_across_chunks_correctly() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0], [1.0, 11.0], [1.0, 11.0]))
    cloud.add_chunk(_chunk([12.0, 21.0], [12.0, 21.0], [12.0, 21.0]))
    cloud.add_chunk(_chunk([2.0, 22.0], [2.0, 22.0], [2.0, 22.0]))

    result = VoxelSampler(voxel_size=10.0, method="closest").sample(cloud)
    assert result.point_count == 3


def test_closest_tie_break_is_strict_and_global_order_independent_of_chunking() -> None:
    """
    The decisive tie-break check: two points EXACTLY equidistant from
    the voxel center, in DIFFERENT chunks -- must resolve to the
    FIRST one by GLOBAL point index, matching pre-PR21.7.5
    np.argmin's "first occurrence wins" on exact ties.
    """
    cloud = PointCloud()
    cloud.add_chunk(_chunk([4.0], [5.0], [5.0]))  # distance 1.0 from center (5,5,5); global index 0
    cloud.add_chunk(_chunk([6.0], [5.0], [5.0]))  # distance 1.0 from center; global index 1 -- EXACT TIE

    result = VoxelSampler(voxel_size=10.0, method="closest").sample(cloud)

    assert _single_chunk_xyz(result)[0, 0] == pytest.approx(4.0)  # the earlier one, not the later tie


def test_closest_matches_reference_algorithm_on_random_data() -> None:
    from topocore.processing.sampling.voxel import (
        _compute_voxel_closest,
        _voxel_indices,
    )

    rng = np.random.default_rng(1)
    n = 2000
    xs, ys, zs = rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(0, 20, n)

    cloud = PointCloud()
    for start, end in ((0, 700), (700, 1300), (1300, 2000)):
        cloud.add_chunk(_chunk(list(xs[start:end]), list(ys[start:end]), list(zs[start:end])))

    voxel_size = 5.0
    new_result = VoxelSampler(voxel_size=voxel_size, method="closest").sample(cloud)
    new_points = _single_chunk_xyz(new_result)

    points = np.column_stack([xs, ys, zs])
    voxel_i, voxel_j, voxel_k = _voxel_indices(points[:, 0], points[:, 1], points[:, 2], voxel_size)
    coords = np.column_stack((voxel_i, voxel_j, voxel_k))
    unique_groups, labels = np.unique(coords, axis=0, return_inverse=True)
    centers = (unique_groups.astype(np.float64) + 0.5) * voxel_size
    old_indices = _compute_voxel_closest(points, labels, centers)
    old_points = points[old_indices]

    new_sorted = new_points[np.lexsort(new_points.T[::-1])]
    old_sorted = old_points[np.lexsort(old_points.T[::-1])]
    np.testing.assert_allclose(new_sorted, old_sorted)


# ----------------------------------------------------------------------
# "all" -- unchanged behavior, confirmed identity-preserving.
# ----------------------------------------------------------------------


def test_all_still_returns_the_same_cloud_instance() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 2.0], [1.0, 2.0], [1.0, 2.0]))

    result = VoxelSampler(voxel_size=0.5, method="all").sample(cloud)

    assert result is cloud


# ----------------------------------------------------------------------
# Validation, unchanged.
# ----------------------------------------------------------------------


def test_empty_cloud_rejected_for_centroid() -> None:
    with pytest.raises(SamplingError, match="empty"):
        VoxelSampler(voxel_size=1.0, method="centroid").sample(PointCloud())


def test_empty_cloud_rejected_for_closest() -> None:
    with pytest.raises(SamplingError, match="empty"):
        VoxelSampler(voxel_size=1.0, method="closest").sample(PointCloud())


def test_random_method_still_uses_the_original_path() -> None:
    """random is deliberately unchanged in this PR -- confirm it still works, not that it's fast/chunked."""
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 2.0, 11.0], [1.0, 2.0, 11.0], [1.0, 2.0, 11.0]))

    result = VoxelSampler(voxel_size=10.0, method="random", seed=42).sample(cloud)

    assert result.point_count == 2  # one representative per voxel (2 voxels: [0,10) and [10,20))
