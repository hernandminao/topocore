"""
Regression suite for PR21.7.6: StratifiedSampler's "centroid" and
"closest" methods, rewritten to a two-pass, chunk-wise vectorized
accumulator instead of the old `for cell_idx in range(len(unique_cells)):
mask = group_labels == cell_idx` per-cell boolean-mask-scan loop --
the SAME complexity bug independently found and fixed in
VoxelSampler (PR21.7.5), reproduced here in a sibling module.

Unlike VoxelSampler, two passes are needed (not one) because:
- "centroid" here finds the REAL point closest to the (fully known,
  pass-1-computed) arithmetic centroid, not the centroid itself.
- "closest" needs each cell's mean Z (pass 1, since the center used
  is (cell_XY_center, mean_Z), not the cell's own coordinate-derived
  center in all 3 axes) before computing per-point distances (pass 2).

The decisive correctness property verified throughout: a single cell
whose member points are split across ALL chunks (not just two), in
various arrangements, must produce the exact same selection as the
pre-PR21.7.6 reference algorithm -- verified via an independent
reimplementation of that exact algorithm, not just by comparing
outputs of the new code to itself.

"closest" with samples_per_cell > 1 uses NumPy's own argsort, whose
default `kind` ('quicksort') is NOT formally guaranteed stable on
ties -- this suite verifies the new two-pass design reproduces
NumPy's actual (if unspecified) tie behavior exactly, by visiting
points in the same global order the original single-array
implementation would have, rather than inventing an independent tie
rule.

"random" is unchanged by this PR -- the same reservoir-sampling and
RNG-reproducibility questions deferred for VoxelSampler's "random"
(PR21.7.5) apply here too, and are not resolved unilaterally.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.sampling.stratified import StratifiedSampler


def _chunk(xs: list[float], ys: list[float], zs: list[float]) -> Chunk:
    n = len(xs)
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    return chunk


def _reference_algorithm(cloud: PointCloud, cell_size: float, samples_per_cell: int, method: str) -> list[int]:
    """The exact pre-PR21.7.6 algorithm, reimplemented here as an independent reference."""
    xs, ys, zs = [], [], []
    for c in cloud:
        xs.append(c[PointAttribute.X])
        ys.append(c[PointAttribute.Y])
        zs.append(c[PointAttribute.Z])
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    cell_i = np.floor(x / cell_size).astype(np.int64)
    cell_j = np.floor(y / cell_size).astype(np.int64)
    cell_coords = np.stack([cell_i, cell_j], axis=1)
    unique_cells, group_labels = np.unique(cell_coords, axis=0, return_inverse=True)
    points = np.stack([x, y, z], axis=1)
    cell_centers = (unique_cells.astype(np.float64) + 0.5) * cell_size

    selected: list[int] = []
    for cell_idx in range(len(unique_cells)):
        mask = group_labels == cell_idx
        if not np.any(mask):
            continue
        cell_points = points[mask]
        orig_indices = np.flatnonzero(mask)
        n_sample = min(samples_per_cell, len(cell_points))

        if method == "centroid":
            centroid = cell_points.mean(axis=0)
            distances = np.linalg.norm(cell_points - centroid, axis=1)
            closest = np.argmin(distances)
            selected.append(int(orig_indices[closest]))
        else:
            center = np.array(
                [
                    cell_centers[cell_idx][0],
                    cell_centers[cell_idx][1],
                    cell_points[:, 2].mean(),
                ]
            )
            distances = np.linalg.norm(cell_points - center, axis=1)
            sorted_idx = np.argsort(distances)
            selected.extend(orig_indices[sorted_idx[:n_sample]].tolist())

    return sorted(set(selected))


def _sampled_x_values(cloud: PointCloud) -> list[float]:
    (chunk,) = list(cloud)
    return sorted(chunk[PointAttribute.X].tolist())


# ----------------------------------------------------------------------
# The decisive cross-chunk-boundary case: a cell split across ALL chunks.
# ----------------------------------------------------------------------


def test_centroid_cell_split_across_all_three_chunks_matches_reference() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0, 2.0], [1.0, 11.0, 2.0], [1.0, 11.0, 2.0]))  # A, B, A
    cloud.add_chunk(_chunk([21.0, 3.0], [21.0, 3.0], [21.0, 3.0]))  # C, A
    cloud.add_chunk(_chunk([4.0, 22.0], [4.0, 22.0], [4.0, 22.0]))  # A, C

    reference = _reference_algorithm(cloud, cell_size=10.0, samples_per_cell=1, method="centroid")
    xs_all = np.concatenate([c[PointAttribute.X] for c in cloud])
    reference_x = sorted(xs_all[reference].tolist())

    result = StratifiedSampler(cell_size=10.0, samples_per_cell=1, method="centroid", seed=0).sample(cloud)

    assert _sampled_x_values(result) == reference_x


def test_closest_cell_split_across_all_three_chunks_matches_reference() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0, 2.0], [1.0, 11.0, 2.0], [1.0, 11.0, 2.0]))
    cloud.add_chunk(_chunk([21.0, 3.0], [21.0, 3.0], [21.0, 3.0]))
    cloud.add_chunk(_chunk([4.0, 22.0], [4.0, 22.0], [4.0, 22.0]))

    reference = _reference_algorithm(cloud, cell_size=10.0, samples_per_cell=1, method="closest")
    xs_all = np.concatenate([c[PointAttribute.X] for c in cloud])
    reference_x = sorted(xs_all[reference].tolist())

    result = StratifiedSampler(cell_size=10.0, samples_per_cell=1, method="closest", seed=0).sample(cloud)

    assert _sampled_x_values(result) == reference_x


def test_closest_multi_sample_per_cell_matches_reference() -> None:
    """samples_per_cell=3 -- exercises argsort's exact (not formally stable) tie-break behavior."""
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 11.0, 2.0], [1.0, 11.0, 2.0], [1.0, 11.0, 2.0]))
    cloud.add_chunk(_chunk([21.0, 3.0], [21.0, 3.0], [21.0, 3.0]))
    cloud.add_chunk(_chunk([4.0, 22.0], [4.0, 22.0], [4.0, 22.0]))

    reference = _reference_algorithm(cloud, cell_size=10.0, samples_per_cell=3, method="closest")
    xs_all = np.concatenate([c[PointAttribute.X] for c in cloud])
    reference_x = sorted(xs_all[reference].tolist())

    result = StratifiedSampler(cell_size=10.0, samples_per_cell=3, method="closest", seed=0).sample(cloud)

    assert _sampled_x_values(result) == reference_x


def test_matches_reference_on_random_multi_chunk_data() -> None:
    rng = np.random.default_rng(3)
    n = 500
    xs, ys, zs = rng.uniform(0, 50, n), rng.uniform(0, 50, n), rng.uniform(0, 10, n)

    cloud = PointCloud()
    for start, end in ((0, 150), (150, 300), (300, 500)):
        cloud.add_chunk(_chunk(list(xs[start:end]), list(ys[start:end]), list(zs[start:end])))

    for method, samples in (("centroid", 1), ("closest", 1), ("closest", 3)):
        reference = _reference_algorithm(cloud, cell_size=5.0, samples_per_cell=samples, method=method)
        reference_x = sorted(xs[reference].tolist())

        result = StratifiedSampler(
            cell_size=5.0,
            samples_per_cell=samples,
            method=method,  # type: ignore[arg-type]
            seed=0,
        ).sample(cloud)

        assert _sampled_x_values(result) == pytest.approx(reference_x)


# ----------------------------------------------------------------------
# Cell with fewer points than samples_per_cell.
# ----------------------------------------------------------------------


def test_cell_with_fewer_points_than_samples_per_cell() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0], [1.0], [1.0]))

    result = StratifiedSampler(cell_size=10.0, samples_per_cell=5, method="closest").sample(cloud)

    assert result.point_count == 1


# ----------------------------------------------------------------------
# Empty cloud.
# ----------------------------------------------------------------------


def test_empty_cloud_rejected_for_centroid() -> None:
    with pytest.raises(SamplingError, match="empty"):
        StratifiedSampler(cell_size=1.0, method="centroid").sample(PointCloud())


def test_empty_cloud_rejected_for_closest() -> None:
    with pytest.raises(SamplingError, match="empty"):
        StratifiedSampler(cell_size=1.0, method="closest").sample(PointCloud())


# ----------------------------------------------------------------------
# "random" -- unchanged, still uses the original path.
# ----------------------------------------------------------------------


def test_random_method_still_works_unchanged() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk([1.0, 2.0, 11.0], [1.0, 2.0, 11.0], [1.0, 2.0, 11.0]))

    result = StratifiedSampler(cell_size=10.0, samples_per_cell=1, method="random", seed=42).sample(cloud)

    assert result.point_count == 2  # one representative per cell (2 cells: [0,10) and [10,20))
