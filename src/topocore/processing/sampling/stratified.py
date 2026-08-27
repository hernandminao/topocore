"""
topocore.processing.sampling.stratified
=======================================

Stratified sampling.

This module implements stratified sampling where the point cloud is
divided into a 2D grid (XY plane) and a fixed number of points is
sampled from each cell. This ensures uniform spatial distribution
regardless of the original point density.

Parameters
----------
cell_size
    Size of each grid cell.
samples_per_cell
    Number of points to sample from each cell.
method
    Sampling method within each cell: "random", "centroid", or "closest".

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import ClassVar, Literal, override

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError

from .base import Sampler
from .random import _build_sampled_cloud


class StratifiedSampler(Sampler):
    """
    Stratified sampling.

    Divides the point cloud into a 2D grid and samples a fixed number
    of points from each cell. This ensures uniform spatial distribution.

    Parameters
    ----------
    cell_size
        Size of each grid cell.
    samples_per_cell
        Number of points to sample from each cell.
    method
        Sampling method within each cell:
        - "random": random points from the cell
        - "centroid": point closest to the cell centroid
        - "(representative point, preserves attributes)
        - "closest": point closest to cell center (preserves real points)

    Notes
    -----
    PR21.7.6 (performance): "centroid" and "closest" are computed via
    a two-pass, chunk-wise, vectorized accumulator -- see `sample()`'s
    own docstring for the O(N x G) complexity bug this replaced (the
    same bug independently found and fixed in VoxelSampler, PR21.7.5,
    reproduced here in this sibling module).

    "random" is deliberately NOT chunk-optimized. For
    ``method="random"``, sampling is performed over the complete
    logical point set. The result is reproducible for a given `seed`
    and INDEPENDENT of the physical chunk partitioning -- confirmed
    directly: the same seed selects the exact same points whether the
    cloud is held in 1 chunk, 4 equal chunks, or unevenly sized
    chunks (see this module's own regression suite). This behavior is
    intentionally preserved, not an oversight: streaming "random"
    (e.g. via reservoir sampling) would necessarily make its RNG
    consumption -- and therefore its selected points -- depend on how
    the cloud happens to be chunked, silently breaking this existing,
    real (if previously undocumented) guarantee. Materializing the
    full point set for "random" is therefore a deliberate
    compatibility decision, not unaddressed technical debt.

    Examples
    --------
    >>> sampler = StratifiedSampler(cell_size=1.0, samples_per_cell=5)
    >>> downsampled = sampler.sample(cloud)  # 5 points per cell

    >>> sampler = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method="centroid")
    >>> downsampled = sampler.sample(cloud)  # 1 centroid per cell
    """

    __slots__ = (
        "_cell_size",
        "_method",
        "_samples_per_cell",
        "_seed",
    )

    _SUPPORTED_METHODS: ClassVar[set[str]] = {"random", "centroid", "closest"}

    def __init__(
        self,
        cell_size: float,
        samples_per_cell: int = 1,
        method: Literal["random", "centroid", "closest"] = "random",
        seed: int | None = None,
    ) -> None:
        if cell_size <= 0:
            raise SamplingError(f"cell_size must be positive, got {cell_size}.")

        if samples_per_cell < 1:
            raise SamplingError(f"samples_per_cell must be >= 1, got {samples_per_cell}.")

        if method not in self._SUPPORTED_METHODS:
            raise SamplingError(f"method must be one of {self._SUPPORTED_METHODS}, got {method}.")

        self._cell_size = cell_size
        self._samples_per_cell = samples_per_cell
        self._method = method
        self._seed = seed

    @override
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Stratified sample the point cloud."""

        if cloud.is_empty:
            raise SamplingError("Cannot sample an empty point cloud.")

        # PR21.7.6: "centroid" and "closest" are computed via a
        # two-pass, chunk-wise accumulator instead of first
        # concatenating every chunk's X/Y/Z into one global array and
        # then looping `for cell_idx in range(len(unique_cells)):
        # mask = group_labels == cell_idx` -- the same O(N x G)
        # per-group boolean-scan bug independently found in
        # VoxelSampler (PR21.7.5), reproduced here in a sibling
        # module. Two passes are needed here (not one, unlike
        # VoxelSampler's "centroid") because this method's "centroid"
        # doesn't just return the arithmetic centroid -- it finds the
        # REAL point closest to it, which requires the centroid to be
        # fully known (pass 1) before searching for that real point
        # (pass 2); "closest" similarly needs each cell's mean Z
        # (pass 1) before computing per-point distances to the
        # (XY-cell-center, Z-mean) point (pass 2). Both passes
        # iterate the SAME already-in-memory list of Chunk objects
        # PointCloud already holds -- no data is re-read from disk,
        # and neither pass concatenates chunk data into one global
        # array. Exact bit-for-bit equivalence to the pre-PR21.7.6
        # reference (including "closest"'s per-cell argsort-based
        # multi-sample selection, whose default `kind` is NOT
        # formally stable-guaranteed by NumPy) was verified by
        # ensuring points are visited in the same global order the
        # original single-array implementation would have -- see
        # this PR's own regression suite.
        if self._method == "centroid":
            return self._sample_centroid_two_pass(cloud)

        if self._method == "closest":
            return self._sample_closest_two_pass(cloud)

        # "random" is deliberately NOT optimized here, for the exact
        # same reason as VoxelSampler's "random" (PR21.7.5): making
        # it chunk-safe for `samples_per_cell` > 1 without-replacement
        # draws would require a generalized reservoir-sampling
        # algorithm, a genuine change to RNG consumption order that
        # could not preserve exact seeded reproducibility, only the
        # sampling distribution -- a design decision not made
        # unilaterally here.
        return self._sample_random_legacy(cloud)

    def _cell_key(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (cell_i, cell_j) for an array of X/Y coordinates."""
        cell_i = np.floor(x / self._cell_size).astype(np.int64)
        cell_j = np.floor(y / self._cell_size).astype(np.int64)
        return cell_i, cell_j

    def _sample_centroid_two_pass(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Select, per cell, the real point closest to that cell's
        centroid -- via a two-pass chunk-wise accumulator.

        Pass 1 accumulates (sum_x, sum_y, sum_z, count) per cell key,
        vectorized within each chunk (np.unique + np.add.at +
        np.bincount, no Python loop over points or groups), merged
        into a global dict keyed by (cell_i, cell_j). Pass 2 revisits
        the same chunks, and for each point computes its distance to
        its cell's now-fully-known centroid, keeping the point with
        the STRICTLY smallest distance per cell (matching
        np.argmin's own documented "first occurrence wins" tie-break
        exactly, since ties are only replaced on a strictly smaller
        distance, and points are visited in the same global order the
        original single-array implementation's np.argmin would have
        seen them in).
        """
        sums: dict[tuple[int, int], np.ndarray] = {}
        counts: dict[tuple[int, int], int] = {}

        for chunk in cloud:
            if chunk.size == 0:
                continue

            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

            cell_i, cell_j = self._cell_key(x, y)
            coords = np.column_stack((cell_i, cell_j))
            unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

            local_sums = np.zeros((len(unique_coords), 3), dtype=np.float64)
            np.add.at(local_sums, inverse, np.column_stack((x, y, z)))
            local_counts = np.bincount(inverse, minlength=len(unique_coords))

            for local_index, key_array in enumerate(unique_coords):
                key = (int(key_array[0]), int(key_array[1]))
                if key in sums:
                    sums[key] += local_sums[local_index]
                    counts[key] += int(local_counts[local_index])
                else:
                    sums[key] = local_sums[local_index]
                    counts[key] = int(local_counts[local_index])

        centroids = {key: sums[key] / counts[key] for key in sums}

        best_distance: dict[tuple[int, int], float] = {}
        best_index: dict[tuple[int, int], int] = {}

        global_offset = 0
        for chunk in cloud:
            if chunk.size == 0:
                continue

            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

            cell_i, cell_j = self._cell_key(x, y)

            for local_index in range(len(x)):
                key = (int(cell_i[local_index]), int(cell_j[local_index]))
                centroid = centroids[key]
                distance = float(
                    np.sqrt(
                        (x[local_index] - centroid[0]) ** 2
                        + (y[local_index] - centroid[1]) ** 2
                        + (z[local_index] - centroid[2]) ** 2
                    )
                )
                if key not in best_distance or distance < best_distance[key]:
                    best_distance[key] = distance
                    best_index[key] = global_offset + local_index

            global_offset += chunk.size

        selected_array = np.unique(np.asarray(sorted(best_index.values()), dtype=np.intp))
        return _build_sampled_cloud(cloud, selected_array)

    def _sample_closest_two_pass(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Select, per cell, the `samples_per_cell` real points closest
        to that cell's (XY center, mean Z) point -- via a two-pass
        chunk-wise accumulator.

        Pass 1 accumulates (sum_z, count) per cell key (the cell's XY
        center is derived directly from the cell key itself --
        `(cell_i + 0.5) * cell_size` -- needing no accumulation at
        all). Pass 2 revisits the same chunks and buckets each
        point's (global_index, distance) into its cell's own list, IN
        GLOBAL VISITING ORDER -- exactly matching the order the
        original single concatenated array would have held that
        cell's members in. Because NumPy's default argsort `kind`
        ('quicksort') is not formally guaranteed stable on ties, this
        preserves bit-for-bit equivalence by reproducing the EXACT
        per-cell distances array (same values, same order) the
        original implementation would have passed to `np.argsort`,
        then applying that identical call, rather than inventing an
        independent streaming top-K tie-break rule that might diverge
        from NumPy's actual (if unspecified) tie behavior.
        """
        z_sums: dict[tuple[int, int], float] = {}
        z_counts: dict[tuple[int, int], int] = {}

        for chunk in cloud:
            if chunk.size == 0:
                continue

            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

            cell_i, cell_j = self._cell_key(x, y)
            coords = np.column_stack((cell_i, cell_j))
            unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

            local_z_sums = np.bincount(inverse, weights=z, minlength=len(unique_coords))
            local_counts = np.bincount(inverse, minlength=len(unique_coords))

            for local_index, key_array in enumerate(unique_coords):
                key = (int(key_array[0]), int(key_array[1]))
                if key in z_sums:
                    z_sums[key] += float(local_z_sums[local_index])
                    z_counts[key] += int(local_counts[local_index])
                else:
                    z_sums[key] = float(local_z_sums[local_index])
                    z_counts[key] = int(local_counts[local_index])

        cell_members: dict[tuple[int, int], list[tuple[int, float]]] = {}

        global_offset = 0
        for chunk in cloud:
            if chunk.size == 0:
                continue

            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

            cell_i, cell_j = self._cell_key(x, y)

            for local_index in range(len(x)):
                key = (int(cell_i[local_index]), int(cell_j[local_index]))
                center_x = (key[0] + 0.5) * self._cell_size
                center_y = (key[1] + 0.5) * self._cell_size
                center_z = z_sums[key] / z_counts[key]
                distance = float(
                    np.sqrt(
                        (x[local_index] - center_x) ** 2
                        + (y[local_index] - center_y) ** 2
                        + (z[local_index] - center_z) ** 2
                    )
                )
                cell_members.setdefault(key, []).append((global_offset + local_index, distance))

            global_offset += chunk.size

        selected_indices: list[int] = []
        for members in cell_members.values():
            distances = np.array([distance for _index, distance in members], dtype=np.float64)
            indices = np.array([index for index, _distance in members], dtype=np.intp)
            n_sample = min(self._samples_per_cell, len(members))
            sorted_order = np.argsort(distances)
            selected_indices.extend(indices[sorted_order[:n_sample]].tolist())

        selected_array = np.unique(np.asarray(selected_indices, dtype=np.intp))
        return _build_sampled_cloud(cloud, selected_array)

    def _sample_random_legacy(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Random sampling within each cell -- unchanged from before PR21.7.6 (see sample()'s own docstring)."""
        # Extract coordinates
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for chunk in cloud:
            xs.append(chunk[PointAttribute.X])
            ys.append(chunk[PointAttribute.Y])

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        # Compute cell indices (2D)
        cell_i = np.floor(x / self._cell_size).astype(np.int64)
        cell_j = np.floor(y / self._cell_size).astype(np.int64)

        cell_coords = np.stack([cell_i, cell_j], axis=1)

        unique_cells, group_labels = np.unique(
            cell_coords,
            axis=0,
            return_inverse=True,
        )

        selected_indices: list[int] = []

        # Reproducible random generator -- seed is configurable (see
        # class docstring's `seed` parameter), matching every other
        # sampler in this module (RandomSampler, VoxelSampler). A
        # prior version of this file hardcoded seed=42 with no way
        # to override it -- found and fixed in PR19 (session audit):
        # confirmed the "random" method always returned the exact
        # same selection, regardless of instance, call count, or
        # global NumPy random state. `seed=None` here (the new
        # default) means non-deterministic, like the rest of the
        # module -- pass an explicit seed for reproducibility.
        rng = np.random.default_rng(self._seed)

        for cell_idx in range(len(unique_cells)):
            mask = group_labels == cell_idx

            if not np.any(mask):
                continue

            orig_indices = np.flatnonzero(mask)

            n_cell_points = len(orig_indices)
            n_sample = min(
                self._samples_per_cell,
                n_cell_points,
            )

            chosen = rng.choice(
                orig_indices,
                size=n_sample,
                replace=False,
            )

            selected_indices.extend(chosen.tolist())

        if not selected_indices:
            raise SamplingError("No points selected. Try increasing samples_per_cell.")

        selected_array = np.unique(
            np.asarray(
                selected_indices,
                dtype=np.intp,
            )
        )

        return _build_sampled_cloud(
            cloud,
            selected_array,
        )

    @override
    def name(
        self,
    ) -> str:
        return (
            f"stratified(cell={self._cell_size}, samples={self._samples_per_cell}, "
            f"method={self._method}, seed={self._seed})"
        )


__all__ = [
    "StratifiedSampler",
]
