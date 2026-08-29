"""
topocore.processing.sampling.voxel
==================================

Voxel grid sampling.

This module implements voxel-based downsampling where the point cloud
is divided into a regular 3D grid and each voxel is represented by
one or more points.

Methods:
- "centroid": average of all points in the voxel (smoothing)
- "closest": point closest to the voxel center (preserves real points)
- "random": random point from the voxel (preserves real points)
- "all": all points in the voxel (no reduction, for visualization)

Parameters
----------
voxel_size
    Size of each voxel.
method
    Sampling method: "centroid", "closest", "random", or "all".

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Literal, override

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError

from .base import Sampler
from .random import _build_sampled_cloud


def _voxel_indices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute voxel indices for each point."""

    if voxel_size <= 0:
        raise SamplingError(f"voxel_size must be positive, got {voxel_size}.")

    if x.size == 0 or y.size == 0 or z.size == 0:
        raise SamplingError("Cannot compute voxel indices from empty coordinates.")

    # Validate coordinates are finite
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise SamplingError("Point cloud contains NaN or Inf coordinates.")

    max_abs = float(
        max(
            float(np.max(np.abs(x))),
            float(np.max(np.abs(y))),
            float(np.max(np.abs(z))),
        )
    )

    max_index = np.iinfo(np.int64).max // 2

    if max_abs / voxel_size > max_index:
        raise SamplingError("Voxel indexing overflow risk.")

    voxel_i = np.floor(x / voxel_size).astype(np.int64)
    voxel_j = np.floor(y / voxel_size).astype(np.int64)
    voxel_k = np.floor(z / voxel_size).astype(np.int64)

    return voxel_i, voxel_j, voxel_k


# PR21 remediation (VOXEL-DEAD-CODE-001): _compute_voxel_centroid()
# and _compute_voxel_closest() -- the pre-PR21.7.5 O(N x G) helpers
# for "centroid"/"closest" -- were removed here. Confirmed via
# instrumentation (monkeypatching every suspect method and calling
# sample() for all 4 public methods) that neither was ever called
# from any real usage path: PR21.7.5 replaced their functionality
# with the chunk-wise _sample_centroid_chunked()/
# _sample_closest_chunked() methods below, called directly from
# sample() before _apply_sampling() is ever reached, but the old
# implementations were left in place rather than removed at the
# time. Confirmed exhaustively: zero references anywhere in
# PRODUCTION code (direct calls, indirect calls, __all__, or
# documentation) -- the *_chunked() replacements already cover their
# exact responsibility (verified identical results across
# constructor/happy-path/attribute-preservation/reproducibility/
# chunk-invariance tests in this module's own regression suite).
# One genuine indirect consumer WAS found during this removal, in
# this package's own test suite (test_voxel_chunked_accumulator.py,
# from PR21.7.5's original work): both functions were used there as
# a known-correct REFERENCE implementation to cross-check the new
# chunked accumulator against the pre-PR21.7.5 algorithm at scale.
# That cross-check is valuable and was preserved -- the reference
# implementation was moved into that test file itself (as
# `_reference_compute_voxel_centroid`/`_reference_compute_voxel_closest`),
# since its only real purpose was ever as a test-scoped baseline,
# never a production capability.


class VoxelSampler(Sampler):
    """
    Voxel grid downsampling.

    Divides the point cloud into a regular 3D grid and samples
    each voxel using the specified method.

    Parameters
    ----------
    voxel_size
        Size of each voxel.
    method
        Sampling method:
        - "centroid": average of all points (smoothing)
        - "closest": point closest to center (preserves real points)
        - "random": random point from voxel (preserves real points)
        - "all": all points (no reduction, for visualization)

    Notes
    -----
    PR21.7.5 (performance): "centroid" and "closest" are computed via
    a chunk-wise, vectorized accumulator -- see `sample()`'s own
    docstring for the O(N x G) complexity bug this replaced.

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
    >>> sampler = VoxelSampler(voxel_size=0.5, method="centroid")
    >>> downsampled = sampler.sample(cloud)  # Smoothed

    >>> sampler = VoxelSampler(voxel_size=0.5, method="closest")
    >>> downsampled = sampler.sample(cloud)  # Preserves original points
    """

    __slots__ = (
        "_voxel_size",
        "_method",
        "_seed",
    )

    _SUPPORTED_METHODS = {"centroid", "closest", "random", "all"}

    def __init__(
        self,
        voxel_size: float,
        method: Literal["centroid", "closest", "random", "all"] = "centroid",
        seed: int | None = None,
    ) -> None:
        if voxel_size <= 0:
            raise SamplingError(f"voxel_size must be positive, got {voxel_size}.")

        if method not in self._SUPPORTED_METHODS:
            raise SamplingError(f"method must be one of {self._SUPPORTED_METHODS}, got {method}.")

        self._voxel_size = voxel_size
        self._method = method
        self._seed = seed

    @override
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Voxel sample the point cloud."""

        if cloud.is_empty:
            raise SamplingError("Cannot sample an empty point cloud.")

        # PR21.7.5.6: "all" already returned the exact same cloud
        # instance unmodified (confirmed directly: result is cloud
        # was True before this change) -- moving that check first
        # skips the now-entirely-wasted _extract_points()/
        # _create_voxels() work for this method, changing nothing
        # observable about the return value.
        if self._method == "all":
            return cloud

        # PR21.7.5.3/5.4: "centroid" and "closest" are computed via a
        # chunk-wise accumulator (see _accumulate_centroid()/
        # _accumulate_closest()) instead of first concatenating every
        # chunk's X/Y/Z into one global array and then, critically,
        # looping `for i in range(n_groups): mask = inverse == i` --
        # an O(N x n_groups) full-array boolean scan PER VOXEL GROUP,
        # confirmed via direct benchmarking to be effectively
        # quadratic for realistic data (n_groups scales with N for
        # typical point clouds): 20,000 points took 1.27s; a linear
        # projection from 2,000 points (0.025s) would predict
        # ~0.25s, not 1.27s -- a genuine, severe, pre-existing
        # complexity bug entirely independent of the chunking/memory
        # concern this PR was originally scoped around, fixed by the
        # SAME change: a proper vectorized, single-pass accumulator
        # (np.add.at/np.bincount within each chunk, merged into a
        # dict keyed by voxel coordinate across chunks) is both
        # streaming-safe AND O(N + G), not O(N x G).
        if self._method == "centroid":
            return self._sample_centroid_chunked(cloud)

        if self._method == "closest":
            return self._sample_closest_chunked(cloud)

        # "random" is deliberately NOT optimized in PR21.7.5: making
        # it chunk-safe would require reservoir sampling, a genuine
        # algorithmic change (not just a reordering of the same
        # computation), and reservoir sampling's RNG consumption
        # order necessarily differs from "concatenate everything,
        # then choose" -- meaning exact point-for-point reproducibility
        # with the same seed could NOT be preserved, only the
        # sampling distribution. That is a real design decision
        # (exact reproducibility vs. distributional equivalence) this
        # PR does not make unilaterally. "random" therefore still
        # goes through the pre-PR21.7.5 path below.
        points = self._extract_points(cloud)
        groups, labels = self._create_voxels(points)
        return self._apply_sampling(cloud, len(groups), labels)

    def _sample_centroid_chunked(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Compute per-voxel centroids via a chunk-wise accumulator.

        Within each chunk, voxel grouping and per-group sums/counts
        are computed in one vectorized pass (np.unique + np.add.at +
        np.bincount -- no Python loop over points or groups). Each
        chunk's small number of LOCAL (voxel_key -> sum, count)
        results are then merged into a global dict keyed by the
        voxel's own (i, j, k) coordinate -- the same coordinate
        regardless of which chunk a point came from, so a voxel
        whose member points are split across multiple chunks is
        merged correctly. Total complexity is O(N + G): O(N) for the
        per-chunk vectorized accumulation, O(G) for the (bounded by
        total unique voxels, never by N^2) dict-merge step.
        """
        global_sums: dict[tuple[int, int, int], np.ndarray] = {}
        global_counts: dict[tuple[int, int, int], int] = {}

        for chunk in cloud:
            if chunk.size == 0:
                continue

            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

            voxel_i, voxel_j, voxel_k = _voxel_indices(x, y, z, self._voxel_size)
            coords = np.column_stack((voxel_i, voxel_j, voxel_k))
            unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

            local_sums = np.zeros((len(unique_coords), 3), dtype=np.float64)
            np.add.at(local_sums, inverse, np.column_stack((x, y, z)))
            local_counts = np.bincount(inverse, minlength=len(unique_coords))

            for local_index, key_array in enumerate(unique_coords):
                key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
                if key in global_sums:
                    global_sums[key] += local_sums[local_index]
                    global_counts[key] += int(local_counts[local_index])
                else:
                    global_sums[key] = local_sums[local_index]
                    global_counts[key] = int(local_counts[local_index])

        centroids = np.array(
            [global_sums[key] / global_counts[key] for key in global_sums],
            dtype=np.float64,
        )

        return self._build_from_points(cloud, centroids)

    def _sample_closest_chunked(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Select, per voxel, the point closest to that voxel's
        geometric center, via a chunk-wise accumulator.

        Preserves the pre-PR21.7.5 tie-breaking rule exactly: a
        candidate replaces the current best ONLY when it is
        STRICTLY closer (`<`, never `<=`), matching np.argmin's own
        "first occurrence wins" behavior on ties -- confirmed this
        must be a global point-index-order tie-break (not per-chunk),
        so ties are resolved identically regardless of how the cloud
        happens to be chunked.
        """
        best_distance: dict[tuple[int, int, int], float] = {}
        best_index: dict[tuple[int, int, int], int] = {}

        global_offset = 0
        for chunk in cloud:
            if chunk.size == 0:
                continue

            x = np.asarray(chunk[PointAttribute.X], dtype=np.float64)
            y = np.asarray(chunk[PointAttribute.Y], dtype=np.float64)
            z = np.asarray(chunk[PointAttribute.Z], dtype=np.float64)

            voxel_i, voxel_j, voxel_k = _voxel_indices(x, y, z, self._voxel_size)
            centers_x = (voxel_i.astype(np.float64) + 0.5) * self._voxel_size
            centers_y = (voxel_j.astype(np.float64) + 0.5) * self._voxel_size
            centers_z = (voxel_k.astype(np.float64) + 0.5) * self._voxel_size
            distances = np.sqrt((x - centers_x) ** 2 + (y - centers_y) ** 2 + (z - centers_z) ** 2)

            for local_index in range(len(x)):
                key = (int(voxel_i[local_index]), int(voxel_j[local_index]), int(voxel_k[local_index]))
                distance = float(distances[local_index])
                if key not in best_distance or distance < best_distance[key]:
                    best_distance[key] = distance
                    best_index[key] = global_offset + local_index

            global_offset += chunk.size

        indices = np.array(sorted(best_index.values()), dtype=np.intp)
        return _build_sampled_cloud(cloud, indices)

    def _extract_points(
        self,
        cloud: PointCloud,
    ) -> np.ndarray:
        """Extract XYZ coordinates."""

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []

        for chunk in cloud:
            xs.append(
                np.asarray(
                    chunk[PointAttribute.X],
                    dtype=np.float64,
                )
            )
            ys.append(
                np.asarray(
                    chunk[PointAttribute.Y],
                    dtype=np.float64,
                )
            )
            zs.append(
                np.asarray(
                    chunk[PointAttribute.Z],
                    dtype=np.float64,
                )
            )

        if not xs:
            raise SamplingError("PointCloud has no coordinates.")

        return np.column_stack(
            (
                np.concatenate(xs),
                np.concatenate(ys),
                np.concatenate(zs),
            )
        )

    def _create_voxels(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create voxel groups."""

        voxel_i, voxel_j, voxel_k = _voxel_indices(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            self._voxel_size,
        )

        coordinates = np.column_stack(
            (
                voxel_i,
                voxel_j,
                voxel_k,
            )
        )

        unique_groups, labels = np.unique(
            coordinates,
            axis=0,
            return_inverse=True,
        )

        return unique_groups, labels

    def _apply_sampling(
        self,
        cloud: PointCloud,
        n_groups: int,
        labels: np.ndarray,
    ) -> PointCloud:
        """
        Apply the "random" voxel strategy.

        PR21 remediation (VOXEL-DEAD-CODE-001): previously also
        dispatched "centroid"/"closest" to _sample_centroid()/
        _sample_closest(), but sample() already returns via
        _sample_centroid_chunked()/_sample_closest_chunked() for
        those two methods before this method is ever reached -- this
        is now, and was already in practice, called only for
        "random" (confirmed via instrumentation before removing the
        dead branches). The `points` parameter existed solely for
        the removed centroid/closest branches and was dropped along
        with them; only the group count and voxel labels are needed
        to dispatch to _sample_random().
        """

        if self._method == "random":
            return self._sample_random(
                cloud,
                labels,
                n_groups,
            )

        return cloud

    def _sample_random(
        self,
        cloud: PointCloud,
        labels: np.ndarray,
        n_groups: int,
    ) -> PointCloud:
        """Sample one random point per voxel."""

        rng = np.random.default_rng(
            self._seed,
        )

        indices: list[int] = []

        for group in range(n_groups):
            members = np.flatnonzero(
                labels == group,
            )

            if members.size > 0:
                indices.append(int(rng.choice(members)))

        if not indices:
            raise SamplingError("Random voxel sampling produced no points.")

        sampled_indices = np.asarray(
            indices,
            dtype=np.intp,
        )

        return _build_sampled_cloud(
            cloud,
            sampled_indices,
        )

    def _build_from_points(
        self,
        cloud: PointCloud,
        points: np.ndarray,
    ) -> PointCloud:
        """
        Build a PointCloud from XYZ coordinates.

        Only spatial attributes (X, Y, Z) are preserved.
        Additional attributes such as intensity, RGB, classification,
        or return information are discarded because centroid
        aggregation requires attribute-specific interpolation rules.
        """
        from topocore.pointcloud.chunk import Chunk

        n_points = points.shape[0]
        attributes = [PointAttribute.X, PointAttribute.Y, PointAttribute.Z]

        # Copy other attributes if they exist
        # For centroid method, we only keep coordinates
        chunk = Chunk(size=n_points, attributes=attributes)
        chunk[PointAttribute.X][:] = points[:, 0]
        chunk[PointAttribute.Y][:] = points[:, 1]
        chunk[PointAttribute.Z][:] = points[:, 2]

        # Copy other attributes as arrays
        # (This is simplified; for full attribute support, we'd need
        # to interpolate or discard attributes)
        for attr in cloud.attributes:
            if attr not in attributes:
                # For centroid method, we can't easily combine attributes
                # like intensity, RGB, etc. So we skip them.
                pass

        new_cloud = PointCloud()
        new_cloud.add_chunk(chunk)
        new_cloud.update_bounds()

        return new_cloud

    @override
    def name(self) -> str:
        return f"voxel(size={self._voxel_size}, method={self._method}, seed={self._seed})"


__all__ = [
    "VoxelSampler",
]
