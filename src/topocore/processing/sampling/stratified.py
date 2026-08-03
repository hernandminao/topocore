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

from typing import Literal, override

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

    Examples
    --------
    >>> sampler = StratifiedSampler(cell_size=1.0, samples_per_cell=5)
    >>> downsampled = sampler.sample(cloud)  # 5 points per cell

    >>> sampler = StratifiedSampler(cell_size=1.0, samples_per_cell=1, method="centroid")
    >>> downsampled = sampler.sample(cloud)  # 1 centroid per cell
    """

    __slots__ = (
        "_cell_size",
        "_samples_per_cell",
        "_method",
    )

    _SUPPORTED_METHODS = {"random", "centroid", "closest"}

    def __init__(
        self,
        cell_size: float,
        samples_per_cell: int = 1,
        method: Literal["random", "centroid", "closest"] = "random",
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

    @override
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Stratified sample the point cloud."""

        if cloud.is_empty:
            raise SamplingError("Cannot sample an empty point cloud.")

        # Extract coordinates
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []

        for chunk in cloud:
            xs.append(chunk[PointAttribute.X])
            ys.append(chunk[PointAttribute.Y])
            zs.append(chunk[PointAttribute.Z])

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)

        # Compute cell indices (2D)
        cell_i = np.floor(x / self._cell_size).astype(np.int64)
        cell_j = np.floor(y / self._cell_size).astype(np.int64)

        cell_coords = np.stack([cell_i, cell_j], axis=1)

        unique_cells, group_labels = np.unique(
            cell_coords,
            axis=0,
            return_inverse=True,
        )

        points = np.stack([x, y, z], axis=1)

        # Cell centers in XY
        cell_centers = (unique_cells.astype(np.float64) + 0.5) * self._cell_size

        selected_indices: list[int] = []

        # Reproducible random generator
        rng = np.random.default_rng(42)

        for cell_idx in range(len(unique_cells)):
            mask = group_labels == cell_idx

            if not np.any(mask):
                continue

            cell_points = points[mask]
            orig_indices = np.flatnonzero(mask)

            n_cell_points = len(cell_points)
            n_sample = min(
                self._samples_per_cell,
                n_cell_points,
            )

            if self._method == "random":
                chosen = rng.choice(
                    orig_indices,
                    size=n_sample,
                    replace=False,
                )

                selected_indices.extend(chosen.tolist())

            elif self._method == "centroid":
                # Centroid sampling does not preserve original points.
                # Until PointCloud construction supports synthetic points,
                # fallback to closest real point to centroid.

                centroid = cell_points.mean(axis=0)

                distances = np.linalg.norm(
                    cell_points - centroid,
                    axis=1,
                )

                closest = np.argmin(distances)

                selected_indices.append(int(orig_indices[closest]))

            else:  # closest
                center = np.array(
                    [
                        cell_centers[cell_idx][0],
                        cell_centers[cell_idx][1],
                        cell_points[:, 2].mean(),
                    ],
                    dtype=np.float64,
                )

                distances = np.linalg.norm(
                    cell_points - center,
                    axis=1,
                )

                sorted_idx = np.argsort(distances)

                selected_indices.extend(orig_indices[sorted_idx[:n_sample]].tolist())

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
    def name(self) -> str:
        return f"stratified(cell={self._cell_size}, samples={self._samples_per_cell}, method={self._method})"


__all__ = [
    "StratifiedSampler",
]
