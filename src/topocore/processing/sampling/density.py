"""
topocore.processing.sampling.density
====================================

Density-based adaptive sampling.

This module implements density-adaptive sampling where the sampling
rate varies based on the local point density. Dense regions are
sampled more aggressively, while sparse regions are preserved.

This is useful for:
- Preserving detail in sparse areas
- Reducing redundancy in dense areas
- Creating uniformly distributed point clouds

Parameters
----------
target_density
    Target point density (points per unit volume).
min_fraction
    Minimum fraction of points to keep (for sparse regions).
max_fraction
    Maximum fraction of points to keep (for dense regions).
radius
    Radius for density computation.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import override

import numpy as np

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError
from topocore.processing.neighbors import NeighborhoodManager
from topocore.processing.types import FloatArray1D

from .base import Sampler
from .random import _build_sampled_cloud


class DensitySampler(Sampler):
    """
    Density-adaptive sampling.

    Samples more aggressively in dense regions and preserves points
    in sparse regions.

    Parameters
    ----------
    target_density
        Target point density (points per unit volume).
    radius
        Radius for density computation.
    min_fraction
        Minimum fraction of points to keep (for sparse regions).
    max_fraction
        Maximum fraction of points to keep (for dense regions).
    seed
        Random seed for reproducibility.

    Examples
    --------
    >>> sampler = DensitySampler(target_density=10.0, radius=1.0)
    >>> downsampled = sampler.sample(cloud)
    """

    __slots__ = (
        "_target_density",
        "_radius",
        "_min_fraction",
        "_max_fraction",
        "_seed",
    )

    def __init__(
        self,
        target_density: float,
        radius: float = 1.0,
        min_fraction: float = 0.1,
        max_fraction: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if target_density <= 0:
            raise SamplingError(f"target_density must be positive, got {target_density}.")

        if radius <= 0:
            raise SamplingError(f"radius must be positive, got {radius}.")

        if not 0.0 < min_fraction <= 1.0:
            raise SamplingError(f"min_fraction must be in (0, 1], got {min_fraction}.")

        if not 0.0 < max_fraction <= 1.0:
            raise SamplingError(f"max_fraction must be in (0, 1], got {max_fraction}.")

        if min_fraction > max_fraction:
            raise SamplingError(f"min_fraction ({min_fraction}) must be <= max_fraction ({max_fraction}).")

        self._target_density = target_density
        self._radius = radius
        self._min_fraction = min_fraction
        self._max_fraction = max_fraction
        self._seed = seed

    @override
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Density-adaptive sample the point cloud."""
        if cloud.is_empty:
            raise SamplingError("Cannot sample an empty point cloud.")

        # Compute local density for each point
        densities = self._compute_density(cloud)

        # Compute sampling fraction based on density
        fractions = self._compute_fractions(densities)

        # Sample points using the computed fractions
        rng = np.random.default_rng(self._seed)

        total_points = cloud.point_count

        indices_to_keep: list[int] = []

        for i in range(total_points):
            if rng.random() < fractions[i]:
                indices_to_keep.append(i)

        if len(indices_to_keep) == 0:
            raise SamplingError("No points selected. Try increasing target_density or min_fraction.")

        return _build_sampled_cloud(
            cloud,
            np.asarray(indices_to_keep, dtype=np.intp),
        )

    def _compute_density(
        self,
        cloud: PointCloud,
    ) -> FloatArray1D:
        """Compute local density for each point."""

        manager = NeighborhoodManager.from_point_cloud(cloud)

        neighbor_lists = manager.radius_many(
            radius=self._radius,
            include_self=True,
        )

        counts = np.asarray(
            [len(neighbors) for neighbors in neighbor_lists],
            dtype=np.float64,
        )

        volume = 4.0 / 3.0 * np.pi * self._radius**3

        return counts / volume

    def _compute_fractions(
        self,
        densities: FloatArray1D,
    ) -> FloatArray1D:
        """
        Compute sampling fractions based on local density.

        Dense regions get lower fractions (more aggressive sampling).
        Sparse regions get higher fractions (preserve more points).
        """
        # Normalize densities relative to target
        # density_ratio = target_density / density
        density_ratio = self._target_density / np.maximum(densities, 1e-12)

        # Clamp to [min_fraction, max_fraction]
        fractions = np.clip(density_ratio, self._min_fraction, self._max_fraction)

        return fractions

    @override
    def name(self) -> str:
        return f"density(target={self._target_density}, radius={self._radius})"


__all__ = [
    "DensitySampler",
]
