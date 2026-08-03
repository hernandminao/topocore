"""
topocore.processing.sampling.random
===================================

Random sampling.

This module implements random sampling of point clouds. Points are
selected uniformly at random without replacement.

Parameters
----------
fraction
    Fraction of points to keep (0, 1].
count
    Exact number of points to keep. Mutually exclusive with fraction.
seed
    Random seed for reproducibility.

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

from .base import Sampler


def _build_sampled_cloud(
    cloud: PointCloud,
    indices: np.ndarray,
) -> PointCloud:
    """Build a new point cloud from selected indices."""
    from topocore.processing._shared import build_cloud, flatten_attributes

    flattened = flatten_attributes(cloud)
    return build_cloud(flattened, indices.astype(np.intp))


class RandomSampler(Sampler):
    """
    Random sampling without replacement.

    Parameters
    ----------
    fraction
        Fraction of points to keep (0, 1].
    count
        Exact number of points to keep. Mutually exclusive with fraction.
    seed
        Random seed for reproducibility.

    Examples
    --------
    >>> sampler = RandomSampler(fraction=0.1, seed=42)
    >>> downsampled = sampler.sample(cloud)

    >>> sampler = RandomSampler(count=10000, seed=42)
    >>> downsampled = sampler.sample(cloud)
    """

    __slots__ = (
        "_fraction",
        "_count",
        "_seed",
    )

    def __init__(
        self,
        fraction: float | None = None,
        count: int | None = None,
        seed: int | None = None,
    ) -> None:
        # Attribute declarations for type checking
        self._fraction: float | None
        self._count: int | None
        self._seed: int | None

        # Validate that exactly one of fraction or count is specified
        if (fraction is None) == (count is None):
            raise SamplingError("Specify exactly one of fraction or count, not both.")

        if fraction is not None:
            if not 0.0 < fraction <= 1.0:
                raise SamplingError(f"fraction must be in (0, 1], got {fraction}.")

            self._fraction = fraction
            self._count = None

        else:
            if count is None or count < 1:
                raise SamplingError(f"count must be >= 1, got {count}.")

            self._fraction = None
            self._count = count

        self._seed = seed

    @override
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Randomly sample the point cloud."""
        total_points = cloud.point_count

        if total_points == 0:
            raise SamplingError("Cannot sample an empty point cloud.")

        # Determine number of points to keep
        if self._fraction is not None:
            n = max(1, round(total_points * self._fraction))
        else:
            if self._count is None:
                raise SamplingError("No count or fraction specified.")

            n = self._count
            if n is None:
                raise SamplingError("No count or fraction specified.")
            if n > total_points:
                # Keep all points if count exceeds total
                n = total_points

        # Generate random indices
        rng = np.random.default_rng(self._seed)
        indices = np.sort(rng.choice(total_points, size=n, replace=False))

        return _build_sampled_cloud(cloud, indices)

    @override
    def name(self) -> str:
        if self._fraction is not None:
            return f"random(fraction={self._fraction})"
        return f"random(count={self._count})"


__all__ = [
    "RandomSampler",
]
