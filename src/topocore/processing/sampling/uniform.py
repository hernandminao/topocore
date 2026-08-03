"""
topocore.processing.sampling.uniform
====================================

Uniform sampling (systematic decimation).

This module implements uniform sampling where every N-th point is kept.
This is deterministic and fast, but does not equalize density.

Parameters
----------
step
    Keep every N-th point. Must be >= 1.
offset
    Starting offset (0 = first point, 1 = second, etc.).

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
from .random import _build_sampled_cloud


class UniformSampler(Sampler):
    """
    Uniform sampling (systematic decimation).

    Keeps every N-th point in the point cloud. This is deterministic
    and fast, but does not equalize density across the cloud.

    Parameters
    ----------
    step
        Keep every N-th point. Must be >= 1.
    offset
        Starting offset (0 = first point, 1 = second, etc.).

    Examples
    --------
    >>> sampler = UniformSampler(step=10)
    >>> downsampled = sampler.sample(cloud)  # Keeps points 0, 10, 20, ...
    """

    __slots__ = (
        "_step",
        "_offset",
    )

    def __init__(
        self,
        step: int,
        offset: int = 0,
    ) -> None:
        if step < 1:
            raise SamplingError(f"step must be >= 1, got {step}.")
        if offset < 0 or offset >= step:
            raise SamplingError(f"offset must be in [0, {step - 1}], got {offset}.")

        self._step = step
        self._offset = offset

    @override
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Uniformly sample the point cloud."""
        total_points = cloud.point_count

        if total_points == 0:
            raise SamplingError("Cannot sample an empty point cloud.")

        # Select every N-th point starting from offset
        indices = np.arange(self._offset, total_points, self._step, dtype=np.intp)

        return _build_sampled_cloud(cloud, indices)

    @override
    def name(self) -> str:
        return f"uniform(step={self._step}, offset={self._offset})"


__all__ = [
    "UniformSampler",
]
