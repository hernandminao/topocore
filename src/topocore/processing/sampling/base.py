"""
topocore.processing.sampling.base
=================================

Abstract base class for point cloud samplers.

This module defines the interface for point cloud sampling/downsampling.
Samplers reduce the number of points in a point cloud while preserving
the essential characteristics of the data.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from topocore.pointcloud.pointcloud import PointCloud


class Sampler(ABC):
    """
    Abstract base class for point cloud samplers.

    Samplers reduce the number of points in a point cloud while
    preserving the essential characteristics.

    Examples
    --------
    >>> sampler = RandomSampler(fraction=0.1)
    >>> downsampled = sampler.sample(cloud)

    >>> sampler = VoxelSampler(voxel_size=0.5, method="centroid")
    >>> downsampled = sampler.sample(cloud)
    """

    @abstractmethod
    def sample(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """
        Sample/downsample the point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.

        Returns
        -------
        PointCloud
            Sampled point cloud.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the sampler name."""
        ...

    def __call__(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        """Callable interface."""
        return self.sample(cloud)


__all__ = [
    "Sampler",
]
