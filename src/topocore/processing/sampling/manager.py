"""
topocore.processing.sampling.manager
====================================

Sampling manager with automatic method selection.

This module provides a high-level manager that selects the appropriate
sampling method based on the point cloud characteristics and user
preferences.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import SamplingError

from .base import Sampler
from .density import DensitySampler
from .random import RandomSampler
from .stratified import StratifiedSampler
from .uniform import UniformSampler
from .voxel import VoxelSampler

SamplerFactory = Callable[..., Sampler]


class SamplingManager:
    """
    High-level manager for point cloud sampling.

    This class provides a unified interface for sampling with
    automatic method selection.

    Supported methods:
    - "random": random sampling
    - "uniform": systematic decimation
    - "voxel": voxel grid sampling
    - "stratified": stratified sampling (2D grid)
    - "density": density-adaptive sampling

    Examples
    --------
    >>> manager = SamplingManager(method="voxel", voxel_size=0.5)
    >>> downsampled = manager.sample(cloud)

    >>> manager.method = "stratified"
    >>> manager.cell_size = 1.0
    >>> downsampled = manager.sample(cloud)
    """

    __slots__ = (
        "_method",
        "_params",
    )

    _SUPPORTED_METHODS: dict[str, SamplerFactory] = {
        "random": RandomSampler,
        "uniform": UniformSampler,
        "voxel": VoxelSampler,
        "stratified": StratifiedSampler,
        "density": DensitySampler,
    }

    def __init__(
        self,
        method: str = "voxel",
        **kwargs: Any,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise SamplingError(f"Unsupported method: {method}. Supported: {list(self._SUPPORTED_METHODS.keys())}")

        self._method = method
        self._params = kwargs

    @property
    def method(self) -> str:
        """Get the current method."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set the method."""
        if value not in self._SUPPORTED_METHODS:
            raise SamplingError(f"Unsupported method: {value}. Supported: {list(self._SUPPORTED_METHODS.keys())}")
        self._method = value

    def set_params(
        self,
        **kwargs: Any,
    ) -> None:
        """Set parameters for the current method."""
        self._params.update(kwargs)

    def sample(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> PointCloud:
        """
        Sample/downsample the point cloud.

        Parameters
        ----------
        cloud
            Input point cloud.
        **kwargs
            Additional arguments passed to the sampler.

        Returns
        -------
        PointCloud
            Sampled point cloud.
        """
        sampler = self._create_sampler(**kwargs)
        return sampler.sample(cloud)

    def _create_sampler(
        self,
        **kwargs: Any,
    ) -> Sampler:
        """Create the appropriate sampler instance."""

        params = self._params.copy()
        params.update(kwargs)

        sampler_factory: SamplerFactory = self._SUPPORTED_METHODS[self._method]

        if self._method == "random":
            # PR21 remediation (SAMPLING-MANAGER-001): unlike the
            # other 4 methods, "random" has no safe universal
            # default for its required parameter -- uniform/voxel/
            # stratified/density all default to a spatial unit
            # (step, voxel_size, cell_size, target_density), which
            # is meaningful independent of dataset size; "fraction"
            # or "count" are not: neither a proportion nor an
            # absolute count has a neutral, dataset-independent
            # default. Confirmed directly, before this change, that
            # omitting both silently produced RandomSampler's own
            # "Specify exactly one of fraction or count, not both."
            # -- a confusing message given the caller specified
            # NEITHER, not both. This validation raises a clear,
            # manager-level error instead, in the correct layer
            # (SamplingManager, not RandomSampler, which keeps its
            # own validation unchanged as a second line of defense).
            # No default value is invented for either parameter.
            if params.get("fraction") is None and params.get("count") is None:
                raise SamplingError(
                    "Method 'random' requires an explicit 'fraction' or 'count' parameter -- "
                    "unlike the other sampling methods, there is no safe default proportion "
                    "or count to assume."
                )

            return sampler_factory(
                fraction=params.get("fraction"),
                count=params.get("count"),
                seed=params.get("seed"),
            )

        if self._method == "uniform":
            return sampler_factory(
                step=params.get("step", 10),
                offset=params.get("offset", 0),
            )

        if self._method == "voxel":
            return sampler_factory(
                voxel_size=params.get("voxel_size", 1.0),
                method=params.get("method", "centroid"),
                seed=params.get("seed"),
            )

        if self._method == "stratified":
            return sampler_factory(
                cell_size=params.get("cell_size", 1.0),
                samples_per_cell=params.get("samples_per_cell", 1),
                method=params.get("method", "random"),
                seed=params.get("seed"),
            )

        if self._method == "density":
            return sampler_factory(
                target_density=params.get("target_density", 10.0),
                radius=params.get("radius", 1.0),
                min_fraction=params.get("min_fraction", 0.1),
                max_fraction=params.get("max_fraction", 1.0),
                seed=params.get("seed"),
            )

        raise SamplingError(f"Unsupported method: {self._method}")

    def __call__(
        self,
        cloud: PointCloud,
        **kwargs: Any,
    ) -> PointCloud:
        """Callable interface."""
        return self.sample(cloud, **kwargs)


__all__ = [
    "SamplingManager",
]
