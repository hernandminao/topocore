"""
topocore.processing.features.manager
====================================

Feature manager for computing and caching features.

This module provides a high-level manager that orchestrates feature
computation, caching, and retrieval. The manager handles:
- On-demand feature computation
- Caching of computed features
- Batch computation of multiple features

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from typing import Any, TypeAlias

from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.cache import LRUCache
from topocore.processing.exceptions import PointDescriptorError

from .base import FeatureComputer

CacheKey: TypeAlias = tuple[int, str, frozenset[str]]


class FeatureManager:
    """
    High-level manager for feature computation and caching.

    This class provides a unified interface for computing and caching
    features. It supports:
    - Computing a single feature on-demand
    - Computing multiple features in batch
    - Caching results to avoid recomputation
    - Clearing the cache

    Examples
    --------
    >>> manager = FeatureManager()
    >>> manager.register("height", HeightFeatureComputer())
    >>> height = manager.compute(cloud, "height")
    >>> features = manager.compute_all(cloud, ["height", "density"])
    """

    __slots__ = (
        "_computers",
        "_cache",
        "_cloud_id",
    )

    def __init__(
        self,
        cloud: PointCloud | None = None,
        cache_size: int = 64,
    ) -> None:
        self._computers: dict[str, FeatureComputer] = {}
        self._cache: LRUCache[CacheKey, Any] = LRUCache(maxsize=cache_size)
        self._cloud_id: int = id(cloud) if cloud is not None else 0

        if cloud is not None:
            self._compute_features(cloud)

    def register(
        self,
        name: str,
        computer: FeatureComputer,
    ) -> None:
        """
        Register a feature computer.
        """
        self._computers[name] = computer

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Unregister a feature computer.

        Returns
        -------
        bool
            True if the feature was registered and removed.
        """
        if name in self._computers:
            del self._computers[name]
            return True

        return False

    def compute(
        self,
        cloud: PointCloud,
        name: str,
        **kwargs: Any,
    ) -> Any:
        """
        Compute a single feature.
        """
        if name not in self._computers:
            raise PointDescriptorError(f"Feature '{name}' is not registered.")

        cache_key: CacheKey = (
            self._cloud_id,
            name,
            frozenset(kwargs.keys()),
        )

        cached = self._cache.get(cache_key)

        if cached is not None:
            return cached

        computer = self._computers[name]

        result = computer.compute(cloud)

        self._cache.set(
            cache_key,
            result,
        )

        return result

    def compute_all(
        self,
        cloud: PointCloud,
        names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Compute multiple features at once.
        """
        if names is None:
            names = list(self._computers.keys())

        results: dict[str, Any] = {}

        for name in names:
            if name in self._computers:
                try:
                    results[name] = self.compute(
                        cloud,
                        name,
                        **kwargs,
                    )
                except Exception as e:
                    raise PointDescriptorError(f"Failed to compute feature '{name}': {e}") from e

        return results

    def clear_cache(
        self,
    ) -> None:
        """Clear the feature cache."""
        self._cache.clear()

    def is_registered(
        self,
        name: str,
    ) -> bool:
        """
        Check if a feature is registered.
        """
        return name in self._computers

    def list_features(
        self,
    ) -> list[str]:
        """
        Return the list of registered feature names.
        """
        return list(self._computers.keys())

    def _compute_features(
        self,
        cloud: PointCloud,
    ) -> None:
        """
        Compute all features for a cloud during initialization.
        """
        for name, computer in self._computers.items():
            cache_key: CacheKey = (
                self._cloud_id,
                name,
                frozenset[str](),
            )

            self._cache.set(
                cache_key,
                computer.compute(cloud),
            )

    def __getitem__(
        self,
        name: str,
    ) -> FeatureComputer:
        """
        Get a feature computer by name.
        """
        if name not in self._computers:
            raise KeyError(f"Feature '{name}' is not registered.")

        return self._computers[name]

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return name in self._computers


__all__ = [
    "FeatureManager",
]
