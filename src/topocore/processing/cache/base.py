"""
topocore.processing.cache.base
==============================

Abstract base class for caching in the processing subsystem.

This module defines the interface for caching expensive computation
results (KD-tree indices, normal estimates, feature values, etc.).
The cache is designed to be transparent: if a result is available,
it is returned immediately; otherwise, it is computed and stored.

The caching layer is essential for performance, as many processing
operations (e.g., neighbor search, normal estimation) are expensive
and may be reused multiple times in a pipeline.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Cache(ABC, Generic[K, V]):
    """
    Abstract interface for a key-value cache.

    The cache stores computed results and retrieves them by key.
    Keys are typically tuples of (operation, parameters, cloud_id)
    or similar unique identifiers.

    Implementations must be thread-safe if used in multi-threaded
    environments.

    Examples
    --------
    >>> cache = LRUCache(maxsize=100)
    >>> result = cache.get(("knn", 8, "cloud_001"))
    >>> if result is None:
    ...     result = compute_knn(cloud, k=8)
    ...     cache.set(("knn", 8, "cloud_001"), result)
    """

    @abstractmethod
    def get(
        self,
        key: K,
    ) -> V | None:
        """
        Retrieve a value from the cache.

        Parameters
        ----------
        key
            The cache key.

        Returns
        -------
        V | None
            The cached value, or None if not found.
        """
        ...

    @abstractmethod
    def set(
        self,
        key: K,
        value: V,
    ) -> None:
        """
        Store a value in the cache.

        Parameters
        ----------
        key
            The cache key.
        value
            The value to store.
        """
        ...

    @abstractmethod
    def contains(
        self,
        key: K,
    ) -> bool:
        """
        Check if a key exists in the cache.

        Parameters
        ----------
        key
            The cache key.

        Returns
        -------
        bool
            True if the key exists.
        """
        ...

    @abstractmethod
    def remove(
        self,
        key: K,
    ) -> bool:
        """
        Remove a key from the cache.

        Parameters
        ----------
        key
            The cache key.

        Returns
        -------
        bool
            True if the key was removed.
        """
        ...

    @abstractmethod
    def clear(
        self,
    ) -> None:
        """
        Clear the entire cache.
        """
        ...

    @abstractmethod
    def size(
        self,
    ) -> int:
        """
        Return the number of entries in the cache.

        Returns
        -------
        int
            Number of entries.
        """
        ...

    @abstractmethod
    def maxsize(
        self,
    ) -> int:
        """
        Return the maximum number of entries.

        Returns
        -------
        int
            Maximum size, or -1 for unlimited.
        """
        ...

    @abstractmethod
    def stats(
        self,
    ) -> dict[str, int]:
        """
        Return cache statistics.

        Returns
        -------
        dict
            Statistics including hits, misses, and current size.
        """
        ...


__all__ = [
    "Cache",
]
