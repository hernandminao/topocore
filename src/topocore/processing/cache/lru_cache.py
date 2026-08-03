"""
topocore.processing.cache.lru_cache
===================================

LRU (Least Recently Used) cache implementation.

This module provides a thread-safe LRU cache using `functools.lru_cache`
as the underlying storage mechanism. The cache automatically evicts
the least recently used entries when it reaches the maximum size.

The cache is designed for use in the processing subsystem to store
expensive computation results.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from threading import RLock
from typing import TypeVar

from .base import Cache

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Cache[K, V]):
    """
    Thread-safe LRU (Least Recently Used) cache.

    This cache uses a dictionary and maintains access order to evict
    the least recently used entries when the maximum size is reached.

    Parameters
    ----------
    maxsize
        Maximum number of entries. Use -1 for unlimited.

    Examples
    --------
    >>> cache = LRUCache(maxsize=100)
    >>> cache.set("key", "value")
    >>> cache.get("key")
    'value'
    >>> cache.size()
    1
    """

    __slots__ = (
        "_maxsize",
        "_cache",
        "_lock",
        "_hits",
        "_misses",
    )

    def __init__(
        self,
        maxsize: int = -1,
    ) -> None:
        self._maxsize = maxsize
        self._cache: dict[K, V] = {}
        self._lock = RLock()
        self._hits: int = 0
        self._misses: int = 0

    def get(
        self,
        key: K,
    ) -> V | None:
        """
        Retrieve a value from the cache.

        This method updates the access order of the key (making it
        recently used) if the key exists.
        """
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used) by re-inserting
                value = self._cache.pop(key)
                self._cache[key] = value
                return value

            self._misses += 1
            return None

    def set(
        self,
        key: K,
        value: V,
    ) -> None:
        """
        Store a value in the cache.

        If the cache exceeds its maximum size, the least recently used
        entry is evicted.
        """
        with self._lock:
            if key in self._cache:
                # Remove old entry to update access order
                self._cache.pop(key)

            self._cache[key] = value

            # Evict if over limit
            if self._maxsize > 0 and len(self._cache) > self._maxsize:
                # Pop the first item (least recently used)
                first_key = next(iter(self._cache))
                self._cache.pop(first_key)

    def contains(
        self,
        key: K,
    ) -> bool:
        """Check if a key exists in the cache."""
        with self._lock:
            return key in self._cache

    def remove(
        self,
        key: K,
    ) -> bool:
        """Remove a key from the cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                return True
            return False

    def clear(
        self,
    ) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def size(
        self,
    ) -> int:
        """Return the number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def maxsize(
        self,
    ) -> int:
        """Return the maximum number of entries."""
        return self._maxsize

    def stats(
        self,
    ) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }


__all__ = [
    "LRUCache",
]
