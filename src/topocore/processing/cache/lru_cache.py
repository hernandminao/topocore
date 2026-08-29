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

from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

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
        "_cache",
        "_hits",
        "_lock",
        "_maxsize",
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

    def __getstate__(self) -> dict[str, Any]:
        """
        Support pickling (e.g. ``joblib.dump`` on an object that
        holds an ``LRUCache``, such as a trained ML classifier's
        internal ``FeatureManager``) -- ``threading.RLock`` is not
        picklable by design, so it is deliberately excluded here and
        recreated fresh in ``__setstate__``, rather than attempting
        to (impossibly) serialize lock state across processes.

        Found and fixed in PR19: this was never reachable before a
        separate bug (``MachineLearningClassifier`` never registering
        any feature computers -- see that module's own PR19 notes)
        made every ``fit()`` call fail immediately; once that was
        fixed, the very next real call (``save()``, which pickles the
        whole classifier including its ``FeatureManager``'s
        ``LRUCache``) failed with ``TypeError: cannot pickle
        '_thread.RLock' object``.
        """
        return {
            "_maxsize": self._maxsize,
            "_cache": self._cache,
            "_hits": self._hits,
            "_misses": self._misses,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._maxsize = state["_maxsize"]
        self._cache = state["_cache"]
        self._hits = state["_hits"]
        self._misses = state["_misses"]
        self._lock = RLock()

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

    def remove_where(
        self,
        predicate: Callable[[K], bool],
    ) -> int:
        """
        Remove every entry whose key matches `predicate`.

        Added for FILTER-MANAGER-001: FilterManager needs to purge
        every cache entry tied to a specific PointCloud once that
        cloud is garbage collected (via a weakref finalizer), without
        knowing in advance which pipeline-stage keys it may have
        cached for that cloud. This is a purely additive method --
        it does not change the behavior of any existing method or
        any other current caller of this class (e.g. NormalManager,
        MachineLearningClassifier's own FeatureManager).

        Parameters
        ----------
        predicate
            Called with each key currently in the cache; entries for
            which it returns True are removed.

        Returns
        -------
        int
            The number of entries removed.
        """
        with self._lock:
            matching_keys = [key for key in self._cache if predicate(key)]
            for key in matching_keys:
                del self._cache[key]
            return len(matching_keys)

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
