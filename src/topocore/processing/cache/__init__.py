"""
topocore.processing.cache
=========================

Caching infrastructure for the processing subsystem.

This package provides caching mechanisms to store expensive computation
results (KD-tree indices, normal estimates, feature values, etc.),
significantly improving performance in multi-step processing pipelines.

The cache is designed to be transparent: algorithms check the cache
before computing results, and store results after computation.

Public API
----------
- Cache: abstract base class for caches.
- LRUCache: thread-safe LRU cache implementation.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from .base import Cache
from .lru_cache import LRUCache

__all__ = [
    "Cache",
    "LRUCache",
]
