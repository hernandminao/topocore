"""
Regression suite for topocore.processing.cache.lru_cache.LRUCache --
PR19.

Includes a real bug found and fixed in this session: LRUCache holds
a genuine threading.RLock for thread-safety, but threading locks are
not picklable by design. Any object holding an LRUCache (e.g. a
trained ML classifier's internal FeatureManager) failed to pickle
with TypeError: "cannot pickle '_thread.RLock' object". This was
never reachable before a SEPARATE bug (MachineLearningClassifier
never registering feature computers) made every fit() call fail
immediately -- once that was fixed, the very next real call
(ClassificationManager.save()) hit this. Fixed with __getstate__/
__setstate__ excluding the lock and recreating it fresh on unpickle.
"""

from __future__ import annotations

import pickle

from topocore.processing.cache import LRUCache


def test_cache_survives_pickle_round_trip() -> None:
    """
    The exact regression this fix targets: before it, this raised
    TypeError: cannot pickle '_thread.RLock' object.
    """
    cache: LRUCache[str, int] = LRUCache(maxsize=10)
    cache.set("a", 1)
    cache.set("b", 2)

    restored: LRUCache[str, int] = pickle.loads(pickle.dumps(cache))

    assert restored.get("a") == 1
    assert restored.get("b") == 2
    assert restored.size() == 2
    assert restored.maxsize() == 10


def test_restored_cache_has_a_genuine_new_lock() -> None:
    """
    Confirms the lock is actually recreated (usable), not merely
    absent -- calling any method that acquires it must not raise.
    """
    cache: LRUCache[str, int] = LRUCache(maxsize=10)
    cache.set("a", 1)

    restored: LRUCache[str, int] = pickle.loads(pickle.dumps(cache))

    restored.set("b", 2)  # acquires the lock internally -- must not raise
    assert restored.contains("b")
    assert restored.remove("a") is True


def test_restored_cache_preserves_stats() -> None:
    cache: LRUCache[str, int] = LRUCache(maxsize=10)
    cache.set("a", 1)
    cache.get("a")  # hit
    cache.get("missing")  # miss

    restored: LRUCache[str, int] = pickle.loads(pickle.dumps(cache))
    stats = restored.stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_restored_cache_can_be_cleared() -> None:
    cache: LRUCache[str, int] = LRUCache(maxsize=10)
    cache.set("a", 1)

    restored: LRUCache[str, int] = pickle.loads(pickle.dumps(cache))
    restored.clear()

    assert restored.size() == 0
