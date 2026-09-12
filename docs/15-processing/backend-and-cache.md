# Backend and Cache

Shared infrastructure used by several Processing algorithms — neither
is a processing algorithm itself.

## `NumPyBackend`

```python
from topocore.processing.backend import NumPyBackend
```

An abstraction over array operations (`array`, `matmul`, `eigh`,
`svd`, `qr`, `mean`, `std`, `concatenate`, and about 30 others) behind
an ABC (`Backend`), with `NumPyBackend` as the sole current
implementation. The purpose is architectural: algorithms elsewhere in
`processing` call through this interface rather than `numpy` directly,
so a different backend (e.g. a GPU-accelerated one) could be
substituted later without changing every algorithm that does linear
algebra.

**Confirmed directly: `NumPyBackend` has no dedicated test file of its
own anywhere in the test suite** — unlike every other component in
this section. This does not mean it's unused or unverified in
practice (every algorithm relying on PCA, SVD, or similar linear
algebra runs through it, and those algorithms' own tests exercise it
indirectly), but there is no test file whose own explicit purpose is
verifying `NumPyBackend`'s own methods in isolation.

## `LRUCache`

```python
from topocore.processing.cache import LRUCache

cache = LRUCache(maxsize=2)
cache.set("a", 1)
cache.set("b", 2)
cache.get("a")          # "a" is now the most recently used
cache.set("c", 3)       # evicts "b" -- the least recently used, not "a"
cache.contains("a")     # True
cache.contains("b")     # False -- evicted
```

Genuine least-recently-used eviction, verified directly: accessing an
entry via `get()` refreshes its own recency, protecting it from
eviction ahead of an entry that was set earlier but never read again.
`maxsize=-1` (the default) means unbounded — no eviction ever happens
unless you set an explicit limit.

```python
cache.size()          # current entry count
cache.stats()          # {"hits": int, "misses": int, "size": int, "maxsize": int}
cache.remove(key)
cache.remove_where(predicate)   # bulk removal by a condition on keys
cache.clear()
```

**Confirmed directly: `LRUCache` has real, if indirect, test
coverage** — through `filters.FilterManager`'s own test suite, which
exercises the caching behavior `FilterManager` builds on top of
`LRUCache` for.

## Which managers actually use `LRUCache` — verified per module, not assumed uniform

Confirmed directly, module by module: `normals.NormalManager`,
`ground.GroundManager`, `features.FeatureManager`, and
`filters.FilterManager` all build on `LRUCache` directly.
`neighbors.NeighborhoodManager` does **not** — its own
`local_density()`/`local_density_many()` use a separate, simpler
plain `dict` keyed by `(point_index, radius)`, with no automatic
eviction (see [`neighbors.md`](./neighbors.md)); its `knn()`/`radius()`
methods aren't cached at all. `sampling.SamplingManager`,
`segmentation.SegmentationManager`, and
`classification.ClassificationManager` use neither — confirmed no
`LRUCache` import in any of the three. Do not assume every manager in
this section caches the same way, or caches at all.
