# Neighbors

Spatial neighbor search over a `PointCloud` — the foundational
operation several other Processing submodules build on directly
(outlier filters, normal estimation, geometric feature computation,
segmentation, and ICP registration all query neighbors internally).

```python
from topocore.processing.neighbors import KDTreeNeighborSearch, NeighborhoodManager

search = KDTreeNeighborSearch.from_point_cloud(cloud)
indices, distances = search.knn(0, k=10)
```

## `NeighborSearch` — the abstract interface

```python
knn(index, k) -> (IntArray1D, FloatArray1D)              # indices, distances -- for one query point
knn_many(indices=None, k) -> (IntArray2D, FloatArray2D)   # batched; None means "every point"
radius(index, radius, *, include_self=False) -> IntArray1D          # indices ONLY -- no distances returned
radius_many(indices=None, radius, *, include_self=False) -> list[IntArray1D]
query_point(...) / query_points_many(...) / query_point_radius(...)  # array-input variants, not tied to a PointCloud's own indices
points() -> FloatArray2D
point_count() -> int
```

`KDTreeNeighborSearch` is the sole implementation, built on
`scipy.spatial.cKDTree`. Constructed via `from_point_cloud(cloud)` or
`from_array(xyz)` — never directly.

**`radius()` returns only indices, no distances** — confirmed
directly, unlike `knn()`, which always returns both. If you need
distances for a radius query, compute them yourself from the
returned indices and the query point's own coordinates.

## `NeighborhoodManager` — caching wrapper, plus density

```python
manager = NeighborhoodManager(search)
manager.knn(0, k=10)             # same methods as NeighborSearch -- NOT cached, delegates directly
manager.local_density(index, radius) -> float   # points per unit area/volume within radius -- IS cached
manager.local_density_many(indices=None, radius) -> FloatArray1D
manager.clear_cache()
manager.config                    # the cache configuration in effect
```

Every query method mirrors `NeighborSearch`'s own signature exactly.
**Only `local_density()`/`local_density_many()` are actually cached**
— confirmed directly by reading the source: `knn()`/`knn_many()`/
`radius()`/`radius_many()` all delegate straight to the underlying
`NeighborSearch` on every call, with no caching at all. The density
cache itself is a plain `dict` keyed by `(point_index,
effective_radius)` — not the `LRUCache` used elsewhere in this
section (see [`backend-and-cache.md`](./backend-and-cache.md)), and
with no eviction: `clear_cache()` empties it manually, there is no
automatic size limit.

## Used internally by

`filters` (statistical/radius outlier removal), `normals` (PCA needs
a local neighborhood), `features` (density and PCA-derived shape
descriptors), `segmentation` (region growing needs adjacency), and
`registration` (ICP's own correspondence search) all construct and
query a `NeighborSearch`/`NeighborhoodManager` internally — you do
not need to build one yourself unless you're calling `neighbors`
directly for your own purpose.
