# Filters

5 point-cloud filters, all sharing the same `.apply(cloud) -> PointCloud`
contract (a `BinaryFilter`: each point is independently kept or
dropped, never moved or modified).

```python
from topocore.processing.filters import StatisticalOutlierFilter

filtered = StatisticalOutlierFilter(k=10, std_ratio=1.0).apply(cloud)
```

Verified directly: a cloud of 100 tightly-clustered points plus 2
extreme outliers (`x = ±50`, versus a cluster std of `~0.1`) is
correctly reduced to exactly 100 points.

## The 5 filters and their real constructor signatures

```python
StatisticalOutlierFilter(k=8, std_ratio=1.0, min_points=3, strict=False)
RadiusOutlierFilter(radius=1.0, min_neighbors=4, include_self=True, min_points=3, strict=False)
PassThroughFilter(axis: Axis, min_value: float, max_value: float)
CropBoxFilter(box: BBox3D)
ClipPolygonFilter(polygon: FloatArray1D)
```

**`PassThroughFilter` takes `axis: Axis`, not an attribute name or
string.** `Axis` is its own enum (`Axis.X`/`Axis.Y`/`Axis.Z`) —
confirmed directly, passing a `PointAttribute` or a bare string
raises `TypeError`, not a validation error with a helpful message.

```python
from topocore.processing.filters import Axis
PassThroughFilter(axis=Axis.X, min_value=-1.0, max_value=1.0)
```

`CropBoxFilter` takes a real `BBox3D` (`topocore.geometry.BBox3D`),
not 6 separate min/max keyword arguments.

## Statistical vs. radius outlier removal — genuinely different criteria

`StatisticalOutlierFilter` removes a point if its own mean distance
to its `k` nearest neighbors deviates from the *cloud-wide* mean by
more than `std_ratio` standard deviations — an adaptive, density-aware
criterion. `RadiusOutlierFilter` removes a point if it has fewer than
`min_neighbors` neighbors within a fixed `radius` — a purely local,
absolute criterion with no notion of the cloud's overall density.
Choose based on whether your outliers are "far from everything" (either
filter catches this) or "in a locally sparse region despite a
reasonable local neighbor count" (only `StatisticalOutlierFilter`
adapts to that).

## `strict` / `min_points` — a safety check on the filter's own output, not on the input cloud's size

Both outlier filters accept `min_points: int = 3` and
`strict: bool = False`. This checks how many points the filter's own
result *would* contain — not the size of the input cloud. If applying
the filter would drop the cloud below `min_points`, `strict=False`
returns the **original, unfiltered** cloud instead (silently
declining to over-filter); `strict=True` raises `FilterError`
instead. Confirmed directly, with a cloud whose outliers are
aggressive enough that filtering would leave only 16 of 25 points
(below a deliberately-set `min_points=100`):

```python
>>> StatisticalOutlierFilter(k=5, std_ratio=0.5, min_points=100, strict=False).apply(cloud).point_count
25   # unfiltered -- filtering would have dropped below min_points
>>> StatisticalOutlierFilter(k=5, std_ratio=0.5, min_points=100, strict=True).apply(cloud)
FilterError: Filter would keep only 16 points, below minimum 100.
```

This is unrelated to whether the input cloud has enough points to
form a neighborhood at all — requesting `k` neighbors from a cloud
with fewer than `k` points raises `NeighborError` from the underlying
neighbor search itself, regardless of `strict`/`min_points`.

## `FilterManager`

A caching wrapper around repeated filter application, following the
same manager pattern as `neighbors.NeighborhoodManager` — see
[`backend-and-cache.md`](./backend-and-cache.md) for the shared
`LRUCache` infrastructure both build on (`FilterManager` is
confirmed to have real, if indirect, test coverage of that cache
through its own test suite).
