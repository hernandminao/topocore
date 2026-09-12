# Segmentation

Clustering a `PointCloud` into discrete segments — 3 general-purpose
clustering algorithms, plus 2 domain-specific segmenters built on top
of them.

```python
from topocore.processing.segmentation import DBSCANSegmenter

result = DBSCANSegmenter(eps=1.0, min_samples=5).segment(cloud)
```

**The method is `.segment(cloud)`.** Verified directly on 2
well-separated synthetic clusters (30 points each, centered ~50 units
apart): `DBSCANSegmenter` correctly identifies exactly 2 segments.

## `SegmentationResult`

```python
result.labels          # IntArray1D, shape (N,) -- one segment id per point (-1 = noise, where applicable)
result.num_segments    # int
result.segment_sizes   # point count per segment
result.has_noise       # bool
result.get_segment(id) -> PointCloud     # points belonging to one segment
result.get_segments() -> list[PointCloud]  # every segment, materialized
```

## 3 general-purpose segmenters

```python
DBSCANSegmenter(eps=0.5, min_samples=5, use_adaptive_eps=False, include_noise=True, cache_neighbors=True)
ConnectedComponentsSegmenter(distance_threshold=0.5, min_points=10, use_adaptive_threshold=False)
RegionGrowingSegmenter(k=10, curvature_threshold=0.05, normal_angle_threshold=15.0, min_region_size=10, max_region_size=1_000_000, use_adaptive_k=False)
```

`DBSCANSegmenter`/`ConnectedComponentsSegmenter` cluster purely by
spatial proximity. `RegionGrowingSegmenter` additionally requires
locally consistent normals/curvature (it computes these internally via
`normals`) — appropriate for separating genuinely distinct surfaces
that happen to be spatially close (e.g. two walls meeting at a corner),
which a purely distance-based method would merge into one segment.

### `RegionGrowingSegmenter` — two classes share this name; import from the package, not `.base`

`topocore.processing.segmentation.base.RegionGrowingSegmenter` is
abstract. `topocore.processing.segmentation.region_growing.RegionGrowingSegmenter`
is the concrete implementation, and is also what
`from topocore.processing.segmentation import RegionGrowingSegmenter`
gives you — confirmed directly. See
[`overview.md`](./overview.md) for why this pattern exists and where
else it recurs.

## 2 domain-specific segmenters, built on the general ones

```python
TreeSegmenter(min_height=0.5, max_height=50.0, eps=0.5, min_samples=5, min_points_per_tree=10, ground_method="grid")
BuildingSegmenter(min_height=1.0, max_height=100.0, k=10, curvature_threshold=0.02, normal_angle_threshold=10.0, min_points_per_building=100, ground_method="grid")
```

Both first classify ground internally (via `ground.GroundManager`,
selected by `ground_method`) to compute relative height, filter to a
plausible height band (`min_height`/`max_height`), then cluster the
remainder — `TreeSegmenter` via DBSCAN-style proximity,
`BuildingSegmenter` via region growing (buildings have planar,
low-curvature roof/wall surfaces; trees don't). Neither takes a
pre-classified ground cloud as input — you pass the full `PointCloud`,
and `ground_method` controls how ground gets separated internally.

## `SegmentationManager`

Selects between the segmenters above by name, matching the manager
pattern used throughout this section.
