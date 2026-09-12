# Workflow Integration

Only 2 of the 10 `processing` submodules are wired into `Workflow` as
their own chainable stages. This page documents both, and is
explicit about what remains standalone.

## `Workflow.classify_ground(**manager_kwargs)`

```python
workflow.classify_ground(method="grid", cell_size=1.0)
```

Wraps `ground.GroundManager(**manager_kwargs).extract()` — never
`.classify()`, so the algorithm runs exactly once (see
[`ground.md`](./ground.md)). Requires `ArtifactType.POINT_CLOUD`,
produces `ArtifactType.GROUND_CLOUD`.

**Propagates the source `PointCloud`'s own `.crs` onto the result —
`GroundManager` itself does not do this.** Confirmed directly: this
was a real, confirmed gap found while building
[`transform_crs()`'s own safety check](../14-geodesy/workflow-integration.md)
— `GroundManager.extract()` constructs a fresh `PointCloud` for the
ground subset and never carried the source's own `.crs` forward.
Ground classification is a pure geometric subset (points removed,
none moved or reprojected), so the CRS of the remaining points is
genuinely unchanged; `Workflow.classify_ground()` sets
`ground_cloud.crs = cloud.crs` explicitly after calling
`GroundManager`, rather than relying on `GroundManager` to do it.

## `Workflow.classify_points(method="auto", **manager_kwargs)`

```python
workflow.classify_points(method="random_forest")
```

Wraps `classification.ClassificationManager(method=method,
**manager_kwargs).classify()`. Requires `ArtifactType.POINT_CLOUD`,
produces `ArtifactType.CLASSIFICATION_RESULT`. Distinct from
`classify_ground()`: this is multi-class ASPRS classification, not
the binary ground/non-ground split — the two stages produce different
`ArtifactType`s and can both run against the same source
`PointCloud`, independently.

## What has no `Workflow` stage of its own

`neighbors`, `normals`, `filters`, `features`, `sampling`,
`segmentation`, and `registration` are standalone utilities — you
call them directly on a `PointCloud` you already have (e.g. via
`workflow.artifact(ArtifactType.POINT_CLOUD)`), then work with the
result outside the `Workflow` chain, the same way
[`geodesy`'s own plain `transform_*` functions](../14-geodesy/transform-functions.md)
work before `Workflow.transform_crs()` wraps one of them. Several of
these are used *internally* by `classify_ground()`/`classify_points()`
(ground classification and the ML/rule-based classifiers both consume
`features`' geometric descriptors, for instance), but that internal
use doesn't expose them as their own stages.

## Relationship to CRS/geodesy — genuinely separate concerns

Processing operates purely on point coordinates and per-point
attributes; it has no CRS awareness of its own. A `PointCloud`'s own
`.crs` (populated by a reader's CRS detection, or by an earlier
`transform_crs()`/`georeference()` call — see
[`../14-geodesy/`](../14-geodesy/)) is simply carried through
unexamined by every Processing algorithm except where explicitly
noted above (`classify_ground()`'s own propagation). None of
`ground`, `classification`, `segmentation`, `sampling`, `filters`,
`normals`, `features`, `neighbors`, or `registration` reads, validates,
or reprojects based on a `PointCloud`'s own CRS — they operate on raw
`X`/`Y`/`Z` values, in whatever units and reference frame those
values already are in. If your workflow needs coordinates in a
specific CRS *before* running a Processing algorithm that's sensitive
to real-world distances (e.g. `DensitySampler`'s `target_density`, or
any `radius`/`cell_size` parameter across this section), reproject
first via `transform_crs()` — Processing itself will not do this for
you, and will not warn you if your data is in, say, geographic degrees
where a `radius` parameter intended for meters would silently produce
a meaningless result.

## Relationship to `analysis`

`topocore.analysis` (documented separately) consumes some of the same
building blocks Processing does — e.g. quality-comparison operations
may use neighbor search or point cloud distances similarly. The 2
packages are not fully independent, though: confirmed directly,
`topocore.analysis.exceptions` imports `ProcessingError` from
`topocore.processing.exceptions` — a shared exception-hierarchy
dependency, not a shared algorithm. No core `processing` algorithm
imports anything from `topocore.analysis`; the dependency runs one
way, and only at the exception-handling level.
