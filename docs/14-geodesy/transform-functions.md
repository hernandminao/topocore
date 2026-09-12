# `transform_point_cloud`, `transform_survey`, `transform_feature_collection`

```python
from topocore.geodesy.transform import (
    transform_point_cloud,
    transform_survey,
    transform_feature_collection,
)
```

Plain functions, not a class — each takes `(model, transformer)` and
applies an **already-built** `CoordinateTransformer` to one of
TopoCore's three coordinate-carrying models. None of them construct
a transformer themselves; you always build one first (see
[`transformer.md`](./transformer.md)) and pass it in.

These are the only functions in TopoCore that actually move
coordinates from one CRS to another on a real domain object — `CRS`
and `CoordinateTransformer` alone only describe *how* to transform;
these functions are *what* does the transforming.

## The shared contract, verified for all three

- **Pure**: the input is never mutated. Verified directly for all
  three — the original `SurveyPointSet`/`PointCloud`/
  `FeatureCollection` retains its original coordinate values after
  the call.
- **No `crs` parameters**: only `(model, transformer)` — the
  source/target decision already lives inside the `transformer` you
  built.
- **`SurveyPointSet.crs` and `FeatureCollection.crs` are updated to
  the transformer's own target CRS.** Both fields exist today
  (`SurveyPointSet.crs: CRS | None`,
  `FeatureCollection.crs: str | None` — added after this page was
  first written). Confirmed directly:

```python
>>> survey.crs
CRS(EPSG:4326, 'WGS 84', Geographic)
>>> transformed = transform_survey(survey, transformer)
>>> transformed.crs
CRS(EPSG:32618, 'WGS 84 / UTM zone 18N', Projected)
>>> survey.crs   # original untouched
CRS(EPSG:4326, 'WGS 84', Geographic)
```

`SurveyPointSet.crs` is set to the transformer's own `target_crs`
object directly (matching its own type); `FeatureCollection.crs`
follows the same `f"EPSG:{code}"`-or-name string convention
`PointCloud.crs` already uses. Neither field is populated from
nothing, though — both remain `None` unless the source object
already had a CRS to carry forward, or was produced by a reader that
detected one (see
[`../13-io/crs-detection.md`](../13-io/crs-detection.md)) or by
`Workflow.georeference()` (see
[`georeferencing.md`](./georeferencing.md)).

## `transform_survey(survey, transformer)`

```python
>>> transformed = transform_survey(survey, transformer)
```

Every `SurveyPoint`'s `x`/`y`/`z` is passed through
`transformer.transform_point()`; `id` and `code` are carried over
unchanged. Returns a new `SurveyPointSet`.

## `transform_feature_collection(collection, transformer)`

```python
>>> transformed = transform_feature_collection(collection, transformer)
```

Every `Feature`'s geometry vertices are transformed via
`transformer.transform_array()`. `FeatureGeometry.__post_init__`
re-validates the result automatically (finite coordinates, correct
vertex count) — a transform that somehow produced invalid output
fails immediately here, not silently downstream. For a `MESH`
feature, `faces` (triangle indices into `vertices`) pass through
completely unchanged, since they're indices, not coordinates.

Confirmed directly: even though `FeatureCollection` is itself mutable
(`.add()`/`.extend()`/`.normalize_ids()` all mutate in place by
design), `transform_feature_collection()` never uses those — it
always builds a fresh `FeatureCollection`, leaving the original
untouched.

## `transform_point_cloud(cloud, transformer)`

```python
>>> transformed = transform_point_cloud(cloud, transformer)
```

Transforms every chunk's `X`/`Y`/`Z` via `transformer.transform_array()`;
every other attribute (`INTENSITY`, `CLASSIFICATION`, `COLOR`, ...)
is copied through unchanged. Confirmed directly:

```python
>>> cloud.metadata.crs
None
>>> transformed = transform_point_cloud(cloud, transformer)
>>> transformed.metadata.crs
'EPSG:32617'
>>> cloud.metadata.crs   # original untouched
None
```

`metadata.crs` on the result is set to the transformer's own
`target_crs` (as `"EPSG:{code}"` when the target CRS has one,
otherwise the CRS's own name) — following the exact same convention
`SurveyPointSet.crs`/`FeatureCollection.crs` now use too (see above).
`bounds` is recomputed for the new coordinates rather than left
stale.

## A documented discrepancy: the "TD-003" comment is stale

`transform_point_cloud()`'s own source comments explain that it
builds brand-new `Chunk` objects manually rather than calling
`PointCloud.clone()`, citing a historical defect ("TD-003": `clone()`
allegedly shared `Chunk` objects/arrays with the original rather than
deep-copying them).

**Confirmed directly, exhaustively, against the current codebase:
this is no longer true.** `PointCloud.clone()` and `Chunk.clone()`
both perform genuine, complete deep copies today — verified across
multiple chunks and multiple attributes, including array mutation
after cloning (mutating the original's arrays, or adding a new chunk
to the original, has zero effect on an already-taken clone).

This doesn't mean `transform_point_cloud()`'s current implementation
is wrong — it still produces correct output, just via a more manual
path than would be strictly necessary today. The comment referencing
"TD-003" is simply outdated documentation of a bug that has since
been fixed elsewhere in the codebase, not a live constraint on this
function. Following this project's own documentation discipline
(trust execution over comments when the two disagree), this is
recorded here as a documentation discrepancy, not corrected in the
source as part of this documentation pass.

## What these functions do *not* do

None of the three touch `topocore.workflow` directly — confirmed
directly in their own module's docstring ("Never touches
`topocore.workflow` -- these are geodesy-side utilities Workflow's
own callers can reach for outside the orchestrator, not a new
Workflow capability"). `Workflow` does have its own optional
`transform_crs()` stage that wraps these same three functions — see
[`workflow-integration.md`](./workflow-integration.md) — but these
functions themselves remain plain, `Workflow`-agnostic utilities.
