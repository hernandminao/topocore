# `Workflow` Integration: `transform_crs()`

Everything else in this section describes `topocore.geodesy` as a
standalone package. This page documents the one point where it meets
`Workflow` directly: an optional, explicit, chainable stage that
applies an already-built `CoordinateTransformer` to whichever
artifact you name.

## What this is — and, just as importantly, what it is not

```python
workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)
```

This closes exactly one gap: before this stage existed, applying a
transform inside a `Workflow` chain meant dropping out of the chain
entirely —

```python
# the only way to do this before transform_crs() existed
cloud = workflow.artifact(ArtifactType.POINT_CLOUD)
transformed = transform_point_cloud(cloud, transformer)
# ...then working with `transformed` independently of `workflow`
```

`transform_crs()` lets this happen as a normal chained stage instead,
writing the result back into the same `Workflow`'s own artifact
store, dependency-tracked exactly like every other stage.

**This does not mean `Workflow` decides CRS-to-CRS transformation
for you.** It still does not: there is no
`input_crs`/`processing_crs`/`output_crs` model, and `Workflow` never
constructs a `CRS` or `CoordinateTransformer` on your behalf for this
stage. You still decide the source and target CRS yourself and still
build the `CoordinateTransformer` yourself — this stage only removes
the need to step outside the chain to apply it. (Detecting what CRS
an artifact is *already* in, when its source file declared one, does
now happen automatically at read time — see
[`../13-io/crs-detection.md`](../13-io/crs-detection.md) — but that
is a separate, earlier concern from this stage's own job of applying
a transformation you've already decided on.)

## Supported artifact types — exhaustive, verified

`transform_crs()` supports exactly the 4 types
`topocore.geodesy.transform` has a function for:

| `ArtifactType` | Dispatches to |
|---|---|
| `POINT_CLOUD` | `transform_point_cloud()` |
| `GROUND_CLOUD` | `transform_point_cloud()` (same `PointCloud` type, confirmed directly) |
| `SURVEY_POINT_SET` | `transform_survey()` |
| `FEATURE_COLLECTION` | `transform_feature_collection()` |

Any other type is rejected explicitly, not silently ignored:

```python
>>> workflow.transform_crs(ArtifactType.TIN, transformer)
WorkflowStateError: transform_crs() does not support tin; supported
types are: point_cloud, ground_cloud, survey_point_set, feature_collection.
```

`TIN`, `DTM`, `CONTOURS`, and `CLASSIFICATION_RESULT` have no
transform function anywhere in `topocore.geodesy` — this is not an
oversight in this stage, it reflects a genuine gap in
`topocore.geodesy.transform` itself.

## Worked example

```python
from topocore.workflow import Workflow
from topocore.workflow.artifacts import ArtifactType
from topocore.geodesy import CRS, CoordinateTransformer
from topocore.survey.formats import SurveyFormat

transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))

workflow = (
    Workflow()
    .read_survey("survey.csv", format=SurveyFormat.ID_XYZ_CODE)
    .build_features_from_survey()
    .resolve_sides()
)
workflow.transform_crs(ArtifactType.FEATURE_COLLECTION, transformer)

features = workflow.artifact(ArtifactType.FEATURE_COLLECTION)
```

Confirmed directly: this exact chain (`read_survey` →
`build_features_from_survey` → `resolve_sides` → `transform_crs`)
runs end to end, and the resulting features carry both the resolved
`side` attribute and their transformed, UTM-projected coordinates.

## A real bug this feature's own implementation found and fixed

Building `transform_crs()` surfaced a genuine defect in shared
`Workflow` infrastructure, not specific to this stage: any stage that
both reads and re-produces the **same** `ArtifactType` (this one, and
the earlier `resolve_sides()`) recorded a dependency on that type's
own pre-call version. By the time staleness was checked, the store's
current version for that type was already the *new* one the same
stage had just written — so the comparison always found a mismatch
and reported the artifact stale immediately after its own successful
production.

Confirmed directly: calling `workflow.artifact(ArtifactType.FEATURE_COLLECTION)`
right after `resolve_sides()` raised `StaleArtifactError`
unconditionally — a real, undetected regression from when
`resolve_sides()` was added, missed at the time because its own test
suite read via the internal `_store.get()` directly rather than the
public `artifact()` accessor. Fixed in
`WorkflowValidator._is_stale()` by skipping a dependency in the
staleness comparison whenever it refers to the same `ArtifactType`
currently being checked — confirmed by direct reasoning that no
legitimate staleness scenario depends on this specific
self-referential comparison, and confirmed by testing that genuine
staleness (e.g. rebuilding `TIN` after `FEATURE_COLLECTION` already
depended on it) is still correctly detected after the fix.

## Explicitly out of scope

These remain real, open product decisions, not limitations accepted
as permanent:

- **An `input_crs`/`processing_crs`/`output_crs` model on `Workflow`
  itself** — a genuine architectural decision touching every existing
  stage, not something to decide unilaterally inside a single
  feature's implementation.

## A related, separate stage: `Workflow.georeference()`

`transform_crs()` requires both a source and target CRS to already
be known. If your coordinates are local or arbitrary (no CRS at
all), and you have known correspondences between a set of points in
your local system and their real-world coordinates,
`Workflow.georeference()` is the stage for that — a genuinely
different operation (it *estimates* a transformation, rather than
applying an already-known one). See
[`georeferencing.md`](./georeferencing.md) for the complete
capability, including `ControlPoint`, automatic strategy selection,
and how it composes with `transform_crs()` and vertical
transformation.

## `transform_crs()` never touches vertical datum

`transform_crs()`, like every function in
[`transform-functions.md`](./transform-functions.md), passes `Z`
straight through unmodified — it reprojects `X`/`Y` only. Converting
between ellipsoidal and orthometric height is a separate, now-
implemented stage, `Workflow.transform_vertical()` — see
[`vertical-reference.md`](./vertical-reference.md).
