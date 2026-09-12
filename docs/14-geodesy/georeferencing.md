# Georeferencing by Control Points

This is a fourth, distinct operation from CRS detection,
declaration/propagation, and transformation (see
[`overview.md`](./overview.md) for how all four relate). Detection
reads a CRS a file already declares; transformation converts
coordinates between two *already-known* CRS. Georeferencing is
different: it **estimates** a transformation from local or arbitrary
coordinates to a real CRS, using known correspondences between a set
of points in both systems — and then applies that estimated
transformation to real data.

```text
ControlPoint[] (source, in your local/arbitrary system;
                target, in the real CRS you want)
        │
        ▼
fit_georeferencing(controls, options)
        │
        ▼
GeoreferencingResult
        ├── strategy    (which model was actually used)
        ├── parameters  (the fitted Helmert values)
        ├── residuals   (per-control-point fit error)
        ├── rms         (overall fit quality; None for a single point)
        ├── control_count
        └── warning     (None only when fully determined)
        │
        ▼
apply_georeferencing_to_point_cloud() / _to_survey() / _to_feature_collection()
        │
        ▼
new artifact, in the target CRS
```

## `ControlPoint`

```python
from topocore.geodesy.georeferencing import ControlPoint

control = ControlPoint(
    id="P1",
    source=(0.0, 0.0, 0.0),      # local/arbitrary coordinates
    target=(1000.0, 2000.0, 100.0),  # real-world coordinates, in the target CRS
)
```

`id` is for traceability only — it is never interpreted, and duplicate
ids across control points do not cause any internal mismatch
(residuals are matched to controls by position, not by looking up
`id`). If you reuse an `id`, you only lose your own ability to tell
which of two same-id residuals is which — the fit itself is
unaffected.

## Strategy selection — automatic, by rank

`fit_georeferencing()` never asks which model to use. It fits the
full 3D Helmert model first, and falls back only if the control
points' own geometry genuinely cannot support it:

```text
rank_3d == 7  ->  HELMERT_3D   (full: translation + rotation + scale)
rank_3d < 7   ->  evaluate rank_2d independently
                    rank_2d == 4  ->  HELMERT_2D   (translation + Z-rotation + XY-scale; Z = Z + tz only)
                    rank_2d < 4   ->  TRANSLATION_ONLY  (tx/ty/tz only)
```

`rank_3d < 7` does **not** imply `rank_2d < 4`, and vice versa — both
are checked independently, never inferred from one another. Two
concrete, verified cases:

- **3 non-collinear points are enough for `HELMERT_3D`.** They do
  not need to be non-coplanar — 3 genuinely non-collinear points
  already produce rank 7 in the linearized 3D formulation.
- **2 points need a sufficiently distinct XY projection for
  `HELMERT_2D`.** Two points that are vertically stacked (same X/Y,
  different Z) are degenerate for both the 3D **and** the 2D
  strategy — confirmed directly — and fall all the way through to
  `TRANSLATION_ONLY`.

## Underconstrained geometry requires explicit authorization

`HELMERT_2D` and `TRANSLATION_ONLY` both mean the fit is assuming
values it did not actually estimate (`HELMERT_2D` assumes `rx = ry =
0`; `TRANSLATION_ONLY` assumes no rotation or scale at all).
`fit_georeferencing()` never falls back to either silently:

```python
from topocore.geodesy.georeferencing import GeoreferencingOptions
from topocore.geodesy.georeferencing.exceptions import UnderconstrainedGeoreferencingError

# 1 control point, no authorization:
fit_georeferencing(controls, GeoreferencingOptions())
# UnderconstrainedGeoreferencingError: fit_georeferencing(): only an
# under-determined TRANSLATION_ONLY fit is available for these 1
# control point(s)...

# same call, explicitly authorized:
result = fit_georeferencing(controls, GeoreferencingOptions(accept_underconstrained=True))
```

`GeoreferencingOptions` has 2 fields:

```python
GeoreferencingOptions(
    accept_underconstrained: bool = False,
    minimum_control_distance: float | None = None,  # XY-only, same units as input; None = no check
)
```

`minimum_control_distance` rejects control points that are too close
together horizontally (`ControlPointsTooCloseError`) — a separate
safety check from the rank/authorization one above, since points can
be technically non-collinear yet still close enough together that
small survey errors would produce a wildly unstable fit.

## `GeoreferencingResult`

```python
GeoreferencingResult(
    strategy: GeoreferencingStrategy,   # HELMERT_3D | HELMERT_2D | TRANSLATION_ONLY
    parameters: HelmertParameters,
    residuals: tuple[Residual, ...],
    rms: float | None,                  # None ONLY for a single-point TRANSLATION_ONLY (no redundancy to compute it from)
    control_count: int,
    warning: str | None,                # None ONLY for a fully-determined HELMERT_3D
)
```

Verified directly, for a 4-control-point `HELMERT_3D` fit with small
(sub-20cm) real-world survey noise:

```python
>>> result.strategy
GeoreferencingStrategy.HELMERT_3D
>>> result.rms
0.1471617882998825
>>> result.warning
None
>>> result.residuals[0]
Residual(control_id='P1', dx=..., dy=..., dz=..., magnitude=0.1117...)
```

And for an intentionally under-determined 2-point fit (authorized):

```python
>>> result.strategy
GeoreferencingStrategy.HELMERT_2D
>>> result.warning
'Only 2 control points available: 2D Helmert. tx/ty/tz/rz/scale were estimated fr...'
```

`Residual.magnitude` is the Euclidean norm of `(dx, dy, dz)` for that
control point — how far the fitted transformation's own prediction
for that point misses its declared target. `rms` is the overall root-
mean-square across every residual; it is `None` specifically (and
only) when a single control point leaves no redundancy to compute an
RMS from at all.

## Applying a fitted result to real data

`fit_georeferencing()` only estimates the transformation — it never
touches your point cloud, survey, or features. Applying it is a
separate step, via one of 3 adapters:

```python
from topocore.geodesy.georeferencing.apply import (
    apply_georeferencing_to_point_cloud,
    apply_georeferencing_to_survey,
    apply_georeferencing_to_feature_collection,
)

new_cloud = apply_georeferencing_to_point_cloud(cloud, result.parameters, target_crs)
new_survey = apply_georeferencing_to_survey(survey, result.parameters, target_crs)
new_features = apply_georeferencing_to_feature_collection(collection, result.parameters, target_crs)
```

Each is pure (the input is never mutated) and vectorized internally
for `PointCloud`/`FeatureCollection` (confirmed directly to complete
in ~1.3 seconds for 10 million points and 50 control points, not a
per-point Python loop).

### `apply_scale_to_z` — never inferred from the parameter values

All 3 adapters accept `apply_scale_to_z: bool = True`. This exists
because `HELMERT_2D`'s own model assumes `Z' = Z + tz` (no scale term
on Z at all), while the shared application formula, by default,
would apply the fitted scale to Z too — a real, confirmed
discrepancy that grows with a point's own distance from the control
points' Z range (verified directly: with `Z=500` and a modest 50 ppm
scale, applying it incorrectly introduces a 2.5 cm error in Z alone).

The rule is exact and never varies: `apply_scale_to_z` is `True` if
and only if `result.strategy is GeoreferencingStrategy.HELMERT_3D`.
It is never inferred from `parameters.rx`/`parameters.ry` — those can
coincidentally be `0` in a genuine 3D fit too, so checking them
directly would be unreliable. `Workflow.georeference()` (below)
applies this exact rule for you; if you call the 3 adapters directly
yourself, you must apply it too.

## `Workflow.georeference()`

```python
workflow.georeference(artifact_type, controls, target_crs, options)
```

Combines `fit_georeferencing()` and the appropriate adapter into one
chainable stage, applying the `apply_scale_to_z` rule above
automatically. Supports the same 4 artifact types as
`transform_crs()`/`transform_vertical()`
(`POINT_CLOUD`, `GROUND_CLOUD`, `SURVEY_POINT_SET`,
`FEATURE_COLLECTION`; `GROUND_CLOUD` reuses the `PointCloud` adapter,
since both are the same underlying type).

```python
from topocore.geodesy import CRS
from topocore.geodesy.georeferencing import ControlPoint, GeoreferencingOptions
from topocore.workflow.artifacts import ArtifactType

controls = [
    ControlPoint(id="P1", source=(0.0, 0.0, 0.0), target=(1000.0, 2000.0, 100.0)),
    ControlPoint(id="P2", source=(100.0, 0.0, 0.0), target=(1099.8, 2000.3, 100.1)),
    ControlPoint(id="P3", source=(0.0, 100.0, 0.0), target=(999.7, 2100.2, 99.9)),
    ControlPoint(id="P4", source=(50.0, 50.0, 20.0), target=(1049.9, 2050.1, 119.8)),
]

workflow.georeference(
    ArtifactType.SURVEY_POINT_SET, controls, CRS.from_epsg(3116), GeoreferencingOptions()
)
```

### Requires the artifact's current CRS to genuinely be `None`

Unlike `transform_crs()` (which requires the current CRS to already
match the transformer's own source CRS), `georeference()` requires
the **opposite**: the artifact's current CRS must be `None`.
Control-point georeferencing exists specifically for local,
unreferenced coordinates — `crs=None` is the expected, common
starting point, never rejected for that reason alone.

What **is** rejected: calling `georeference()` a second time on an
artifact that already has a real CRS (from an earlier
`georeference()` call, `transform_crs()`, or a detected/declared
CRS):

```python
>>> workflow.georeference(ArtifactType.SURVEY_POINT_SET, controls, CRS.from_epsg(9377), GeoreferencingOptions())
WorkflowStateError: georeference() refused: survey_point_set's current
CRS is already known (CRS(EPSG:3116, ...)), not None. Control-point
georeferencing is for genuinely local/unreferenced coordinates --
running it again on an artifact that already has a real CRS would
silently re-apply a new transformation on top of an already-transformed
result. If you need a further CRS-to-CRS transformation, use
transform_crs() instead.
```

Without this check, a second `georeference()` call would silently
re-apply a *new* set of control points — computed for the *original*
local coordinates — on top of the *already-transformed* result,
producing a meaningless, double-shifted artifact with no error at
all. The rejected call leaves the artifact and the `Workflow`'s own
stage history completely untouched.

### The full `GeoreferencingResult` is preserved, not summarized away

```python
metrics = workflow.result.stages[-1].metrics
result = metrics.extra["georeferencing_result"]
```

`StageMetrics.extra` (the same general-purpose mechanism already
used elsewhere in `Workflow`, e.g. by `classify_ground()`) carries
the complete `GeoreferencingResult` — `strategy`, `parameters`,
`residuals`, `rms`, `control_count`, `warning` all remain available
after the stage runs, not flattened into a plain count.

### CRS propagation

The `target_crs` you pass becomes the resulting artifact's own CRS,
in each artifact's own established type convention:
`SurveyPointSet.crs` becomes the exact `CRS` object you passed;
`PointCloud.crs`/`FeatureCollection.crs` become its `"EPSG:{code}"`
(or bare name) string form — the same convention `transform_crs()`
already uses.

## What georeferencing does not do

- It does not detect a CRS from a file — that is a separate,
  earlier-stage concern (see [`../13-io/crs-detection.md`](../13-io/crs-detection.md)).
- It does not apply a known CRS-to-CRS transformation — that is
  `CoordinateTransformer`/`transform_crs()` (see
  [`transformer.md`](./transformer.md),
  [`workflow-integration.md`](./workflow-integration.md)).
- It never re-computes or improves an existing georeferencing —
  running it twice on the same artifact is rejected, not silently
  layered.
