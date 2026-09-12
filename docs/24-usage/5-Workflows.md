# Workflows

`Workflow` is the high-level orchestrator: a fluent, chainable object
that manages artifacts, their dependencies, versions, staleness, an
append-only execution history, and progress reporting, on top of the
same modular classes documented in [`terrain.md`](./terrain.md),
[`features.md`](./features.md), and [`export.md`](./export.md).
Every signature below is confirmed against the real, current source.

## Creating a workflow

```python
from topocore.workflow import Workflow

workflow = Workflow(*, progress: ProgressObserver | None = None)
```

`progress`, if given, is any object with an `on_progress(event:
ProgressEvent) -> None` method (a structural `Protocol`, no base
class required). Confirmed directly: an exception raised inside your
own `on_progress()` is caught and logged, never allowed to break the
stage it was reporting on.

## The 8 artifact types

`Workflow` tracks exactly these 8 -- one current version of each,
never more:

```text
SURVEY_POINT_SET       from read_survey()
POINT_CLOUD            from read_point_cloud()
GROUND_CLOUD           from classify_ground()
CLASSIFICATION_RESULT  from classify_points()
TIN                    from build_tin()
DTM                    from build_dtm()
CONTOURS               from extract_contours()
FEATURE_COLLECTION     from detect_features() / build_features_from_survey()
                       (also re-produced in place by resolve_sides())
```

`transform_crs()`/`transform_vertical()`/`georeference()` re-produce
whichever of `POINT_CLOUD`/`GROUND_CLOUD`/`SURVEY_POINT_SET`/
`FEATURE_COLLECTION` you name -- they do not introduce a 9th type.

## The full stage graph

```text
READ_SURVEY  ----------------------------> SURVEY_POINT_SET
READ_POINT_CLOUD ----------------------->  POINT_CLOUD
                                                |
                            (mutually exclusive roots --
                             a Workflow reads exactly one)

POINT_CLOUD --CLASSIFY_GROUND-->      GROUND_CLOUD
POINT_CLOUD --CLASSIFY_POINTS-->      CLASSIFICATION_RESULT

GROUND_CLOUD --BUILD_TIN-->           TIN
TIN --BUILD_DTM-->                    DTM
TIN --EXTRACT_CONTOURS-->             CONTOURS   (never from DTM)

POINT_CLOUD [+ TIN, DTM, CLASSIFICATION_RESULT if present, optionally]
      --DETECT_FEATURES-->            FEATURE_COLLECTION

SURVEY_POINT_SET --BUILD_FEATURES_FROM_SURVEY--> FEATURE_COLLECTION

FEATURE_COLLECTION --RESOLVE_SIDES--> FEATURE_COLLECTION (re-produced)

{POINT_CLOUD, GROUND_CLOUD, SURVEY_POINT_SET, FEATURE_COLLECTION}
      --TRANSFORM_CRS-->        (same type, re-produced)
      --TRANSFORM_VERTICAL-->   (same type, re-produced)
      --GEOREFERENCE-->         (same type, re-produced)

FEATURE_COLLECTION --EXPORT_DXF-->    (external file, no artifact produced)
FEATURE_COLLECTION --EXPORT_GPKG-->   (external file, no artifact produced)
```

## Every stage method, in full

### Ingestion — mutually exclusive roots

```python
workflow.read_point_cloud(path: str | Path, **reader_kwargs) -> Workflow
```
Produces `POINT_CLOUD`. Format (LAS/LAZ/E57/PLY/XYZ/CSV/PTS) is
selected by file extension. `**reader_kwargs` pass straight through
to the selected reader class (see [`point-clouds.md`](./point-clouds.md)
for each reader's own real parameters -- notably `E57Reader`'s
`chunk_size` has no default and is filled in automatically here).

```python
workflow.read_survey(path: str | Path, **reader_kwargs) -> Workflow
```
Produces `SURVEY_POINT_SET`, via `SurveyTXTReader` (see
[`point-clouds.md`](./point-clouds.md)). Requires that no prior READ
stage has already run on this `Workflow` -- `read_point_cloud()` and
`read_survey()` cannot both run on the same instance.

### Classification

```python
workflow.classify_ground(**manager_kwargs) -> Workflow
```
Requires `POINT_CLOUD`. Produces `GROUND_CLOUD`, via
`GroundManager(**manager_kwargs).extract(cloud)`. The resulting
cloud's own `crs` is confirmed propagated from the source
`POINT_CLOUD` (a real, historical gap already fixed).

The underlying `GroundManager` (usable directly, bypassing
`Workflow`) accepts a real, verified set of parameters -- confirmed
from its own docstring's real example:

```python
from topocore.processing.ground import GroundManager

manager = GroundManager(method: str = "grid", cell_size: float = 1.0, height_threshold: float = 0.2, ...)
mask = manager.classify(cloud)     # -> boolean mask
ground_cloud = manager.extract(cloud)  # -> PointCloud, ground points only

manager.method = "progressive_tin"   # switch method on the same instance
```

```python
workflow.classify_points(*, method: str = "auto", **manager_kwargs) -> Workflow
```
Requires `POINT_CLOUD`. Produces `CLASSIFICATION_RESULT` -- a
genuinely different algorithm from `classify_ground()`, not a
variant of it (multi-class classification vs. binary ground
extraction).

The underlying `ClassificationManager` (usable directly):

```python
from topocore.processing.classification.manager import ClassificationManager

manager = ClassificationManager(
    method: str = "auto",   # "rule_based" | "auto" | "random_forest" |
                              # "gradient_boost" | "lightgbm" | "xgboost"
    model_params: dict | None = None,
    rule_params: dict | None = None,
)
result = manager.classify(cloud)   # -> ClassificationResult; raises ClassificationError for an empty cloud

# "auto" starts rule-based (no training needed); promote to the best
# available ML backend by training it:
manager.fit(cloud, labels)   # returns the same manager, fluent
manager.is_trained()          # bool
```

Confirmed directly: `fit()`'s own backend preference order, when
`method="auto"`, is XGBoost > LightGBM > Gradient Boost > Random
Forest, based on what is actually installed.

### Terrain

```python
workflow.build_tin() -> Workflow
```
Requires `GROUND_CLOUD`. Produces `TIN`.

```python
workflow.build_dtm(grid, *, method: InterpolationMethod, power: float = 2.0) -> Workflow
```
Requires `TIN`. Produces `DTM`. `method` has no default -- you must
choose one explicitly (see [`terrain.md`](./terrain.md) for the 4
real `InterpolationMethod` values). `power` only affects `IDW`.

```python
workflow.extract_contours(interval: float, *, base: float = 0.0) -> Workflow
```
Requires `TIN` -- confirmed directly, `ContourGenerator` has no
dependency on `DTM` at all, so building a `DTM` first is never a
prerequisite for contours. Produces `CONTOURS`.

### Features

```python
workflow.detect_features(*, strict: bool = True) -> Workflow
```
Requires `POINT_CLOUD`. Optionally uses `TIN`/`DTM`/
`CLASSIFICATION_RESULT` if already present in this same `Workflow`
(each checked for staleness individually before use). Produces
`FEATURE_COLLECTION`. `strict` here is `FeatureExtractionManager`'s
own real parameter -- `True` aborts the whole detection run on the
first detector failure; `False` isolates that one detector and
continues with the rest (see [`features.md`](./features.md)).

```python
workflow.build_features_from_survey(registry=None) -> Workflow
```
Requires `SURVEY_POINT_SET`. Produces `FEATURE_COLLECTION`, via
`FeatureBuilder` (field-code interpretation) -- a genuinely different
algorithm from `detect_features()`, not an alternate path to the same
one. `registry` defaults to `FeatureCodeRegistry.default()` if not
given.

```python
workflow.resolve_sides(**resolver_kwargs) -> Workflow
```
Requires `FEATURE_COLLECTION`. Re-produces `FEATURE_COLLECTION` (not
a new type) -- resolves left/right laterality for
`PAVEMENT_EDGE`-type features relative to a `CENTERLINE`. Optional:
must be explicitly chained; a collection with no relevant feature
types passes through unmodified, which is not an error.
`**resolver_kwargs` forward to `SideResolver.__init__`.

### CRS / vertical / georeferencing — all optional, all in place

```python
workflow.transform_crs(artifact_type: ArtifactType, transformer) -> Workflow
```
`artifact_type` must be one of `POINT_CLOUD`, `GROUND_CLOUD`,
`SURVEY_POINT_SET`, `FEATURE_COLLECTION` -- any other type (`TIN`,
`DTM`, `CONTOURS`, `CLASSIFICATION_RESULT`) raises `WorkflowStateError`
explicitly, never silently ignored. `transformer` is an
already-built `topocore.geodesy.CoordinateTransformer` -- this stage
never constructs one itself.

```python
workflow.transform_vertical(artifact_type, *, source_datum, target_datum, geoid) -> Workflow
```
Same 4 supported types as `transform_crs()`. Shifts `Z` via a
`VerticalTransformer` built internally from the 3 keyword arguments.
`X`/`Y` must already be geographic degrees in `geoid`'s own CRS --
this stage does not reproject. Confirmed: never leaves `Z`
uncorrected -- a point outside the geoid's own extent fails the
whole stage (`WorkflowExecutionError` wrapping `MissingGeoidGridError`)
rather than producing a partially-shifted result.

```python
workflow.georeference(artifact_type, controls, target_crs, options) -> Workflow
```
Same 4 supported types. `controls: Sequence[ControlPoint]`,
`target_crs: CRS`, `options: GeoreferencingOptions`. Confirmed
directly: requires the artifact's own current `crs` to genuinely be
`None` -- running this a second time on an already-referenced
artifact raises `WorkflowStateError` explicitly, rather than silently
double-transforming it.

### Export — lateral, never written to the artifact store

```python
workflow.export_dxf(path: str | Path, **exporter_kwargs) -> Workflow
```
Requires `FEATURE_COLLECTION`. `**exporter_kwargs` build a
`DXFExportOptions` (e.g. `strict=False`, `dxf_version=...`) --
confirmed fixed: these are no longer passed directly to
`DXFExporter.__init__()` (a real, historical bug). May be called
multiple times, including alongside `export_gpkg()`, on the same
collection.

```python
workflow.export_gpkg(path: str | Path, *, epsg: int | None = None, **exporter_kwargs) -> Workflow
```
Requires `FEATURE_COLLECTION`. `epsg` is optional: if the collection's
own `crs` already gives a real `"EPSG:..."` code, it is used
automatically; if both are given and disagree, `WorkflowStateError`
is raised rather than silently picking one.

## Workflow state — how the pieces fit together

```text
Workflow
 |-- ArtifactStore    current version of each of the 8 artifact types
 |                    (never a dependency graph -- only answers
 |                    "is X present?" / "what version is X at?")
 |-- WorkflowValidator  checks presence (require*) and, since PR19,
 |                      transitive freshness (require_current) --
 |                      never validates a stage's own parameters
 |-- history (list[StageResult])  append-only, never edited after
 |                                 a stage completes
 `-- ProgressObserver (optional)  receives "starting"/"finished"
                                   events; never receives one for a
                                   stage that failed before finishing
```

`workflow.result` returns a `WorkflowResult`:

```python
result.stages       # tuple[StageResult, ...], in execution order
result.started_at    # first stage's own start, UTC
result.finished_at    # last stage's own finish (or started_at, if none ran)
result.duration        # derived property, wall-clock seconds
result.last_stage        # most recent StageResult, or None
```

Each `StageResult` carries `stage`, `status` (`SUCCESS`/`FAILED`/
`SKIPPED`/`NOT_RUN`), `started_at`/`finished_at`, `dependencies`
(every artifact version actually consumed), `produced` (the artifact
written, or `None` for a failed or export stage), and `error` (the
original exception, only set on `FAILED`).

`workflow.artifact(artifact_type)` returns the current value, or
raises `StaleArtifactError` if it exists but was built from
since-superseded input -- confirmed via a real, executed 2-hop
reproduction: re-versioning `POINT_CLOUD` correctly marks `DTM` stale
even though `DTM`'s own immediate dependency (`TIN`) was never
itself rebuilt.

## Error handling — 2 genuinely different categories

Confirmed by direct execution, not merely by the exceptions' own
docstrings:

```text
Precondition failure (checked BEFORE the stage's own work runs)
    -> WorkflowStateError directly, never wrapped
    Examples: missing artifact, unsupported artifact_type for
    transform_crs()/transform_vertical()/georeference(),
    already-referenced artifact given to georeference(), an EPSG
    conflict in export_gpkg()

Failure INSIDE the stage's own work
    -> WorkflowExecutionError, always with the original exception
       attached via .__cause__
    A FAILED StageResult is appended to history before the exception
    is raised. Confirmed directly: a prior, already-successful
    artifact in the same Workflow is never affected -- it keeps its
    own version, remains fully usable, and a subsequent independent
    stage (or a retry of the same failed one, with corrected
    parameters) both succeed normally afterward.
```

`WorkflowValidationError` exists for a stage's own invalid
*parameters* (distinct from a missing artifact) -- confirmed by its
own docstring, though no stage method in the current codebase
actually raises it (parameter validation currently happens inside
the delegated module itself, surfacing as `WorkflowExecutionError`
instead).

## Where `strict` actually exists — do not assume it is universal

Confirmed directly, method by method: only 3 of the 16 stage methods
accept `strict` at all, and each is a distinct downstream option, not
a `Workflow`-wide setting:

```text
detect_features(strict: bool = True)          -- FeatureExtractionManager's own
export_dxf(**exporter_kwargs -> strict=...)    -- DXFExportOptions's own
export_gpkg(**exporter_kwargs -> strict=...)   -- GPKGExportOptions's own
```

The other 13 stage methods have no `strict` parameter at all -- a
failure inside any of them always raises `WorkflowExecutionError`
unconditionally; there is no partial-isolation mode for
`classify_ground()`, `build_tin()`, `build_dtm()`, etc.

## Examples

### Minimal workflow

```python
from topocore.workflow import Workflow

workflow = Workflow()
workflow.read_point_cloud("survey.las")
workflow.classify_ground()
workflow.build_tin()
```

### Terrain workflow

```python
from topocore.terrain.grid import Grid
from topocore.terrain.enums import InterpolationMethod

workflow = Workflow()
workflow.read_point_cloud("survey.las")
workflow.classify_ground()
workflow.build_tin()

grid = Grid(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0, resolution=1.0)
workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)
workflow.extract_contours(interval=1.0, base=0.0)
```

### Feature workflow (point cloud)

```python
workflow = Workflow()
workflow.read_point_cloud("survey.las")
workflow.classify_ground()
workflow.build_tin()
workflow.detect_features(strict=False)  # isolate bad detectors
```

### Feature workflow (surveyed field codes)

```python
workflow = Workflow()
workflow.read_survey("survey.txt")
workflow.build_features_from_survey()
workflow.resolve_sides()  # optional, only affects PAVEMENT_EDGE/CENTERLINE
```

### Export workflow

```python
workflow.export_dxf("output.dxf", strict=False)
workflow.export_gpkg("output.gpkg", epsg=32618, strict=False)
```

### Workflow with a CRS transform

```python
from topocore.geodesy.crs import CRS
from topocore.geodesy.transformer import CoordinateTransformer
from topocore.workflow.artifacts import ArtifactType

workflow = Workflow()
workflow.read_point_cloud("survey.las")
workflow.artifact(ArtifactType.POINT_CLOUD).crs = "EPSG:4326"

transformer = CoordinateTransformer(CRS.from_epsg(4326), CRS.from_epsg(32618))
workflow.transform_crs(ArtifactType.POINT_CLOUD, transformer)
```

### Workflow with recovery after a failure

```python
from topocore.workflow.exceptions import WorkflowExecutionError

workflow = Workflow()
workflow.read_point_cloud("survey.las")
workflow.classify_ground()
workflow.build_tin()

try:
    workflow.extract_contours(interval=-1.0)  # invalid -- raises
except WorkflowExecutionError as exc:
    print("stage failed:", exc.__cause__)

# The Workflow is still fully usable -- TIN is untouched, and either
# an independent stage or a corrected retry of the same one succeeds:
grid = Grid(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0, resolution=1.0)
workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)
workflow.extract_contours(interval=1.0)  # retried with a valid value
```

## Where to go next

- [`point-clouds.md`](./point-clouds.md) -- every reader
  `read_point_cloud()`/`read_survey()` can dispatch to.
- [`features.md`](./features.md) -- `detect_features()`/
  `build_features_from_survey()`'s own underlying model and detectors.
- [`terrain.md`](./terrain.md) -- `TIN`/`DTM`/contours in full.
- [`export.md`](./export.md) -- the complete `strict`/error contract
  for `DXFExporter`/`GeoPackageExporter`.
