# Features

## The data model

```python
from topocore.features.models import (
    Feature, FeatureGeometry, FeatureCollection, FeatureMetadata,
    FeatureType, FeatureCategory, GeometryType,
)
```

```python
FeatureGeometry(
    geometry_type: GeometryType,   # POINT | POLYLINE | POLYGON | MESH
    vertices: NDArray[np.float64],  # shape (n, 3), always finite (enforced)
    closed: bool = False,
    faces: NDArray[np.int64] | None = None,  # required for MESH
)

Feature(
    feature_id: int | None,
    category: FeatureCategory,
    feature_type: FeatureType,
    geometry: FeatureGeometry,
    confidence: float = 1.0,        # must be in [0, 1] (enforced)
    metadata: FeatureMetadata | None = None,
    attributes: Mapping[str, Any] = {},   # immutable once constructed
    source_point_indices: IntArray1D | None = None,
)

FeatureMetadata(
    detector: str,
    version: str = "1.0",
    inputs_used: frozenset[ContextField] = frozenset(),
    extra: Mapping[str, Any] = {},   # immutable once constructed
)
```

Confirmed directly, both construction-time invariants: `vertices`
must be finite and shaped `(n, 3)`; a `Feature`'s own `geometry_type`
must be one this `feature_type` allows (checked against an internal
`_EXPECTED_GEOMETRY` table covering all 85 `FeatureType` members).
`attributes`/`extra` are genuinely immutable (`MappingProxyType`, not
merely a convention) -- assigning into them raises `TypeError`.

```python
collection = FeatureCollection()
collection.crs           # str | None, e.g. "EPSG:4326" -- settable directly
collection.add(feature)
len(collection)
for feature in collection:
    ...
```

`FeatureCollection.crs` uses the same string convention as
`PointCloud.crs` (`"EPSG:{code}"` or a bare name), not
`SurveyPointSet.crs`'s real `CRS` object -- deliberately, since a
collection can originate from either source, and a `CRS` object
always converts safely to this string form, while the reverse is not
always possible.

## `FeatureType` vs. `FeatureCodeDefinition` — 2 different things

Confirmed directly during this project's own catalog audit: a
`Feature`'s `feature_type` never requires a matching
`FeatureCodeDefinition` to exist. These are 2 independent concepts:

```text
FeatureType
    the semantic type a Feature carries -- can come from either an
    automatic detector (no catalog code involved at all) or a
    field-surveyed code (via FeatureCodeDefinition.feature_type)

FeatureCodeDefinition
    a survey field-code's own definition (code, name, feature_type,
    category, geometry_type, layer) -- exists only for
    total-station/GNSS field codes; a code without a matching
    definition simply fails to match during FeatureBuilder.build(),
    routed to "unmatched" -- never a crash
```

8 of the 85 `FeatureType` members are confirmed to have zero catalog
code at all (`breakline`, `contour`, `slope_change`, `drainage`,
`road`, `parking`, `driveway`, `sign`) -- these are detected only
from point-cloud geometry/classification or TIN analysis, never from
a field code, per this project's own author.

## Producing a `FeatureCollection`

**From a point cloud** (automatic detection):

```python
from topocore.features.manager import FeatureExtractionManager
from topocore.features.protocols import DetectionContext

context = DetectionContext(
    cloud=cloud,                  # required
    tin=tin,                       # optional
    dtm=dtm,                        # optional
    classification=classification,   # optional
)
result = FeatureExtractionManager(strict=True).detect_all(context)
```

`strict=True` aborts the whole run on the first detector failure;
`strict=False` isolates that one detector (recorded in the report)
and continues with the rest.

**From a field survey** (feature-code interpretation):

```python
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry

registry = FeatureCodeRegistry.default()
build_result = FeatureBuilder(registry).build(survey_points)
features = build_result.features
```

Both are also reachable through `Workflow` -- see
[`workflows.md`](./workflows.md)'s `detect_features()`/
`build_features_from_survey()`.

## Detector families

22 built-in detectors register themselves automatically on import,
grouped by which base class they build on:

```text
ClusterDetectorBase (point-cloud clustering pipeline)
  buildings, walls, roads, driveways, parking, poles, grass, trees,
  shrubs, manholes  (+ 3 more via a concrete-detector subclass:
  inspection_chambers, light_poles, signs)

BaseFeatureDetector directly, TIN-based
  breaklines, contours, embankments, slope_changes, drainage
  (+ curbs, channels via a concrete-detector subclass)

BaseFeatureDetector directly, but still uses the clustering pipeline
  retaining_walls, roofs
```

Every detector constructor is callable with zero arguments
(a load-bearing contract: the registry itself calls each detector
class with no arguments to read its own name at registration time).
All configuration parameters have real defaults.

## `DetectionContext` — what a detector can actually receive

```python
from topocore.features.protocols import DetectionContext

DetectionContext(
    cloud: PointCloud,
    tin: TIN | None = None,
    dtm: DTM | None = None,
    classification: ClassificationResult | None = None,
    normals: ... | None = None,
    pca_features: PCAFeatureResult | None = None,
)
```

Confirmed directly: there is no `ground_cloud` field at all -- no
detector reachable through this pathway can ever consume
`GROUND_CLOUD` data specifically (a real, historical false-positive
staleness bug in `Workflow.detect_features()` was fixed by no longer
tracking it here).

## `FeatureBuilder`'s 2 grouping modes — legacy vs. field-code grammar

```python
FeatureBuilder(registry, *, closure_tolerance: float = 0.0, use_field_code_grammar: bool = False)
```

**`use_field_code_grammar=False` (default)**: groups only
*consecutive* survey points sharing the same field code into one
linear/polygon run. Confirmed by real execution: a survey walked
cross-section by cross-section (edge, centerline, edge, then move to
the next station and repeat) produces **0 features** -- every code
changes every point, so every "run" is exactly 1 point long, and
every point is diagnosed `INSUFFICIENT_POINTS`. This mode only works
when each linear feature is walked as one uninterrupted sequence.

**`use_field_code_grammar=True`**: opts into
`topocore.features.grammar`'s real `BASE.FIGURE[.S|E|X]` code syntax
(a standard total-station convention -- `S`=start, no suffix=continue,
`E`=end, `X`=close), letting figures be assembled correctly even when
interleaved with other figures in the file. Confirmed by real
execution: the exact cross-section survey that produced 0 features
above produces the correct 3 features (2 `PAVEMENT_EDGE` + 1
`CENTERLINE`, 5 vertices each) once the codes use this syntax
(`"EJE.1.S"`, `"EJE.1"`, ..., `"EJE.1.E"`). Codes without the
separator still use the same legacy consecutive-run rule even in this
mode -- confirmed you can mix grammar-coded and plain-coded figures
in the same file.

**Confirmed real gap**: `Workflow.build_features_from_survey()` has
no way to reach this parameter -- its own real signature is
`build_features_from_survey(self, registry: Any = None)`. A
cross-section survey can only be processed via `FeatureBuilder`
directly; `Workflow` cannot reach grammar mode for this stage at all.

**No integration path back into `Workflow` exists today.** Confirmed
directly: `Workflow` exposes exactly 16 stage methods, none of which
accepts an already-built `FeatureCollection` from outside. Writing
directly into `Workflow`'s own private artifact store
(`wf._store.set(ArtifactType.FEATURE_COLLECTION, collection)`) is
technically possible but confirmed by real execution to leave
`Workflow`'s own history broken -- `wf.result.stages` never records a
`BUILD_FEATURES_FROM_SURVEY` entry, silently violating `Workflow`'s
own "every artifact has an associated stage in history" guarantee.
This is not a supported workaround, and this guide does not recommend
it -- if any part of a survey might use interleaved field codes,
process that entire branch (`FeatureBuilder`, `SideResolver`,
`DXFExporter`/`GeoPackageExporter`) with the modular API, never mixed
with `Workflow`.

**Safe default recommendation, confirmed by real execution**:
`use_field_code_grammar=True` is a strict superset of the legacy
mode -- a survey using only plain, consecutive-run codes (no `.`
separator anywhere) produces the exact same result whether this flag
is `True` or `False`, confirmed directly against the same fixture
both ways. For any code path that doesn't know in advance which
convention a given uploaded survey file used (e.g. a web API or
service accepting field-survey uploads from unknown users), always
passing `use_field_code_grammar=True` to a direct `FeatureBuilder`
call is safe and covers both conventions -- there is no case where
setting it costs you correctness on a legacy-style file.

See [`end-to-end-examples.md`](./end-to-end-examples.md) for the
complete, real cross-section road survey this was verified against.

## `SideResolver` — laterality relative to a centerline

```python
from topocore.features.side.resolver import SideResolver
from topocore.features.side.models import Side, SideMethod

resolver = SideResolver(
    target_types: frozenset[FeatureType] = frozenset({FeatureType.PAVEMENT_EDGE}),
    max_distance: float = 30.0,        # a CENTERLINE farther than this is never a candidate
    ambiguity_margin: float = 0.10,    # 2 candidates within 10% of each other's distance -> UNKNOWN
    cross_tolerance: float = 1e-6,     # a point effectively ON the centerline -> UNKNOWN
)
new_collection = resolver.resolve(collection: FeatureCollection) -> FeatureCollection
```

Confirmed pure -- returns a new `FeatureCollection`, never mutates
the input (`Feature` is frozen). Every feature whose `feature_type`
isn't in `target_types` (only `PAVEMENT_EDGE` by default; pass a
wider set to also resolve e.g. `FeatureType.CURB`) is carried over
unchanged, in original order, with `feature_id`s preserved exactly.
`collection.crs` is preserved unchanged -- this only adds `side`/
`side_method` to `Feature.attributes`, it never moves geometry.

3 real `Side` values: `LEFT`, `RIGHT`, `UNKNOWN` (ambiguous or
on-the-line). `SideMethod` distinguishes *how* a `Side` was reached
-- confirmed real values include a survey-provided one (used as-is if
`attributes["side"]` was already set) and a computed
`"geometric"` one -- kept separate from `Side` itself specifically so
"not yet resolved" and "resolved, but ambiguous" both stay
distinguishable from a confident answer, for auditing.

See [`end-to-end-examples.md`](./end-to-end-examples.md) for a
complete, real road-survey example (1 centerline + 2 pavement edges)
run through `Workflow.resolve_sides()` end to end, including a real
field-survey pitfall this capability's own grouping rule creates if
you aren't careful with field codes.

## Where to go next

- [`workflows.md`](./workflows.md) -- `detect_features()`/
  `build_features_from_survey()`'s full `Workflow` contract.
- [`catalogs.md`](./catalogs.md) -- `FeatureCodeRegistry` in full,
  including external catalog loading.
- [`export.md`](./export.md) -- turning a `FeatureCollection` into a
  DXF or GeoPackage file.
