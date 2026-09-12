# `topocore.features` — Overview

## Scope, clarified before anything else

**This document covers `topocore.features`** -- the classification
and construction of civil-engineering/CAD-style feature entities
(buildings, drainage structures, vegetation, infrastructure) from an
already-processed, already-classified point cloud or an already-built
`TIN`. **It does not cover `topocore.processing.features`**
(`PCAFeatures`/`PCAFeatureComputer`, geometric point-level feature
computation -- PCA, normals, curvature, eigenvalues), which is part
of `topocore.processing` and was already audited and closed under
`15-processing`. The 2 packages share a name fragment but are
otherwise unrelated: `topocore.features` *consumes* `processing`'s
own outputs (classification labels, normal vectors, PCA results, via
`DetectionContext`) -- it never recomputes them.

## Real inventory — 62 files, larger than any single block audited so far

```text
topocore/features/
├── (root) base.py, detector.py, feature_builder.py, feature_codes.py,
│          manager.py, models.py, protocols.py, exceptions.py,
│          _shared.py, _code_utils.py
├── catalogs/       9 domain catalogs + loaders/ (CSV/JSON/YAML) + audit tooling
├── grammar/         opt-in field-code grammar (confirmed never the default)
├── side/             left/right laterality resolution relative to a centerline
├── buildings/         walls, buildings, retaining_walls, roofs
├── drainage/           channels, drainage, inspection_chambers, manholes
├── infrastructure/      curbs, driveways, parking, roads
├── terrain/               contours, embankments, slope_changes, breaklines
│                          (a submodule of THIS package -- not `topocore.terrain`,
│                          despite the shared name; consumes an already-built TIN)
├── utilities/               light_poles, poles, signs
└── vegetation/                grass, shrubs, trees
```

22 detector classes register themselves at import time
(`DetectorRegistry`) -- confirmed, by direct execution, all 22
register with zero name collisions, both before and after every fix
in this audit.

## Architecture — 2 base classes, 3 inheritance shapes, confirmed consistent

`BaseFeatureDetector` (the true root) declares `required_inputs` (a
`ContextField` set) and a `detect()` template method that checks
those inputs are present in the `DetectionContext` *before* calling
the subclass's own `_detect()` -- confirmed the actual mechanism
behind every "missing input" `DetectionError` in this package, not a
separate check duplicated per detector.

`ClusterDetectorBase` (itself a `BaseFeatureDetector`) implements the
shared point-cloud pipeline most detectors build on: extract XYZ,
filter by classification code, cluster in 2D, filter each cluster
(size/elongation/extent), build geometry, emit one `Feature` per
accepted cluster.

Confirmed by direct `issubclass()` inspection of the real registry,
not assumed from file layout:

| Shape | Count | Detectors |
|---|---|---|
| `ClusterDetectorBase` (direct or via a concrete parent) | 13 | `BuildingDetector`, `DrivewayDetector`, `GrassDetector`, `ManholeDetector`→`InspectionChamberDetector`, `ParkingDetector`, `PoleDetector`→`LightPoleDetector`/`SignDetector`, `RoadDetector`, `ShrubDetector`, `TreeDetector`, `WallDetector` |
| `BaseFeatureDetector` directly, but still calls `_shared.py`'s clustering/hull functions | 2 | `RetainingWallDetector`, `RoofDetector` |
| `BaseFeatureDetector` directly, TIN-based, no `_shared.py` clustering | 7 | `BreaklineDetector`→`CurbDetector`, `ContourDetector`, `DrainageDetector`→`ChannelDetector`, `EmbankmentDetector`, `SlopeChangeDetector` |

The middle row matters specifically because it is NOT visible from
inheritance alone -- confirmed only by reading each file's own
imports. The 5 "detector inherits a concrete sibling detector"
cases (`InspectionChamberDetector(ManholeDetector)`, etc.) were each
confirmed to correctly override both `feature_type` and `name()`,
avoiding any registry collision or silent type confusion.

**Every one of the 22 detector constructors is callable with zero
arguments** -- confirmed a real, load-bearing contract, not an
accident: `DetectorRegistry.register()` itself calls
`detector_cls()` to read `.name()` at registration time, so any
detector with a required constructor parameter would break import
of the whole package. All 22 give every parameter a default.

## `models.py` — the shared data model, audited fully, no defects found

`Feature`/`FeatureGeometry`/`FeatureCollection` (frozen dataclasses).
Confirmed directly: all 85 `FeatureType` members have a matching
entry in the internal `_EXPECTED_GEOMETRY` table (no silent
validation skip for any type); `confidence` is validated to `[0,1]`;
vertex coordinates are validated finite at construction time;
`attributes`/`extra` are genuinely immutable (`MappingProxyType`, not
merely convention). `FeatureGeometry` has no `area`/`length`
property -- only `bounds` -- a fact that directly shaped one of this
audit's own fixes (see below).

## 3 real defects found and fixed — full account in `validation.md`

All 3 share the same underlying pattern: an external, non-`features`
exception (from `scipy` or `numpy`) escaping this package's own
domain-exception boundary, for a condition confirmed reachable in
the real production pipeline (`Workflow.detect_features()`, which
passes its own `PointCloud` with no finite-coordinate or
non-emptiness validation anywhere upstream):

1. **Non-finite (NaN/Inf) X/Y coordinates** reaching
   `scipy.spatial.cKDTree`/`ConvexHull` directly inside
   `_shared.py`, raising a raw `ValueError`. Affects 15 of 22
   detectors.
2. **3-or-more distinct but collinear points** reaching
   `scipy.spatial.ConvexHull`, raising a raw `QhullError` -- the
   existing `< 3` distinct points fallback didn't cover this
   genuinely different, also-real case (confirmed reachable from
   `BuildingDetector` with a realistic straight line of points, e.g.
   a fence or narrow structure).
3. **A genuinely empty `PointCloud`** reaching `np.concatenate()`
   inside `extract_xyz()`, raising a raw `ValueError`, instead of
   propagating to the already-correct "no candidates -> empty
   result" path every one of these detectors already has.

All 3 now raise `DetectionError` -- confirmed the correct exception
by direct precedent already existing elsewhere in this same package
(`DrainageDetector` already validates TIN finiteness the same way;
`ContourDetector` already wraps an underlying failure with `from
exc`), not invented for this fix. `GeometryError` was confirmed NOT
the right choice: it is raised exclusively by `models.py` itself, to
validate an already-constructed output geometry, never by a
detector's own input-side validation.

## Investigated and confirmed NOT defects

- **`cluster_points_2d()` vs. `processing.segmentation
  .ConnectedComponentsSegmenter`**: structurally similar (both do
  radius-based connected-components clustering) but confirmed a
  legitimate separation of responsibility, not duplication -- 2D vs.
  3D by deliberate domain need, different input/output contracts,
  and zero overlapping consumers confirmed by search.
- **`grammar/`'s own opt-in status**: confirmed directly --
  `feature_builder.py`'s own `use_field_code_grammar` defaults to
  `False`, and the grammar path is only exercised when a caller sets
  it explicitly. No contradiction with the package's own stated
  intent.
- **4 `FeatureType` members with zero catalog code**
  (`road`/`parking`/`driveway`/`sign`, alongside the expected
  automatic-only `breakline`/`contour`/`slope_change`/`drainage`):
  confirmed deliberate by the project's own author -- catalog codes
  exist only for field-surveyed features (total station/GNSS, where
  a surveyor assigns a code per point); these 4 are detected only
  from point-cloud geometry/classification, never from a field code.
- **The recovery contract** (`FeatureExtractionManager.detect_all()`):
  confirmed deliberate and controllable, not ambiguous -- a `strict`
  flag (`True` by default in `Workflow.detect_features()`) decides
  whether one detector's own `DetectionError` aborts the whole
  extraction run or is isolated (logged, skipped) so the remaining
  detectors still run.

## One confirmed, unaddressed cosmetic finding

`FeatureExtractionManager.detect_all()`'s own non-strict skip path
logs `"Skipping detector '%s': required inputs not available."` for
*every* `DetectionError`, including the 3 new causes this audit's
own fixes introduced (non-finite coordinates, empty cloud) -- neither
of which is actually a "missing input" in the sense the message
describes. Not corrected during this audit: it does not affect
behavior, and does not rise to the same standard of evidence
(reachable in real production, confirmed real cost) the 3 fixed
defects met. Recorded in [`limitations.md`](./limitations.md) for a
future, separate decision.

## Where to go next

- [`validation.md`](./validation.md) -- the 3 fixed defects, in full,
  with the same evidence chain used throughout this project
  (finding -> evidence -> classification -> fix -> behavior test ->
  regression).
- [`limitations.md`](./limitations.md) -- the cosmetic log-message
  finding, and anything else confirmed out of this audit's own scope.
