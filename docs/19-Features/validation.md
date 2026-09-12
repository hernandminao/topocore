# Validation — 3 defects found and fixed

## Methodology, identical to `16-analysis`/`16-terrain`

```text
finding -> evidence -> classification -> fix -> behavior test -> regression
```

All 3 defects below share the same underlying pattern (an external,
non-`features` exception escaping this package's own domain boundary)
and the same root cause (no upstream guarantee, anywhere in the real
pipeline, that a `PointCloud` has finite coordinates or is non-empty)
-- confirmed directly, not assumed, at every step: `PointCloud`/
`Chunk` validates neither; `Workflow.detect_features()` passes its
own `cloud` straight from the artifact store with no intermediate
check; `processing` (a different package) already validates
`isfinite` explicitly in 10 of its own modules, confirming the
project's own real convention is that each specific algorithm
validates its own numeric preconditions, not the raw data container.

---

## 1. Non-finite (NaN/Inf) X/Y coordinates reaching `scipy` directly

**Finding**: `cluster_points_2d()`/`convex_hull_polygon()`
(`topocore.features._shared`) passed their own `xy`/`points_xyz`
arguments straight to `scipy.spatial.cKDTree`/`ConvexHull`, with no
validation.

**Evidence**: confirmed directly, a `NaN` in a candidate point's own
X coordinate raised `ValueError: data must be finite, check for nan
or inf values` from `cKDTree`, and separately `ValueError: Points
cannot contain NaN` from `ConvexHull` -- both raw, un-wrapped scipy
exceptions, confirmed to propagate all the way through a real,
registered detector (`TreeDetector().detect()`) with no `features`
exception anywhere in between. Confirmed reachable in the real
production pipeline specifically, not merely in a synthetic unit
test: `Workflow.detect_features()`'s own source shows `cloud`
retrieved directly from the artifact store and passed to
`FeatureExtractionManager(strict=strict).detect_all(context)` with
no intermediate validation of any kind.

**Classification**: a real, confirmed defect. `DetectionError` was
confirmed the correct exception by direct precedent already existing
in this same package: `topocore.features.drainage.DrainageDetector`
already raises `DetectionError("TIN vertices must contain only
finite coordinates.")` for the identical condition on its own TIN
input. `GeometryError` was confirmed NOT appropriate -- checked
directly, it is raised exclusively by `models.py` itself, to
validate an already-constructed output geometry, never by a
detector's own input-side validation.

**Fix**: both functions now check `np.all(np.isfinite(...))`
explicitly before their own `scipy` call, raising
`DetectionError("Point coordinates must contain only finite
coordinates.")` -- matching the message style of the existing
`DrainageDetector` precedent, not inventing a new convention.

**Behavior test**: confirmed directly -- the exact real-pipeline
reproduction (`TreeDetector`, a point cloud with one `NaN` X
coordinate among otherwise-valid vegetation candidates) now raises
`DetectionError` instead of a raw `ValueError`; every previously-valid
input (confirmed with a real, hand-checked cluster and a real,
hand-checked convex hull) is confirmed byte-for-byte unaffected.

**Regression**: `ruff`/`mypy` (real `pyproject.toml`) clean on
`_shared.py`; 22/22 detectors still register with no collision; full
available test suite passes.

---

## 2. 3-or-more distinct but collinear points reaching `ConvexHull`

**Finding**: `convex_hull_polygon()`'s own existing fallback covered
fewer than 3 distinct XY positions (a tiny, fixed-offset triangle),
but not 3-or-more distinct positions that are all collinear -- a
geometrically different, separately degenerate case `ConvexHull`
also cannot handle.

**Evidence**: confirmed directly, 5 exactly-collinear points raised
`scipy.spatial._qhull.QhullError` ("Initial simplex is flat").
Confirmed reachable from a real, registered detector, not only from
calling the private function directly: `BuildingDetector` (which
uses `geometry_type = GeometryType.POLYGON` and, confirmed
separately, does not configure an elongation filter that would have
rejected this cluster earlier) raised the same raw `QhullError` given
a realistic straight line of building-classified points -- a
plausible real case (a narrow structure, or a sparse/regular
sampling artifact), not a contrived one.

**Classification**: a real, confirmed defect, distinct from defect 1
above (finite but geometrically degenerate input, not non-finite
input). Reusing the existing `< 3` distinct points fallback
unmodified was investigated and confirmed NOT appropriate: that
fallback's own tiny, fixed `1e-3` offset is sized for a handful of
near-coincident points with no real spatial extent to preserve; a
genuinely long collinear cluster (the reproduction case spans 9
distinct meters) has real, meaningful extent that `FeatureGeometry`'s
own `bounds` (its only derived property -- confirmed directly, it has
no `area`/`length`) would otherwise silently discard.

**Fix**: collinearity is detected via `_is_collinear()`, using the
same criterion already established elsewhere in this project
(`terrain.algorithms.delaunay.DelaunayTriangulator
._validate_collinear_points()`: the rank of the mean-centered
coordinates, via SVD) -- reused for consistency, not reinvented.
When detected, `_collinear_sliver()` builds a thin, non-degenerate
rectangle spanning the cluster's own real length and orientation
(its 2 extreme points along the principal direction, each offset by
a small perpendicular width), instead of `ConvexHull`. As a final
safeguard, any residual `QhullError` `ConvexHull` might still raise
for an unanticipated reason is now also caught and wrapped into
`DetectionError`, rather than left to escape.

**Behavior test**: confirmed directly -- the 5-point collinear case no
longer raises, and its own resulting geometry's X-extent is exactly
`4.0` (the real span), not collapsed to the `<3`-fallback's own tiny
`1e-3` scale. The exact `BuildingDetector` reproduction now succeeds,
producing 1 feature whose own `bounds` confirms the real 9-meter
extent. Every previously-valid case (a normal 4-point square, and the
existing `< 3` distinct points fallback itself) confirmed unaffected.

**Regression**: `ruff`/`mypy` clean; 22/22 detectors register; full
available test suite passes.

---

## 3. A genuinely empty `PointCloud` reaching `np.concatenate()`

**Finding**: `extract_xyz()` called `np.concatenate()` on each
coordinate's own list of chunk arrays unconditionally, with no check
for a `PointCloud` containing zero chunks.

**Evidence**: confirmed directly, a `PointCloud()` with no chunks
added raised `ValueError: need at least one array to concatenate` --
reachable from every one of the 15 detectors that call `extract_xyz()`
(all of Family A plus `RetainingWallDetector`/`RoofDetector`),
including via a real, registered detector
(`TreeDetector().detect()`) given an empty cloud directly.

**Classification**: a real, confirmed defect, but of a different
shape from defects 1 and 2 -- a genuinely empty `PointCloud` is not
*invalid* input in the way non-finite or collinear coordinates are;
it is a normal, valid state (e.g. after an earlier pipeline stage
filtered every point out of a given tile). Confirmed directly: every
detector already treats "zero classification-matched candidates" as
a valid, non-error outcome (`candidate_idx.size == 0 -> return
FeatureCollection()`), so the correct fix was to let an empty cloud
converge to that SAME existing path, not to raise a new exception
for it.

**Fix**: `extract_xyz()` now checks `cloud.is_empty` (an existing,
real `PointCloud` property, confirmed already used elsewhere in this
project's own `conversion.py`) and returns an empty `(0, 3)` array
directly, before attempting any concatenation.

**Behavior test**: confirmed directly -- `extract_xyz(PointCloud())`
now returns shape `(0, 3)` instead of raising; the exact real-pipeline
reproduction (`TreeDetector` given a fully empty cloud) now returns
an empty `FeatureCollection` (0 features), matching this package's
own already-established convention for "no candidates found," not a
newly-invented one. A real, non-empty cloud confirmed to produce a
byte-for-byte identical result to before the fix.

**Regression**: `ruff`/`mypy` clean; 22/22 detectors register; full
available test suite passes.

---

## Consolidated regression, all 3 fixes together

- `ruff`, with the real project's own `pyproject.toml`
  (auto-discovered from the package root): clean on `_shared.py`,
  the only file modified across all 3 fixes.
- `mypy`, same real configuration, `_shared.py` checked in isolation:
  clean.
- **Every line of the final diff against the pre-fix source was
  reviewed directly** and confirmed to belong to one of these 3
  fixes (plus 1 pre-existing-pattern `# noqa: RUF022` needed only
  because editing the file caused `ruff` to re-evaluate its own
  `__all__` block) -- no incidental or unrelated change.
- 11 new regression tests (4 for defect 1, 5 for defect 2, 2 for
  defect 3) -- all pass.
- The full available test suite: 315 of 315 pass.
- `DetectorRegistry`: 22 of 22 detectors still register, with zero
  name collisions, confirmed both before and after all 3 fixes.

## What this audit's own 3 fixes do not cover

No test file for `topocore.features` existed anywhere available to
this audit -- confirmed directly, not assumed: neither the
reconstructed repository export nor any upload provided one. The 11
regression tests added here are, honestly, the first tests this
package has had in this audit's own working environment; they were
written specifically to prove the 3 fixed defects stay fixed, and
should not be read as comprehensive coverage of `topocore.features`
as a whole -- the extensive manual, hand-verified executions
documented throughout [`overview.md`](./overview.md) (85/85
`FeatureType` coverage, 22/22 registry, the family-by-family error-path
audit) provide real, additional confidence for the rest of this
package, but are not a substitute for a real, maintained test suite.

One further finding -- a log message that no longer precisely
describes every cause of `DetectionError` after these 3 fixes -- was
investigated and deliberately left uncorrected; see
[`limitations.md`](./limitations.md) for why.
