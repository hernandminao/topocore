# Contracts and Deliberate Inconsistencies

This page consolidates the specific, confirmed points across
`topocore.analysis` most likely to cause confusion -- each one
already noted in its own submodule page, gathered here for reference
in one place. Most of these are not defects (see
[`validation.md`](./validation.md) for the ones that are -- the
original 3 found during this project's own Protocol/statistics audit,
plus a 4th found during the later manager/dispatcher audit,
`StatisticsAnalysis.slope()`); the rest are real API decisions and
real inconsistencies worth knowing about precisely.

## Every manager leaks a raw `TypeError` for wrong/missing arguments

Confirmed directly, and confirmed uniform -- not scattered or
inconsistent between managers: all 6 dispatcher classes
(`DistanceAnalysis`, `ProfileAnalysis`, `VisibilityAnalysis`,
`StatisticsAnalysis`, `QualityAnalysis`, `VolumeAnalysis`) correctly
translate an *unrecognized method name* into their own domain
exception (`DistanceError`, `ProfileError`, etc.), but **none of them
translate wrong or missing arguments** -- that always leaks a plain
Python `TypeError` from whichever underlying method/class actually
received the call, e.g.:

```python
>>> ProfileAnalysis(method="longitudinal").compute((0.0, 0.0))
TypeError: ProfileAnalysis.longitudinal() missing 2 required positional arguments: 'target' and 'surface'
>>> VolumeAnalysis(method="cut_fill").compute(existing, proposed)   # missing cell_area
TypeError: VolumeAnalysis.cut_fill() missing 1 required positional argument: 'cell_area'
```

(`DistanceAnalysis`'s own `euclidean` branch is the sole exception,
validating its own argument count explicitly -- see
[`distance.md`](./distance.md).)

**Classified as a consistent design limitation of this section's own
dispatch layer, not a defect** -- the pattern is uniform across all 6
managers, not an accidental omission in one of them, and blanket-wrapping
every possible `TypeError` into a domain exception risks masking
genuine programming errors (a wrong keyword name, a caller's own bug)
behind a misleadingly domain-specific message. Documented here as the
contract as it actually stands: a wrong method name is always a clean,
catchable domain exception; wrong/missing arguments are always a
plain Python `TypeError`.

## Argument order: `DistanceAnalysis` vs. `EuclideanDistance`

```python
DistanceAnalysis.compute()      # (x1, y1, z1, x2, y2, z2) -- point-grouped
EuclideanDistance.compute()      # (x1, y1, x2, y2, z1, z2) -- EuclideanDistance's own native order
```

Confirmed deliberate: a real, severe defect existed previously where
`DistanceAnalysis` forwarded `*args` straight through unreordered,
silently computing a wrong distance with no error (a 3-4-12 triangle
silently returning `8.544` instead of `13.0`). The fix adopted the
natural, point-grouped order as `DistanceAnalysis`'s own public
contract -- matching what `SlopeDistance.compute()`/
`GeodesicDistance.compute()` already exposed -- rather than matching
`EuclideanDistance`'s own internal order. See
[`distance.md`](./distance.md) for the full account.

## `StrEnum` vs. plain classes: the 5 dispatcher "Method" selectors

| Selector | Where | Real `StrEnum`? |
|---|---|---|
| `DistanceMethod` | `distance/manager.py` | Yes |
| `ProfileMethod` | `profile/manager.py` | Yes |
| `VisibilityMethod` | `visibility/manager.py` | No -- plain class |
| `StatisticsMethod` | `statistics/manager.py` | No -- plain class |
| `QualityMethod` | `quality/manager.py` | No -- plain class |

Confirmed directly: the 3 "No" entries are not iterable, not
comparable as enum members, and don't support `list(...)` the way the
2 real `StrEnum`s do -- though all 5 still hold usable string values
(e.g. `QualityMethod.CLOUD_TO_CLOUD == "c2c"`). This inconsistency is
confined to these 5 dispatcher-selector classes specifically --
`VolumeMethod`, `ProfileType`, `VisibilityType`, and every other
`StrEnum` defined in `topocore.analysis.types` are genuine `StrEnum`s,
unaffected by this.

## `GridVolume` and `CutFillVolume` compute the same thing

See [`volume.md`](./volume.md) for the full, verified account:
identical formula (both delegate to the same shared
`compute_cut_fill()`), identical results confirmed with non-uniform
input, `method` field metadata only. The real difference is each
class's own DTM-integration method (`compute_from_dtm()` vs.
`compute_with_dtm()`), which have different resolution-mismatch
behavior by deliberate design, not by accident.

## The `PointCloudData` contract mismatch -- found, and corrected

`PointCloudData` (a `Protocol` in `analysis.protocols`) used to type
both `elevation_array` and `xy_array` as `@property`. `TIN` -- the
sole concrete implementation of this protocol anywhere in the
codebase, confirmed by search -- exposes both as plain methods
instead. This was corrected at the source: `PointCloudData` now types
both as methods, matching `TIN`'s own real, working, tested behavior
-- `TIN` itself was never the one considered wrong here, since it is
the only implementation that exists and every real usage already
depended on its method-based behavior.

This one contract defect had 3 separate, independently-reproducible
functional consequences before the fix, all raising `TypeError`
unconditionally: `ElevationStatistics.from_tin()`,
`StatisticsAnalysis`'s own duck-typed elevation dispatch, and
`DensityStatistics.compute_from_tin()` (the third occurrence,
`.xy_array` rather than `.elevation_array`, found only once the audit
deliberately searched for every consumer of both protocol members
rather than stopping at the first fix). All 3 call sites now call the
corrected method directly (`tin.elevation_array()`, `tin.xy_array()`),
with no `# type: ignore` needed for this reason anymore. See
[`validation.md`](./validation.md) for the full account, and
[`statistics.md`](./statistics.md) for where each of the 3 surfaces.

**`PointCloudData.array` was deliberately left unchanged.** `TIN`
does not implement `array` at all -- confirmed by search, there is no
method or property under that name anywhere on `TIN` -- and no
consumer anywhere requires it. Changing `array`'s own declared shape
without any implementation or consumer to verify against would be
speculative, not a confirmed fix; `TIN` therefore still does not
fully satisfy `PointCloudData` as a Protocol (confirmed directly via
`mypy`, which reports `array` as `TIN`'s sole remaining missing
member) -- a separate, correctly-scoped gap, not something this fix
addressed or needed to.

## `topocore.analysis` does not export its own algorithm classes

```python
from topocore.analysis import EuclideanDistance   # ImportError
from topocore.analysis.distance import EuclideanDistance  # correct
```

The top-level package's own `__all__` (53 names) contains result
types, configuration dataclasses, exceptions, and protocols -- never
an algorithm class. Every class documented across
[`distance.md`](./distance.md) through [`volume.md`](./volume.md)
must be imported from its own submodule.

## A dispatcher's own selector string does not always match its dedicated method's name

Confirmed directly, 2 real cases where guessing the dispatcher string
from the dedicated method's own name gives a wrong, rejected value:

```python
VisibilityAnalysis(method="line_of_sight")   # wrong -- VisibilityError
VisibilityAnalysis(method="los")              # correct
VisibilityAnalysis().line_of_sight(...)       # the dedicated method itself, unaffected

QualityAnalysis(method="cloud_to_cloud")      # wrong -- QualityError
QualityAnalysis(method="c2c")                  # correct
```

Every other manager's own selector strings were verified directly and
matched their obvious name (`"elevation"`, `"euclidean"`,
`"longitudinal"`, `"tin_volume"`, `"grid_volume"`, etc.) -- these 2
are the only confirmed exceptions.

## `CoordinateTransformer` (the protocol) has no consumer within `analysis` -- classified as an intentional, unwired structural-typing bridge

`analysis.protocols.CoordinateTransformer` is exported from
`topocore.analysis`'s own top-level `__all__`, but confirmed by
search: no class, function, or method anywhere in `topocore.analysis`
actually uses it as a type hint today.

This was investigated as 3 possibilities: (A) an intentional public
API, (B) a stale, historical declaration with no real basis, or (C) a
genuine integration gap. The evidence points to (A), though without
an explicit authorial statement confirming it: `transform_point(x, y,
z=None) -> tuple[float, float, float | None]` is an **exact** match
-- confirmed directly, same parameter names, defaults, and return
shape -- for `topocore.geodesy.CoordinateTransformer.transform_point()`,
the real, extensively-used class throughout `topocore.geodesy` and
`topocore.workflow`. A stale or speculative declaration would be
unlikely to reproduce a real, actively-maintained class's own exact
signature this precisely by coincidence. The most likely purpose:
letting a future `analysis` function accept a
`geodesy.CoordinateTransformer` (or anything shaped like it)
structurally, without `analysis` needing to import `topocore.geodesy`
directly -- consistent with `analysis`'s own confirmed pattern
elsewhere of taking raw arrays/protocol-typed parameters rather than
importing concrete classes from sibling packages.

**Not currently wired into any real `analysis` function** -- no
coordinate transformation happens anywhere in `analysis` today, via
this protocol or otherwise (see
[`../15-processing/workflow-integration.md`](../15-processing/workflow-integration.md)
for the identical, deliberate absence already documented for
`processing`). Left as-is: removing it would delete a precisely-matched,
plausibly-intentional piece of public API on the strength of "nothing
calls it yet"; wiring it into a function it was never designed for
would be inventing an integration rather than confirming one.

## `comparison` has no dispatcher -- verified as an intentional asymmetry, not an oversight

Unlike the other 6 submodules, `comparison` exports no `*Analysis`
manager class. Investigated directly rather than assumed either way:

- No module docstring, comment, or historical note explains this
  either way -- `comparison/__init__.py`'s own docstring is a single
  line ("Surface comparison and cut/fill analysis."), no design
  rationale stated.
- No code anywhere in the project (`Workflow`, tests, or otherwise)
  references or expects a `ComparisonAnalysis`.
- **The 3 classes' own signatures are genuinely incompatible with a
  shared dispatcher, confirmed directly**:

```python
SurfaceComparison.compute(existing: FloatArray, proposed: FloatArray) -> SurfaceComparisonResult
TINComparison.compute(existing: TriangulatedSurface, proposed: TriangulatedSurface) -> SurfaceComparisonResult
SurfaceCutFill.compute(existing: FloatArray, proposed: FloatArray) -> tuple[SurfaceComparisonResult, VolumeResult]
```

`TINComparison` takes triangulated surfaces, not arrays -- a
genuinely different input type from the other 2. `SurfaceCutFill`
returns a 2-tuple, not a single result -- a genuinely different
output shape from `SurfaceComparison`. This is structurally different
from every other submodule's own dispatcher: `DistanceAnalysis`'s 5
methods, `QualityAnalysis`'s 10, and every other manager's own
method set all share one common result type and a uniform calling
convention, which is precisely what makes a `method="..."` dispatcher
coherent for them. `comparison`'s own 3 classes could not be unified
behind one `.compute()` return type without either changing one of
them or accepting a dispatcher whose return type varies by which
method was selected -- a materially different, and weaker, contract
than every other manager in this section provides. **Classified as a
verified, intentional asymmetry**: building a `ComparisonAnalysis`
purely to make the submodule table symmetric would introduce an
interface `comparison`'s own real classes don't naturally support,
not confirm one that was missing.

This is a separate question from `comparison`'s own use of
`VolumeError`, addressed next -- lacking a dispatcher and sharing an
exception with `volume` are 2 independent design questions.

## `comparison`'s use of `VolumeError` -- investigated as a possible defect, confirmed intentional

`SurfaceComparison` and `TINComparison` raise `VolumeError` for
their own validation failures, including `comparison/surface.py`'s
own reuse of `_shared.volume.validate_volume_arrays()` for its
shape/NaN checks. This was investigated seriously during this
audit as a candidate defect -- `VolumeError`'s own docstring ("Raised
when a volume calculation fails") and `_shared.volume`'s own
documented list of intended consumers (grid volume, cut/fill
analysis, terrain volume methods) don't explicitly name surface
comparison, which initially looked like evidence of an unintended
leak from `volume`'s own domain.

**That conclusion was reversed once the real project's own test
suite was available.** `tests/analysis/comparison/test_public_api.py`
includes `test_documented_error_paths_raise_only_volume_error` -- a
test whose own name asserts `VolumeError` as this API's documented,
intended contract. A fix was implemented and then fully reverted
once this came to light. The methodological lesson, recorded here
deliberately: the absence of a counter-example in an incomplete test
copy is not evidence of its absence in the real project, and a
docstring's silence on a consumer is much weaker evidence than an
existing, explicitly-named test. See
[`comparison.md`](./comparison.md) and
[`validation.md`](./validation.md) for the full account.

## Protocol/implementation conformance, verified systematically

Beyond `PointCloudData` (see above), every other protocol in
`analysis.protocols` was checked against its own real implementation,
via `mypy` structural typing plus direct execution:

| Protocol | Real implementation | Verified |
|---|---|---|
| `TerrainSurface` | `TIN` | Clean, both `mypy` and execution |
| `TriangulatedSurface` | `TIN` | Clean, both `mypy` and execution |
| `GriddedSurface` | `DTM` | Clean via `mypy` -- already fixed once, in an earlier PR, for this exact class of defect (see that protocol's own docstring) |
| `CRSType` | `topocore.geodesy.CRS` | Clean via `mypy` |
| `CoordinateTransformer` | none within `analysis` (matches `geodesy.CoordinateTransformer` exactly) | N/A -- classified as an intentional, unwired structural-typing bridge; see below |

A systematic search across every consumer in `topocore.analysis` for
the attribute patterns `.array`, `.points`, `.coordinates`,
`.vertices`, `.faces`, `.normals`, `.values`, `.elevations`, `.grid`,
`.resolution`, `.x`/`.y`/`.z` found no further property/method
mismatches -- `.elevations`/`.grid`/`.resolution` (via
`GriddedSurface`) and `.x`/`.y`/`.z` (via `Point3D`, a plain
dataclass) are all consumed exactly as declared.  `.coordinates`,
`.vertices`, `.faces`, `.normals`, `.values`, and `.array` have zero
consumers anywhere in `topocore.analysis`'s own code.

## Relationship to `processing`

`topocore.analysis.exceptions.AnalysisError` inherits directly from
`topocore.processing.exceptions.ProcessingError` -- confirmed by
reading the class definition, not merely an import. This is the only
confirmed dependency between the 2 packages in either direction --
see [`overview.md`](./overview.md).

## Cross-dependencies within `analysis` itself

Only [`comparison`](./comparison.md) depends on another submodule
within `analysis` (`volume`, specifically `CutFillVolume`) --
confirmed by searching every submodule's own imports. `distance`,
`profile`, `visibility`, `statistics`, `quality`, and `volume` do not
import from each other.

## `DistanceConfig.ellipsoid` is declared but never consumed

See [`distance.md`](./distance.md) -- validated at construction
(rejects an empty string) but never read anywhere else in the
codebase. `GeodesicDistance`'s own ellipsoid comes entirely from the
`CRS` object you construct it with.
