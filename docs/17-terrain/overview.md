# `topocore.terrain` — Overview

## A note on how this document was built

Every claim below comes from one of 3 sources, and each is labeled
when it matters: **(a)** direct execution against the real source
files, performed during this audit's own 8-point review (`Grid`,
`TIN`, `DTM`, `Raster`/`Cell`, the 4 interpolators, `Slope`/`Aspect`/
`Hillshade`, `weights`/Delaunay, `contours`/`filters`/`breaklines`/
`validation`/utilities); **(b)** the real source files themselves,
read in full; or **(c)**, for 7 files this session's own sandbox lost
access to mid-audit (`terrain/__init__.py`,
`algorithms/__init__.py`, `algorithms/constrained_delaunay.py`,
`constants.py`, `enums.py`, `types.py`, `exceptions.py`) and were
never re-uploaded as standalone files, this document relies on the
detailed findings already recorded earlier in this same session
(exact class/constant/enum names, `__all__` contents, docstrings
quoted verbatim) rather than fresh re-verification. Where (c)
applies, this document says so. Nothing here describes a capability
because a terrain module "normally" has it -- every capability
described was read directly out of real code or a real, passing test.

## Purpose

`topocore.terrain` is the geometric and computational core for
representing, deriving, and analyzing terrain surfaces. It owns 2
independent representations of a terrain surface -- an irregular one
(`TIN`, a triangulated network built from arbitrary XYZ points) and a
regular one (`DTM`, a raster grid of elevations derived from a
`TIN`) -- plus the algorithms that operate on each: interpolation,
slope/aspect/hillshade, contour generation, smoothing/spike removal,
and breakline enforcement.

## Real scope — what this package does and does not own

**Owns**: triangulation (`TIN`, via `algorithms.delaunay` and
`algorithms.constrained_delaunay`), raster geometry and storage
(`Grid`, `Raster`, `Cell`, `DTM`), the 4 interpolation algorithms and
their facade, per-triangle slope/aspect/hillshade, marching-triangle
contour generation, TIN-vertex smoothing and spike detection,
breakline-constrained retriangulation, and the shared low-level
geometry/validation/NoData utilities all of the above are built on.

**Does not own**: statistical analysis of elevation/slope/density
values (that is `topocore.analysis.statistics`, which consumes `TIN`
directly -- see the "Integration" section below), surface comparison
and cut/fill volumes (`topocore.analysis.comparison`/`.volume`), or
feature extraction from terrain (`topocore.features.terrain`). Point
cloud ingestion is also external -- `terrain.conversion` is the one
narrow bridge, converting an already-built `PointCloud`
(`topocore.pointcloud`) into the `tuple[Point3D, ...]` this
package's own constructors expect.

## Real inventory — 30 files across 4 tiers

```text
topocore/terrain/
├── Core surfaces
│   ├── tin.py            TIN
│   ├── grid.py            Grid
│   ├── raster.py          Raster
│   ├── cell.py            Cell
│   └── dtm.py              DTM
├── algorithms/ (deliberately isolated from the public API -- see below)
│   ├── delaunay.py                  DelaunayResult, DelaunayTriangulator
│   └── constrained_delaunay.py      ConstrainedDelaunayTriangulator
├── Interpolation
│   ├── base.py             BaseTIN, BaseDTM, BaseInterpolator (ABCs)
│   ├── linear.py            LinearInterpolator
│   ├── barycentric.py        BarycentricInterpolator
│   ├── idw.py                 IDWInterpolator
│   ├── nearest.py              NearestInterpolator
│   └── interpolation.py         TerrainInterpolator (facade)
├── Terrain analysis (per-triangle, not raster-based -- see below)
│   ├── _geometry.py (private)    oriented_normal, slope_radians, aspect_radians
│   ├── slope.py                   SlopeCalculator, triangle_slope
│   ├── aspect.py                   AspectCalculator, triangle_aspect
│   └── hillshade.py                 HillshadeCalculator, triangle_hillshade
├── Derived products
│   ├── contours.py         ContourGenerator (marching triangles)
│   ├── filters.py           LaplacianSmoother, SpikeDetector
│   └── breaklines.py         BreaklineEnforcer
└── Shared utilities
    ├── weights.py       barycentric_weights, triangle_area, inverse_distance_weight, ...
    ├── nodata.py         is_nodata, valid_mask, replace_nodata, ...
    ├── conversion.py      pointcloud_to_points
    ├── sampling.py         RasterSampler (nearest/bilinear raster sampling)
    ├── validation.py        validate_resolution, validate_triangle, ...
    ├── models.py             Edge, Triangle, Breakline, GridDefinition, ContourLine
    ├── enums.py, types.py, constants.py, exceptions.py
    └── __init__.py            the package's own public surface
```

## Architecture — 2 surfaces, not 1

`TIN` and `DTM` are genuinely independent representations, not one
built as a thin wrapper over the other in a single direction that
consumers must all share:

- **`TIN`** is built directly from points (`TIN.from_points()`, via
  Delaunay triangulation) or from an already-known mesh
  (`TIN.from_mesh()`, preserving the caller's own triangle choices
  exactly -- confirmed directly: for a perfect square, where 2
  diagonals are equally valid Delaunay choices, `from_mesh()` keeps
  exactly the diagonal it was given, never silently re-triangulating
  to whichever one `scipy.spatial.Delaunay` itself would have
  picked). Every terrain-analysis algorithm in this package that
  operates per-triangle (interpolation, slope/aspect/hillshade,
  contours, filters, breaklines) operates on a `TIN` directly, never
  through a `DTM`.
- **`DTM`** is a raster derived FROM a `TIN` -- `DTM.from_tin(tin,
  grid, interpolator)` samples an already-built `TIN` at every cell
  of an already-built `Grid`, using a caller-supplied interpolator,
  producing a `Raster` internally. A grid cell whose query point
  falls outside the TIN's convex hull is left as `NaN`, not treated
  as a failure -- confirmed directly, and documented in `DTM`'s own
  docstring as the expected, common case for a rectangular grid over
  an irregular real survey boundary.

**Why slope/aspect/hillshade operate on the TIN, not a DTM raster**:
this is a deliberate, documented design choice, not an oversight.
Each module's own docstring states the reasoning explicitly:
computing per-triangle, directly from the exact plane each triangle
defines, gives an exact value for that facet; resampling to a raster
first and using a finite-difference (Horn's method) approximation
would introduce error a TIN-native computation doesn't have to
accept. `slope.py`'s own docstring names this as the same approach
"used by professional TIN-based earthwork engines."

## `Grid` / `Raster` / `Cell` — pure geometry, separate from values, separate from position

`Grid` (a frozen dataclass) defines only spatial geometry --
`min_x`/`min_y`/`max_x`/`max_y`/`resolution` and the row/column
arithmetic derived from them. It stores no elevations. `Raster` pairs
a `Grid` with a `values` array and a `nodata` sentinel (defaulting to
`NaN`); `Cell` is the single-point unit both `Grid` and `Raster`
return from their own per-cell accessors (`Grid.cell()`,
`Raster.cell()`), carrying row/column indices and x/y/z together.
This 3-way split is what lets `DTM` compose them rather than
re-implementing raster storage itself -- confirmed directly: `DTM`'s
own `.elevation()`, `.cell()`, `.statistics()`, `.array()`,
`__getitem__`, `__iter__`, `__len__` are all one-line delegations to
`self.raster`.

A real, historical fix lives in `Grid`'s own docstring (PR19,
already fixed, confirmed still correct by this audit): `columns`/
`rows` round UP (`math.ceil`) to guarantee full coverage of the
requested extent when `resolution` doesn't evenly divide
`width`/`height` -- correct and unchanged -- but `bounds`/
`contains()`/`Raster.transform` used to still describe the original,
nominal `max_x`/`max_y` rather than the grid's own actual generated
extent, so a grid's own last row/column could fail its own
`contains()` check, and GDAL/GeoTIFF export (`Raster.transform`)
could misalign. Fixed via `Grid.actual_max_x`/`actual_max_y`; this
audit re-confirmed the fix directly, including at real UTM-scale
coordinates.

## Interpolation — 4 independent algorithms, 1 facade

`LinearInterpolator`, `BarycentricInterpolator`, `IDWInterpolator`,
`NearestInterpolator` each implement `BaseInterpolator`
(`interpolate()`, `interpolate_many()`) independently over a `TIN`.
`LinearInterpolator` is confirmed, by its own docstring and by this
audit's direct execution, to be mathematically identical to
`BarycentricInterpolator` on a TIN (each triangle defines a plane) --
it exists as a distinct, named class for callers who want to state
their intent explicitly, not because the math differs.
`TerrainInterpolator` is the public facade selecting among all 4 via
an `InterpolationMethod` enum.

**A real defect in this facade was found and fixed during this
audit** (see [`validation.md`](./validation.md) for the full
account): its own dispatch used to compare `self._method` with `is`
rather than `==`, so a value that was genuinely *equal* to a valid
method but not the *identical object* -- including the natural,
easy-to-write case of passing a plain string like `method="linear"`,
since `InterpolationMethod` is a `StrEnum` -- silently fell through
to `NEAREST` instead, with no error, just a wrong elevation. This
compounded an earlier, separately-fixed bug in the same class (2
duplicate `InterpolationMethod` enum definitions, also compared with
`is`) -- that earlier fix addressed only the specific duplicate-enum
symptom, not this dispatch's own structural cause. Now fixed at the
root: explicit `==` comparison, with a clear `InterpolationError` for
any value that isn't one of the 4 known methods.

## Terrain analysis: slope / aspect / hillshade

All 3 share one private module, `_geometry.py`, as their own single
source of truth for a triangle's oriented normal
(`oriented_normal()`, always flipped to `z >= 0`) and for what
counts as a "flat" facet (`is_zero(normal.x) and is_zero(normal.y)`)
-- the module's own docstring states this explicitly, and this
audit confirmed the 3 modules can never disagree at the tolerance
boundary as a structural consequence, not a coincidence. Aspect is
the DOWNSLOPE compass bearing (0=North, 90=East, 180=South,
270=West), confirmed directly against all 4 cardinal directions and
against the module's own explicit statement that an ascending plane
toward East has aspect West, not East. `HillshadeCalculator`
implements the standard Lambertian illumination model (matching
GDAL's `gdaldem hillshade`), confirmed exactly against hand-derived
analytic cases including perpendicular incidence (exactly 255) and a
facet facing away from a low sun (clamped to exactly 0, never
negative).

## Contours, filters, breaklines

`ContourGenerator` implements marching triangles: intersect every
triangle against a horizontal plane at a given elevation, then weld
and stitch the resulting segments into `ContourLine` polylines (open
or closed) via a tolerance-quantized node-welding graph. A real,
CRITICAL historical bug (PR19, already fixed, re-confirmed by this
audit including at real UTM coordinate scale) lived here: a contour
level landing exactly on vertex elevations -- a flat pad or a graded
slope's toe at a round design elevation, a common real construction
shape, not a synthetic corner case -- used to disappear from the
output entirely.

`LaplacianSmoother`/`laplacian_smooth()` move only Z toward each
vertex's triangulation-neighbor average, confirmed directly to leave
X/Y bit-for-bit unchanged and, as a structural consequence (Delaunay
triangulation depends only on XY), to leave the triangulation's own
connectivity genuinely identical afterward -- confirmed directly, not
merely asserted by the docstring. `SpikeDetector`/`detect_spikes()`
flags vertices whose elevation deviates from their neighbor average
by more than a threshold; its own test suite documents (and this
audit confirmed) a real, inherent characteristic of neighbor-mean
detection, not a bug: an extreme outlier with very few neighbors also
skews those neighbors' own local averages enough to flag them too --
confirmed not to happen in denser, more realistic configurations.

`BreaklineEnforcer`/`apply_breaklines()` is the public entry point
for `algorithms.constrained_delaunay` -- it merges breakline vertices
into the existing point set (deduplicating by XY; a breakline point
whose XY coincides with an existing vertex has ITS OWN elevation
silently discarded in favor of the original point's, confirmed
directly and documented in the function's own docstring -- worth
knowing deliberately, since it could surprise a caller expecting a
breakline to correct an existing point's own elevation), then builds
the constraint-edge list for `HARD` breaklines only (`SOFT`
breaklines contribute vertices for elevation accuracy without forcing
rigid edges).

## Shared utilities

`weights.py` holds the low-level, TIN-independent math
(`barycentric_weights()`, `triangle_area()`, `inverse_distance_weight()`,
...) that `barycentric.py`/`idw.py` build on. `nodata.py` provides a
small, consistent vocabulary around the `NaN`-as-NoData convention
used throughout this package (`Raster.nodata` defaults to `NaN`,
`Cell.has_data`/`.is_nodata` check `math.isfinite(z)`,
`DTM.from_tin()` leaves out-of-hull cells as `NaN`) -- though this
audit could not confirm any of `nodata.py`'s own 7 functions have a
real caller anywhere in the codebase beyond their own tests.
`sampling.py`'s `RasterSampler` provides nearest and bilinear raster
sampling independent of `Raster`'s own `.elevation()` method;
confirmed directly that `RasterSampler.nearest()`/`.cell()` validate
against `.contains()` before indexing -- a pattern `Raster.elevation()`
itself was missing until this audit's own fix (see
[`validation.md`](./validation.md)). `conversion.py`'s
`pointcloud_to_points()` is this package's one bridge from
`topocore.pointcloud.PointCloud` into the `Point3D` tuples every
constructor here expects.

## Integration with `processing` and `analysis`

Confirmed directly, by import: nothing in `topocore.processing`
imports from `topocore.terrain`, and nothing in `topocore.terrain`
imports from `topocore.processing` -- the 2 packages are independent.
`topocore.analysis` is a real, confirmed consumer of `terrain`
types: `analysis.statistics.ElevationStatistics.from_tin()` and
`analysis.statistics.DensityStatistics.compute_from_tin()` call
`TIN.elevation_array()`/`.xy_array()` directly (both plain methods,
not properties -- the exact contract whose mismatch with
`analysis.protocols.PointCloudData` was found and fixed during
`16-analysis`'s own audit; this session confirmed the real `TIN`
source matches that fix exactly). `analysis`'s own `TerrainSurface`/
`TriangulatedSurface`/`GriddedSurface` protocols are structurally
satisfied by `TIN`/`DTM` respectively.

## NoData / NaN — one consistent convention, confirmed across every layer

`NaN` is the single NoData sentinel used consistently from the
lowest level to the highest: `Point3D`'s own validation rejects `NaN`
outright (confirmed directly -- this is what made `Grid.point()`
genuinely unusable, see [`validation.md`](./validation.md)) so a
"point with unknown elevation" is never represented as a `Point3D`
anywhere in this package; `Cell`/`Raster` freely allow and expect
`NaN` (`Cell.has_data` is exactly `math.isfinite(z)`); `DTM.from_tin()`
leaves out-of-hull cells as `NaN` rather than failing; `Raster.fill()`/
`.mask()` operate on the same convention; bilinear sampling
(`RasterSampler.bilinear()`) returns `NaN` whenever any of the 4
neighboring cells is NoData, confirmed directly.

## Validation — real callers, and real orphans

`validation.py`'s `validate_resolution`/`validate_interval`/
`validate_points`/`validate_triangle`/`validate_breakline` all have
confirmed real callers (`Grid.__post_init__`, `ContourGenerator`,
`_geometry.oriented_normal()`, `breaklines.py`). `validate_grid_definition()`
and `validate_tin()` are confirmed, by search, to have zero callers
anywhere beyond their own declaration and tests -- the same is true
of the `GridDefinition` model they validate, whose own status as
unused legacy is self-documented directly in `base.py`'s own
docstring ("an unrelated, older grid model that `DTM` never actually
used"). See [`limitations.md`](./limitations.md) for the complete,
itemized account of this and the rest of this package's declared-but-
unconnected surface (`GridError`, 9 of 12 constants, 4 of 6 enums).

## Implementation status

9 real defects were found and fixed during this audit's own
8-point review, verified individually with direct execution, `ruff`,
and `mypy` against the real project configuration, plus a full
project test-suite run showing no regressions. See
[`validation.md`](./validation.md) for the complete account of each,
including the evidence that established it as a genuine defect
rather than an intentional design choice. A further 19 `ruff`
findings were confirmed pre-existing in files this specific
correction batch did not touch, and were deliberately left
unaddressed to keep this audit's own scope to what was agreed --
documented in full in [`limitations.md`](./limitations.md), not
hidden.

## Where to go next

- [`contracts.md`](./contracts.md) -- the cross-module contracts this
  audit specifically verified (Protocol/implementation conformance,
  `TIN`/`DTM` consumption by `analysis`, the `GridDefinition`/
  `validate_grid_definition()` legacy question).
- [`grid-raster-cell.md`](./grid-raster-cell.md), [`tin.md`](./tin.md),
  [`interpolation.md`](./interpolation.md),
  [`slope-aspect-hillshade.md`](./slope-aspect-hillshade.md),
  [`contours.md`](./contours.md),
  [`filters-breaklines.md`](./filters-breaklines.md),
  [`utilities.md`](./utilities.md) -- per-component detail, each
  verified the same way as this overview: real execution against
  real code, not assumed behavior.
- [`validation.md`](./validation.md) -- the 9 fixed defects, in full,
  with evidence.
- [`limitations.md`](./limitations.md) -- the 20 pre-existing
  findings and the declared-but-unconnected surface, in full.
