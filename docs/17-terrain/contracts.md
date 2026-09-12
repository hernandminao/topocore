# `topocore.terrain` — Contracts

Every claim below was checked directly -- `issubclass()`,
`mypy` structural typing against a real usage site, or a real search
across the whole codebase -- not inferred from a docstring or a class
name. Where a docstring's own claim was checked against the real
code and found accurate, that's stated; where it wasn't, that's
stated just as plainly.

## `base.py`'s own 3 ABCs: 2 are real contracts, 1 is not

`base.py` declares `BaseTIN`, `BaseDTM`, `BaseInterpolator` as "the
public contracts implemented by TIN, DTM and interpolation
algorithms throughout the Terrain module" (its own module docstring).
Checked individually, by `issubclass()` against the real classes:

| ABC | Real implementers | Verified |
|---|---|---|
| `BaseDTM` | `DTM` | `issubclass(DTM, BaseDTM)` is `True` |
| `BaseInterpolator` | `LinearInterpolator`, `BarycentricInterpolator`, `IDWInterpolator`, `NearestInterpolator` | All 4 confirmed `True`; also confirmed by this project's own `test_interpolate_many.py::test_interpolator_inherits_base_interpolator`, which exists specifically because none of the 4 inherited from it at all in an earlier version (TD-001, already fixed) |
| `BaseTIN` | **none** | `BaseTIN.__subclasses__()` returns `[]`. `TIN` -- the sole TIN implementation anywhere in this codebase, confirmed by search -- does not inherit from it |

**`BaseTIN` is a real, confirmed gap, not a documentation nitpick.**
Its own docstring states plainly: "Base interface implemented by
every TIN model." That claim is false for the one TIN model that
exists. This isn't a naming mismatch that structural typing papers
over, either -- `BaseTIN` declares `triangles`/`points`/
`point_count`/`bounds()` (a method); `TIN` has `simplices`/
`vertices`/`vertex_count`/`bounds` (a property, not a method). A
caller typed against `BaseTIN` would not be able to use a real `TIN`
at all.

**This audit did not attempt to fix it**, for a reason different
from the other 8 corrections in [`validation.md`](./validation.md):
unlike those, closing this gap isn't a small, contained change --
it would mean either renaming several of `TIN`'s own attributes
(used throughout `analysis`, and confirmed structurally required by
`analysis.protocols.TriangulatedSurface` under their current names --
see below), or reworking `BaseTIN` itself to match `TIN`'s real
shape, neither of which fits inside this audit's own agreed scope.
Recorded here as a confirmed finding for a future, deliberate
decision, not silently left out of the documentation. See
[`limitations.md`](./limitations.md).

**`TerrainInterpolator` (the facade) is confirmed NOT a
`BaseInterpolator`** -- `issubclass(TerrainInterpolator,
BaseInterpolator)` is `False`. This is architecturally sound, not a
gap: the facade doesn't perform interpolation itself, it dispatches
to one of the 4 real interpolators that do, each of which already
satisfies the ABC on its own.

## `TIN` / `DTM` conformance to `analysis.protocols` -- confirmed via `mypy`, not assumed

Checked directly with `mypy`'s own structural typing, passing a real
`TIN`/`DTM` where each protocol is expected:

- `TIN` satisfies `analysis.protocols.TriangulatedSurface` -- confirmed,
  0 errors.
- `DTM` satisfies `analysis.protocols.GriddedSurface` -- confirmed,
  0 errors.
- `TIN` does **not** fully satisfy `analysis.protocols.PointCloudData`
  -- confirmed, `mypy` reports the single missing member: `array`.
  This is the same, already-investigated gap documented during this
  project's own `16-analysis` audit: `PointCloudData.array` has no
  implementation anywhere in the codebase and no consumer anywhere
  in `analysis` (confirmed by search during that audit) -- a
  deliberately unenforced protocol member, not a regression or a new
  finding here. `elevation_array()`/`xy_array()` -- the 2 members
  that *do* have real consumers (`analysis.statistics.ElevationStatistics.from_tin()`,
  `DensityStatistics.compute_from_tin()`) -- are confirmed to match
  exactly, both declared and implemented as plain methods, not
  properties. That contract mismatch was the one found and fixed
  during `16-analysis`'s own audit (on the `analysis` side, in
  `PointCloudData` itself -- `TIN` was already correct and was never
  changed); this session re-confirmed the real `TIN` source matches
  that fix precisely.

## `Grid` → `Raster` → `DTM`: composition, not inheritance

`DTM` does not reimplement raster storage or indexing -- confirmed
directly by reading every one of its own methods: `.elevation()`,
`.cell()`, `.statistics()`, `.array()`, `__getitem__`, `__iter__`,
`__len__` are each a one-line delegation to `self.raster`, and
`.rows`/`.columns`/`.bounds` delegate to `self.grid`/`self.raster`
the same way. The one piece of real logic `DTM` owns is
`DTM.from_tin()`'s own sampling loop (iterate every grid cell, query
the given interpolator, catch `InterpolationError` and leave `NaN`
for cells outside the TIN's hull) -- everything else is
composition. `Raster` itself is the same relationship one level down:
`Raster.cell()` delegates its own coordinate lookup to
`self.grid.coordinate()`, `Raster.contains()` to `self.grid.contains()`.

**One real, confirmed asymmetry in this chain**: `RasterSampler.nearest()`/
`.cell()` (in `sampling.py`, a separate utility class) validate
against `.contains()` before indexing into the raster; `Raster.elevation()`
itself did not, until this audit's own fix (see
[`validation.md`](./validation.md)) -- 2 pieces of code doing
functionally the same nearest-cell lookup, previously with 2
different levels of safety. Now consistent.

## The exception hierarchy -- confirmed by direct inheritance check

Every exception in `terrain.exceptions` inherits from `TerrainError`,
which itself inherits from `topocore.core.exceptions.TopoCoreError`
-- confirmed by reading the class definitions directly, not merely
by their names. 8 exceptions are declared
(`TerrainError`, `TriangulationError`, `InterpolationError`,
`ContourError`, `BreaklineError`, `GridError`, `TerrainValidationError`,
`ConversionError`); 7 of the 8 are exported at package level.
`GridError` is the one exception -- confirmed by search across the
entire codebase -- that is never raised anywhere, including within
`Grid`'s own source, despite its own name suggesting it's `Grid`'s
domain exception. See [`limitations.md`](./limitations.md).

## `algorithms/` -- deliberately isolated, confirmed by its own stated intent and by real usage

`terrain.algorithms`'s own package docstring states its isolation is
deliberate: "intentionally independent from the public domain models
... so they can be replaced or extended without affecting the public
API." Confirmed consistent with real usage: `DelaunayTriangulator`/
`ConstrainedDelaunayTriangulator`/`DelaunayResult` are never imported
by name from `terrain/__init__.py`'s own top-level exports, but are
used internally -- `TIN.from_points()` calls `DelaunayTriangulator`
directly, and `breaklines.apply_breaklines()` calls
`ConstrainedDelaunayTriangulator.triangulate()` directly, constructing
the resulting `TIN` via its own private `TIN(_result: DelaunayResult)`
constructor (confirmed directly: this is not an error or an unusual
back door -- `TIN.from_points()`/`.from_mesh()` themselves are built
the same way, and `ConstrainedDelaunayTriangulator.triangulate()`'s
own return value was confirmed, by real execution, to be
structurally accepted by this same constructor).

## `GridDefinition` / `validate_grid_definition()` -- confirmed legacy, self-documented

Unlike `BaseTIN` above, this one's own status is not left for this
audit to characterize -- `base.py`'s own docstring on `BaseDTM.grid`
states directly: `GridDefinition` is "an unrelated, older grid model
that `DTM` never actually used." Confirmed independently by this
audit, by search: `GridDefinition` has exactly one real consumer
anywhere in the codebase (`validate_grid_definition()`), and
`validate_grid_definition()` itself has zero callers anywhere beyond
its own declaration and its own tests. `validate_tin()` (a separate
function) is in the same position -- zero real callers, confirmed by
search. Both remain exported at package level. See
[`limitations.md`](./limitations.md) for the complete account,
alongside the 9 (of 12) unexported `constants.py` constants and 4
(of 6) unexported `enums.py` members this audit found in the same
category.

## What this document does not cover

Component-level behavior (exact numeric contracts for interpolation,
slope/aspect/hillshade, contour generation, filters) is covered in
each component's own document, not repeated here. This document is
specifically the cross-module contract surface: what one part of
`terrain` (or `terrain` and `analysis` together) promises another,
and whether that promise was confirmed to hold.
