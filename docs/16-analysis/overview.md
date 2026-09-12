# Analysis — Overview

`topocore.analysis` computes distances, terrain profiles, visibility,
descriptive statistics, quality metrics between point clouds/surfaces,
surface-to-surface comparisons, and earthwork volumes. It does not
reduce, filter, or classify a `PointCloud` into another `PointCloud`
-- that is [`topocore.processing`](../15-processing/overview.md)'s own
job. Analysis consumes the outputs of reading, processing, and
terrain-building; it produces numbers, results, and reports, not new
point-cloud or terrain artifacts.

## Where this fits

```text
IO / Processing / Terrain
          |
          v
  PointCloud, TIN, DTM, coordinate arrays
          |
          v
      topocore.analysis
          |
          v
  distances, profiles, statistics, quality
  metrics, comparisons, volumes
```

## The 7 submodules

| Submodule | Computes | Public classes |
|---|---|---|
| [`distance`](./distance.md) | Point-to-point distance, 5 methods | 7 |
| [`profile`](./profile.md) | Terrain profiles along a path | 6 |
| [`visibility`](./visibility.md) | Line of sight, viewshed, intervisibility | 5 |
| [`statistics`](./statistics.md) | Descriptive statistics (elevation, slope, area, density, distribution) | 7 |
| [`quality`](./quality.md) | Point-cloud/surface accuracy and comparison metrics | 12 |
| [`comparison`](./comparison.md) | Surface-to-surface spatial comparison | 4 |
| [`volume`](./volume.md) | Earthwork cut/fill and single-surface volume | 6 |

Cross-dependencies between the 7 are minimal, confirmed directly by
searching every submodule's own imports: only `comparison` depends on
another submodule (`volume`, specifically `CutFillVolume` -- see
[`comparison.md`](./comparison.md)). `distance`, `profile`,
`visibility`, `statistics`, `quality`, and `volume` do not import from
each other at all.

**6 of the 7 submodules expose a `*Analysis` dispatcher class
(`DistanceAnalysis`, `ProfileAnalysis`, `VisibilityAnalysis`,
`StatisticsAnalysis`, `QualityAnalysis`, `VolumeAnalysis`) -- each
verified directly via real execution to correctly dispatch through
both its own dedicated methods and its generic `__call__`/`.compute()`
path.** `comparison` has no dispatcher at all -- verified as an
intentional asymmetry, not an oversight: its own 3 classes have
genuinely incompatible input/output shapes with each other (one takes
triangulated surfaces rather than arrays; another returns a 2-tuple
rather than a single result), unlike every other submodule's own
method set, which shares one common result type. See
[`contracts.md`](./contracts.md) for the full evidence.

## Relationship to `processing` — a real, confirmed dependency, not a coincidence

`topocore.analysis.exceptions.AnalysisError` **inherits directly**
from `topocore.processing.exceptions.ProcessingError` -- confirmed by
reading the class definition itself, not merely an import. Every
exception raised anywhere in `analysis` (`DistanceError`,
`VolumeError`, `ProfileError`, `VisibilityError`, `StatisticsError`,
`QualityError`) is therefore also a
`ProcessingError`. This is the **only** confirmed dependency between
the two packages -- no `analysis` algorithm imports or calls into
`processing`'s own algorithms (neighbor search, filters,
classification, etc.), and
`processing` does not import from `analysis` at all.

## What you import from where

**`topocore.analysis` itself does not export the working algorithm
classes.** Confirmed directly: `from topocore.analysis import
EuclideanDistance` raises `ImportError`. The top-level package exports
53 names -- result types (`DistanceResult`, `VolumeResult`,
`ElevationStats`, ...), configuration dataclasses (`DistanceConfig`,
`VolumeConfig`, ...), exceptions, and protocols (`TerrainSurface`,
`PointCloudData`, ...) -- but every actual algorithm class
(`EuclideanDistance`, `LineOfSight`, `RMSEAnalysis`, `CutFillVolume`,
and everything else covered in the 7 pages linked above) must be
imported from its own submodule directly:

```python
from topocore.analysis.distance import EuclideanDistance   # correct
from topocore.analysis import EuclideanDistance             # ImportError
```

## State of test coverage — read this before assuming "implemented" means "test-covered"

**A methodological correction, stated plainly**: this section used to
claim `topocore.analysis` had "zero test coverage of any kind" before
this documentation effort's own audit. That was true only of the
sandboxed environment this audit's own tooling ran in, which had no
`tests/analysis/` directory at all -- it was never true of the real
project, which has substantial, pre-existing test coverage for
`analysis` (dedicated test files for `statistics`, `comparison`, and
others). Generalizing a sandbox's own limited visibility into a claim
about the real project was a mistake; see
[`validation.md`](./validation.md) for how this was caught (a "fix"
for a `comparison` exception-hierarchy question was reverted in full
once the real project's own test suite showed an existing test
explicitly asserting the original behavior as its own intended
contract) and what it changed.

What remains true, and was verified directly against the real
project's own pre-fix source code, not merely inferred: a contract
defect (`PointCloudData.elevation_array`/`.xy_array` declared as
properties while `TIN`, the sole implementation, exposes both as
methods), 3 separate functional defects that were its direct
consequence, 1 independent numeric defect in `SlopeStatistics`, and 1
API defect in `StatisticsAnalysis.slope()` itself (narrower than the
class it wraps) -- see [`validation.md`](./validation.md) for the
complete account, including the one investigated defect that turned
out not to be one. This audit added 12 regression tests, alongside
the real project's own pre-existing coverage. **12 regression tests
are evidence that these specific fixed defects stay fixed -- they are
not, and should not be read as, this package's only test coverage.**
Every claim in this documentation set of "implemented" reflects
direct verification performed during this audit (execution against
known, hand-checked results, or direct comparison against the real
project's own source), not an assumption that importing successfully
means correct behavior. Where a claim could not be verified this way,
it is stated as such rather
than presented as confirmed.

The honest summary: **implementation functionally audited and
validated through direct execution and regression tests; the
project's full test suite passes with no regressions.** This is
meaningfully different from, and should never be conflated with,
"comprehensively tested."

## Terminology used consistently across this section

- **Implemented**: works according to its own observed contract,
  verified by direct execution.
- **Defect**: the code does not do what its own contract/docstring
  says (all 3 found during this audit are already fixed -- see
  [`validation.md`](./validation.md)).
- **Known limitation**: works, but with a real, stated restriction.
- **Not implemented**: the capability/API does not exist.
- **Out of scope**: not part of this documentation effort's own
  target.

## In this section

- [`distance.md`](./distance.md)
- [`profile.md`](./profile.md)
- [`visibility.md`](./visibility.md)
- [`statistics.md`](./statistics.md)
- [`quality.md`](./quality.md)
- [`comparison.md`](./comparison.md)
- [`volume.md`](./volume.md)
- [`contracts.md`](./contracts.md) -- decisions and inconsistencies
  worth understanding precisely before they cause confusion.
- [`validation.md`](./validation.md) -- what was audited, what was
  found, what was fixed, and what running the fix's own regression
  tests actually proves.
- [`limitations.md`](./limitations.md) -- consolidated, by category.
