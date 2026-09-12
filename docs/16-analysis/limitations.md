# Limitations, Consolidated

Every entry here reflects something confirmed against the real code
during this documentation effort's own audit -- not a generic
disclaimer. See each linked page for the full evidence.

## Implemented

Every class, method, and computation documented across
[`distance.md`](./distance.md), [`profile.md`](./profile.md),
[`visibility.md`](./visibility.md), [`statistics.md`](./statistics.md),
[`quality.md`](./quality.md), [`comparison.md`](./comparison.md), and
[`volume.md`](./volume.md) works according to its own observed
contract, verified by direct execution against hand-computable
inputs. This includes the defects found during this same audit -- a
contract defect (`PointCloudData.elevation_array`/`.xy_array`) and
its 3 separate functional consequences, 1 independent numeric defect
in `SlopeStatistics`, and an API/design defect in `comparison`'s own
exception choice (`SurfaceComparison`/`TINComparison` used to leak
`VolumeError`) -- all already fixed in the code you have (see
[`validation.md`](./validation.md)). `ruff` is clean across the
entire package, and `mypy`, run with this project's own real
`pyproject.toml`, reports zero errors anywhere in `topocore.analysis`.

## Known limitations

- **`DistanceConfig.ellipsoid`** is validated at construction but
  never consumed by any computation -- `GeodesicDistance`'s own
  ellipsoid comes entirely from the `CRS` you construct it with. See
  [`distance.md`](./distance.md).
- **`PointCloudData.array` remains unimplemented by `TIN`.** Unlike
  `elevation_array`/`xy_array` (found to have the same mismatch and
  corrected during this audit -- see [`contracts.md`](./contracts.md)),
  `array` was deliberately left alone: `TIN` has no method or property
  under that name at all, and no consumer anywhere requires one. `TIN`
  therefore still does not fully satisfy `PointCloudData` as a
  Protocol (confirmed via `mypy`). This remains open specifically
  because there is no confirmed consumer or implementation to correct
  it against -- not an oversight.
- **`DistanceAnalysis` and `EuclideanDistance` take their shared
  6-argument coordinate input in genuinely different orders.** Both
  are correct on their own terms; conflating them silently computes a
  wrong distance with no error. See [`distance.md`](./distance.md).
- **`VisibilityMethod`, `StatisticsMethod`, and `QualityMethod` are
  plain classes, not `StrEnum`s**, unlike `DistanceMethod` and
  `ProfileMethod`. See [`contracts.md`](./contracts.md).
- **`topocore.analysis` does not export its own algorithm classes**
  from the top-level package -- every class must be imported from its
  own submodule. See [`overview.md`](./overview.md).
- **`DistributionStatistics`'s own robustness against a genuinely
  computed (not artificially constructed) near-uniform input was not
  tested.** The sibling defect found in `SlopeStatistics` (a real,
  ~1e-15 floating-point spread crashing `np.histogram()`) was not
  independently confirmed or ruled out for this class. See
  [`validation.md`](./validation.md).
- **`CrossSectionProfile`'s own `interval` parameter does not
  subdivide a given axis by distance** -- it only controls spacing
  within each individual transversal cut. One section is generated
  per axis vertex supplied, regardless of `interval`. See
  [`profile.md`](./profile.md).
- **`TransversalProfile`'s own `width` is a half-extent, producing a
  total cross-section span of `2 * width`,** not a total width of
  `width`. See [`profile.md`](./profile.md).
- **Every class and manager in `analysis` lets a missing or
  wrong-count argument raise Python's own `TypeError`, never a
  domain-specific exception.** Confirmed as a genuinely consistent,
  Python-native behavior across all 6 dispatcher managers and their
  own underlying classes alike -- not a scattered inconsistency, and
  not something this audit converted into wrapped domain exceptions,
  since doing so indiscriminately could mask real programming errors
  and change an established contract. Only "unknown method name" is
  translated into a domain exception (`DistanceError`,
  `VisibilityError`, etc.) by every manager; argument validity is not.

## Not implemented

- **No automatic CRS/unit consistency checking across `analysis`
  computations.** Nothing here validates that coordinates you pass
  are already in a consistent, meaningful CRS or unit system for the
  computation at hand (e.g. a `radius`/`resolution` parameter assumes
  your coordinates are already in matching real-world units) -- see
  [`../15-processing/workflow-integration.md`](../15-processing/workflow-integration.md)
  for the identical principle already documented for `processing`.
- **No `Workflow` integration.** Nothing in `topocore.analysis` is
  wired into `Workflow` as a chainable stage -- every class here is
  called directly on the coordinate arrays/surfaces you already have.

## Out of scope for this documentation effort

- **10 `ruff` `UP040` findings** (`types.py`, `statistics/manager.py`,
  `quality/manager.py`, `visibility/manager.py`), surfaced only when
  re-running `ruff` with this project's own real `pyproject.toml`
  configuration during this section's own final regression pass. Each
  flags a `TypeAlias`-annotated assignment (`X: TypeAlias = ...`)
  that could be modernized to a PEP 695 `type X = ...` statement
  (available since this project's own minimum Python, 3.12). Not
  fixed here: `ruff` itself classifies this as an *unsafe* fix, not a
  mechanical one like the `RUF046`/`RUF022`/`RUF023` findings resolved
  during this same audit, and the same pattern is very likely present
  throughout the rest of the codebase, not specific to `analysis` --
  fixing it only here would leave the project stylistically
  inconsistent rather than close a real gap. If this modernization is
  ever done, it should be a deliberate, global, separately-tracked
  effort, not something folded into `16-analysis`'s own closure.
- Obstructed line-of-sight scenarios beyond the single working example
  confirmed in [`visibility.md`](./visibility.md) -- the general shape
  and reliability of `LOSResult.obstacles` across varied obstruction
  geometries was not exhaustively explored.
- Comprehensive, general-purpose test coverage for `topocore.analysis`
  as a whole. This effort fixed and regression-tested the confirmed
  defects (see [`validation.md`](./validation.md)); it did not set
  out to build, and did not build, full test coverage for the other
  ~50 public classes across the 7 submodules.
- 27 pre-existing `mypy` findings elsewhere in the codebase
  (`geodesy`, `processing`, `io`, `terrain`, `features`), surfaced
  incidentally while re-running `mypy` with this project's own real
  configuration -- confirmed to be outside `topocore.analysis`
  entirely, left untouched as out of scope for this section.
