# Validation

## A methodological correction, stated first because it affects how to read everything below

Earlier drafts of this page claimed `topocore.analysis` had "zero
test coverage of any kind" before this documentation effort's own
audit. **That claim was wrong for the real project, and should never
have been generalized this way.** It was true only of the sandbox
environment this audit's own tooling ran in, which had a stripped-down
test tree with no `tests/analysis/` directory at all. The real
project has substantial, pre-existing test coverage for `analysis` --
confirmed directly, by the project's own test run showing dedicated,
passing test files for `statistics` (`test_elevation.py`,
`test_slope.py`, `test_manager.py`, `test_density.py`, `test_area.py`,
`test_distribution.py`) and for `comparison`
(`test_surface_comparison.py`, `test_tin_comparison.py`,
`test_edge_cases.py`, `test_public_api.py`), among others. The lesson
recorded here deliberately: a sandbox's own limited visibility is not
evidence about the real project, and a claim this sweeping should
never have been made without being able to verify it directly against
the real, complete codebase.

This matters concretely, not just as an apology: one defect described
below (`comparison`'s use of `VolumeError`) was investigated, "fixed,"
and then **fully reverted** once the real project's own pre-existing
test suite turned out to explicitly assert the original behavior as
its own intended, documented contract
(`test_public_api.py::test_documented_error_paths_raise_only_volume_error`).
The other defects described below were independently confirmed
against the real project's own pre-fix source code (shared directly,
not inferred) before being counted as genuine here.

## What the audit actually did

Every public class in all 7 submodules was exercised through direct
execution against small, deliberately hand-computable inputs -- a
3-4-5 triangle, a flat or uniformly-sloped plane, a known trapezoidal
cross-section -- so the expected result could be checked by hand, not
just "did it run without raising." Constructors, `compute()`/`generate()`
signatures, result types, and error handling (invalid coordinates,
empty input, unrecognized dispatch methods) were each verified this
way. Cross-submodule dependencies were checked by searching actual
import statements, not inferred from behavior. Public exports were
checked by attempting real imports.

## Defects found and fixed, in 2 distinct categories

### Category 1: a contract defect, with 3 separate functional consequences

`PointCloudData` (a `Protocol`) declared both `elevation_array` and
`xy_array` as `@property`. `TIN` -- the sole concrete implementation
of this protocol anywhere in the codebase, confirmed by search --
exposes both as plain methods instead. Every call site that treated
either as a property (accessing it without calling it) passed a bound
method object, not the array, straight into `np.isfinite()` or
equivalent, raising `TypeError` unconditionally.

This one contract defect surfaced independently, in 3 separate call
sites, each its own reproducible failure -- **confirmed directly
against the real project's own pre-fix source**, not merely this
audit's own sandbox copy:

1. `ElevationStatistics.from_tin(tin)` -- `tin.elevation_array` (no
   call).
2. `StatisticsAnalysis(method="elevation").compute(tin)` -- the same
   defect, in a second, separate, duck-typed dispatch to the same
   underlying logic.
3. `DensityStatistics.compute_from_tin(tin)` -- `tin.xy_array` (no
   call), the same underlying cause via the sibling protocol member;
   found only once the audit deliberately searched for every consumer
   of both `elevation_array` and `xy_array`, rather than stopping
   after fixing the first 2.

**Fixed at the source**: `PointCloudData` now types both
`elevation_array` and `xy_array` as methods, matching `TIN`'s own
real, working, tested behavior. `TIN` itself was never changed --
it was already correct; the Protocol was declaring something false
about its own sole implementation. All 3 call sites above now call
the corrected method directly, with no `# type: ignore` needed for
this reason anymore. See [`contracts.md`](./contracts.md) for the
full account, including why `PointCloudData.array` (a third protocol
member, which `TIN` does not implement at all) was deliberately left
unchanged.

### Category 2: an independent numeric defect, plus a related API limitation

`SlopeStatistics.compute_from_array()` -- near-degenerate histogram
range.

A genuinely uniform slope -- computed via real floating-point
arithmetic (`np.gradient` -> `np.hypot` -> `np.arctan` ->
`np.degrees`), not constructed artificially with `np.full()` -- does
not produce bit-identical values across the grid. It differs by an
amount as small as `~1e-15`. `np.histogram()` only self-widens a
degenerate range when the minimum and maximum are **exactly** equal;
this tiny but nonzero spread instead crashed with `"Cannot create N
finite-sized bins"`. **Confirmed directly against the real project's
own pre-fix source**: `_build_statistics()` called
`np.histogram(values, bins=self._num_bins)` with no range argument
and no degenerate-range handling of any kind -- the crash was
genuinely reachable, not a sandbox artifact. The fix detects a
near-degenerate range using a *relative* tolerance (slopes span very
different magnitudes, so an absolute epsilon would be wrong) and
widens the histogram's own internal binning range explicitly -- the
reported `minimum`/`maximum` statistics themselves are never altered,
only the histogram's own bin edges. See
[`statistics.md`](./statistics.md) for the full numeric account.

**This class's own sibling, `DistributionStatistics`, was not fixed
and was not confirmed to have the same defect** -- it handles an
*exactly* uniform input correctly (relying on `np.histogram()`'s own
automatic behavior for the exact-equality case), but whether a
genuinely computed, near-uniform distribution would trigger the same
crash there was not tested. Treat this as an open question, not a
cleared one -- see [`limitations.md`](./limitations.md).

**A related, separate API limitation was found and fixed in the same
audit**: `StatisticsAnalysis.slope()` (the manager's own dedicated
method) accepted only a `GriddedSurface`, unlike `SlopeStatistics`
itself, which also supports raw elevation arrays via
`compute_from_array()`. **Confirmed directly against the real
project's own pre-fix source**: `slope()`'s own signature was
`(self, dtm: GriddedSurface, num_bins=None)`, with no array-input
branch at all -- passing a raw array raised an uninformative
`AttributeError`, not a domain exception. Fixed by mirroring the same
`isinstance`-branching pattern the manager's own `elevation()` method
already used, adding a `resolution` parameter required only for the
array path.

## Investigated and reverted: `comparison`'s use of `VolumeError`

`SurfaceComparison` and `TINComparison` raise `VolumeError` for their
own validation failures, including via `comparison/surface.py`'s own
reuse of `_shared.volume.validate_volume_arrays()`. This was
investigated as a candidate defect during this audit -- `VolumeError`'s
own docstring ("Raised when a volume calculation fails") and
`_shared.volume`'s own documented list of intended consumers (grid
volume, cut/fill analysis, terrain volume methods) don't explicitly
name surface comparison, which looked at the time like an unintended
leak from `volume`'s own domain. A fix was implemented (a new
`ComparisonError`, threaded through both classes and a
`validate_volume_arrays()` parameter).

**That fix was reverted in full once the real project's own test
suite became available.**
`tests/analysis/comparison/test_public_api.py` includes
`test_documented_error_paths_raise_only_volume_error` -- a test whose
own name states plainly that `VolumeError` is this API's documented,
intended contract. Running the fix against the real project's own
complete test suite produced 23 failures, all in
`tests/analysis/comparison/`, all of the same shape: an existing,
passing test expected `VolumeError` and got the new exception
instead. The fix was reverted file-by-file back to its original
state, re-verified with real execution, `ruff`, `mypy`, and the full
project test suite.

**The methodological lesson, recorded deliberately**: absence of a
counter-example in an incomplete or sandboxed test copy is not
evidence of its absence in the real project, and a docstring's
silence on a consumer is much weaker evidence than an existing,
explicitly-named test. See [`comparison.md`](./comparison.md) and
[`contracts.md`](./contracts.md) for the full account of both the
investigation and the reversal.

## Not a defect: `DistanceAnalysis`'s own argument order

Investigated as a candidate defect, resolved as intentional: see
[`contracts.md`](./contracts.md). The dispatcher's own 6-argument
order genuinely differs from `EuclideanDistance`'s own, but this is
the correct, deliberate fix for an earlier, separate, already-fixed
defect (silently wrong results from unreordered argument forwarding).

## Not a defect: `comparison` has no dispatcher class

Investigated with the same rigor as the `VolumeError` question above,
resolved oppositely -- as an intentional asymmetry, not a defect. See
[`contracts.md`](./contracts.md) for the full evidence (the 3
classes' own signatures are genuinely incompatible with a shared
dispatcher).

## Regression tests added

12 tests, in
`tests/analysis/statistics/test_elevation_and_slope_regression.py` --
covering the confirmed defects above:

1. `TIN` satisfies `PointCloudData`'s own `elevation_array`/`xy_array`
   members correctly, confirmed via `mypy` -- this only became true
   after the contract fix (before it, `mypy` rejected passing a `TIN`
   anywhere either was declared this way).
2. `ElevationStatistics.from_tin()` no longer raises, and gives the
   correct statistics for a known 3-vertex TIN.
3. `StatisticsAnalysis(method="elevation").compute()` no longer
   raises, via the same known TIN.
4. The 2 call paths above agree exactly with each other.
5. `DensityStatistics.compute_from_tin()` no longer raises, for the
   same known TIN.
6. `SlopeStatistics` on a real, computed near-uniform ramp no longer
   raises, and correctly places every value in a single histogram bin.
7. `SlopeStatistics` on an exactly-uniform array (the case that
   already worked before this fix) still works identically --
   confirming no regression on the previously-working path.
8. `SlopeStatistics` on genuinely non-uniform terrain produces exactly
   the same statistics as before the fix (checked against specific,
   recorded values from a fixed random seed) -- confirming the fix
   changed nothing for the non-degenerate case.
9. `StatisticsAnalysis.slope()` accepts a raw array when `resolution`
   is supplied.
10. `StatisticsAnalysis.slope()` on a raw array without `resolution`
    raises `StatisticsError`, not `AttributeError`.
11. `StatisticsAnalysis.slope()` via `.compute()` dispatch matches
    calling `.slope()` directly.
12. `StatisticsAnalysis.slope()`'s own `GriddedSurface` path is
    byte-for-byte unaffected by the array-support fix.

No regression tests were added for `comparison` -- the fix
investigated there was reverted, not kept, and the real project's own
pre-existing `tests/analysis/comparison/` suite already covers this
area more completely than anything this audit would have added.

## What this does and does not prove

**These 12 tests prove the 2 confirmed defect categories above (the
`PointCloudData` contract defect and its 3 functional consequences,
and the `SlopeStatistics`/`StatisticsAnalysis.slope()` pair) are
fixed and stay fixed. They are not, and must not be read as,
comprehensive test coverage of `topocore.analysis`.** This audit's
own manual, hand-verified executions (documented across this
section's own pages) provide real, additional confidence for the
other submodules, but real, automated regression coverage for most of
`analysis` already existed in the project before this audit, and
remains the stronger source of protection against a future change.

## Verification performed on the fixes

- `ruff check` -- clean across all of `topocore/analysis/`, not just
  the files directly touched by a given fix. A separate, dedicated
  lint pass (13 findings: `RUF022`/`RUF023` ordering, `RUF046`
  redundant casts) found and resolved everything ruff flagged in the
  whole package, including 3 deliberately-grouped `__all__` lists
  preserved with a scoped, justified `# noqa: RUF022` rather than
  flattened to satisfy the linter.
- `mypy` -- **a methodological correction applies here too**: earlier
  checks in this audit used hand-picked flags
  (`--disallow-untyped-defs`, `--warn-return-any`, etc.), not this
  project's own `pyproject.toml`. Re-run with the real configuration
  (which already includes a `scipy`/`scipy.*` override and lists
  `scipy-stubs` as a dev dependency), `topocore/analysis` has **zero**
  `mypy` errors of any kind -- the 4 `scipy` `import-untyped` findings
  reported earlier in this audit never existed under the project's
  own real settings; they were an artifact of not having that
  configuration file available yet, not a defect or an unresolved
  limitation. (Running the same real configuration surfaced 27
  pre-existing errors elsewhere in the codebase -- `geodesy`,
  `processing`, `io`, `terrain`, `features` -- none in `analysis`;
  out of scope for this section, left untouched.)
- 12/12 regression tests pass.
- Full project test suite: passes with no regressions from either the
  kept fixes or the full reversion of the `comparison` one.

