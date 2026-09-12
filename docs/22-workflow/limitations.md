# Limitations

## No functional defects, no code changes -- stated plainly

This audit found 0 functional defects in `topocore.workflow` and
made 0 changes to its source. This is not an abbreviated audit: the
same 5-point methodology used for `18-features`/`19-gpkg`/`20-dxf`
was applied in full, including real-execution reproductions of the
highest-risk scenarios (transitive staleness, a malicious
`ProgressObserver`, a deliberately-failed stage, consecutive
in-place transforms, EPSG conflict resolution) -- see
[`validation.md`](./validation.md) for the complete account of what
was executed, not merely read.

## 6 historical defects -- already fixed, not current limitations

This package's own source documents 6 distinct defects that were
found and fixed in prior work (PR19, PR20, and the design review for
`transform_crs()`/`georeference()`), each confirmed during this
audit's own real-execution testing to still be correctly fixed:
`read_point_cloud()`'s format dispatch, `export_dxf()`/`export_gpkg()`'s
own kwargs handling, `GroundManager.extract()`'s CRS propagation,
`detect_features()`'s own false-positive staleness check, and the 2
"never silently double-transform an already-referenced artifact"
safety checks in `transform_crs()`/`georeference()`. These are listed
in full in [`overview.md`](./overview.md) as historical context and
evidence of this package's own hardening over time -- they are
deliberately NOT repeated here as current limitations, since nothing
about them is outstanding.

## No test suite was available for this package

Confirmed directly: no `tests/workflow/` (or equivalent) directory
was present in the reconstructed repository export this audit worked
from, and none was otherwise provided -- the same situation already
noted for `18-features`/`19-gpkg`/`20-dxf`. Unlike those 3 blocks,
this audit did not add a regression test file here either, since no
defect was found to regress-test against; per this project's own
established discipline, tests are added to prove a specific fix
stays fixed, not manufactured to pad coverage for behavior that
was directly verified and found correct. This means `topocore.workflow`
remains, as far as this audit could confirm, without any automated
test coverage of its own -- a real gap, but a pre-existing one this
audit did not introduce and had no fix to attach a test to.

## `mypy` findings transitively reachable from `workflow/` are not this package's own debt

A plain `mypy` run against `topocore/workflow/`'s own 8 files (with
the real project's own `pyproject.toml`, default import-following)
reports 32 findings -- all of them in `topocore/processing/classification/*`
(`random_forest.py`, `gradient_boost.py`, `manager.py`), reached only
because `mypy` follows imports transitively by default. Confirmed,
with an isolated `--follow-imports=silent` run against `workflow/`'s
own 8 files alone: 0 findings. These 32 are pre-existing debt in
`processing`, already known from that package's own earlier audit
work, not attributable to `workflow` and not corrected here, since
doing so is outside this block's own scope.

## `ruff` findings

None. A full `ruff` run against `topocore/workflow/`, with the real
project's own `pyproject.toml`, reports 0 findings -- unlike every
other block audited in this PR22 pass so far (`16-analysis` through
`20-dxf`), which each had some number of pre-existing, deliberately
unaddressed `RUF022`/`RUF023`-style findings. `workflow/`'s own
`__all__` lists and class attributes are confirmed already
alphabetically sorted throughout, so there was nothing to document
here as a "deliberate grouping, not flattened" exception either.

## What this audit did not attempt

A live, end-to-end run of the complete 16-stage pipeline against a
real, large-scale production LAS/LAZ file (rather than the small,
synthetic point clouds and hand-built artifacts used throughout this
audit's own real-execution testing) was not performed -- this audit's
own reproductions were sized to exercise the specific mechanism under
test (staleness, recovery, observer isolation, EPSG resolution)
directly and verifiably, not to simulate a realistic production
workload's own scale or timing characteristics. No evidence of a
scale-dependent defect was found or suspected during this audit.
