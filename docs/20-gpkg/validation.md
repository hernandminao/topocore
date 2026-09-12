# Validation — 1 defect found and fixed

## Methodology, identical to `16-analysis`/`16-terrain`/`18-features`

```text
finding -> evidence -> classification -> fix -> behavior test -> regression
```

---

## `strict=False` did not isolate an invalid feature -- it aborted the whole export

**Finding**: `GPKGValidator.validate()`'s own checks were a strict
subset of what `gpkg.geometry._to_shapely()` (called later, inside
the write transaction) actually required.

**Evidence**: confirmed directly, by constructing a `Feature` with a
MESH face whose 3 vertices are exactly collinear (a real, zero
cross-product degenerate triangle -- not merely a synthetic corner
case): `topocore.features.models.FeatureGeometry`'s own
construction-time validation allows it (it checks vertex shape and
finiteness, not that a face forms a genuine triangle), and
`GPKGValidator.validate()` also reported zero issues for it. The
identical geometry, run through `GeoPackageExporter.export()` with
`strict=False`, raised `GPKGExportError` and produced **no `.gpkg`
file at all** -- confirmed by checking `os.path.exists()` directly --
even though the same collection also contained a genuinely valid
feature. This directly contradicts `GPKGExportOptions.strict`'s own
documented contract: "If False, that feature is skipped and recorded
in the report instead." Root cause confirmed by reading
`exporter.py`'s own control flow: the per-feature, `strict`-aware
skip-or-abort logic lives entirely in `export()`'s own validation
loop, which runs *before* the write transaction; `_write()`'s own
`except (sqlite3.Error, GPKGGeometryError)` block, guarding the
transaction itself, treats any `GPKGGeometryError` reaching it as
fatal to the whole export, with no `strict` check at all -- correct
for a genuinely unanticipated failure, but wrong for a condition
`GPKGValidator` should have already screened out earlier.

**Classification**: a real, confirmed defect -- a documented public
contract (`GPKGExportOptions.strict`'s own docstring) violated by a
concrete, reproducible case, not a hypothetical one.

**Fix**: `GPKGValidator.validate()` now includes 2 additional checks,
mirroring `_to_shapely()`'s own checks exactly, so the validator's
own coverage matches what the write path actually requires:

- `GPKG006` -- non-finite (NaN/Inf) vertex coordinates, checked for
  every geometry type (matching `_to_shapely()`'s own
  `_validate_vertices()`).
- `GPKG007` -- a degenerate (zero cross-product) MESH triangular
  face, checked only when the face indices are already confirmed
  in-range (mirroring `_to_shapely()`'s own per-face loop exactly,
  including its own cross-product test).

No change was made to `build_gpb()`, `_to_shapely()`, `strict`'s own
semantics, `GPKGExportReport`, or any public API -- the fix is
confined entirely to `validation.py`, adding checks that duplicate
(deliberately, for this specific pair of conditions) logic that
already existed in `geometry.py`, rather than refactoring either
module to share it. This was a deliberate choice, not an oversight:
sharing the logic would have meant modifying `_to_shapely()` itself,
which this audit's own agreed scope explicitly avoided unless
proven indispensable -- confirmed here that simple, narrowly-scoped
duplication of 2 self-contained checks was sufficient and lower-risk.

**Behavior test**: confirmed directly, with a real, queried `.gpkg`
file, not merely a caught exception:

- A collection with 1 valid feature + 1 degenerate-MESH feature,
  under `strict=False`: the resulting `.gpkg` file now exists, is a
  genuine, valid GeoPackage (`PRAGMA application_id` confirmed to be
  the GPKG magic value), and its own feature table contains exactly
  1 row -- the valid feature. The report confirms `written_count=1`,
  `skipped_count=1`, with a warning naming the degenerate-face issue.
- The identical collection under `strict=True`: still aborts the
  whole export with `GPKGValidationError` (now reporting `GPKG007`
  specifically), and no file is left at the target path -- confirmed
  `strict=True`'s own contract is unchanged by this fix.
- A fully valid collection: confirmed to export identically to
  before the fix (`written_count` matches, `skipped_count=0`).
- `GPKGValidator.validate()` on a genuinely valid MESH feature:
  confirmed to still return zero issues -- no false positive
  introduced.

**Regression** -- the 6 previously-correct behaviors this audit
specifically re-checked, confirmed unaffected by this fix:

- Empty `FeatureCollection` -- still produces a valid, minimal
  `.gpkg` (0 feature tables).
- Invalid EPSG (`999999`) -- still raises `GPKGExportError` cleanly.
- Unclosed POLYGON -- still raises `GPKGValidationError` (`GPKG001`).
- Out-of-range MESH face indices -- still rejected, confirmed even
  earlier than `gpkg` itself, at `FeatureGeometry`'s own construction.
- Atomic write -- confirmed no orphaned temp file after a successful
  export.
- Writing to an already-existing path -- confirmed to still overwrite
  silently and produce a valid file, unchanged.

---

## Consolidated regression

- `ruff`, with the real project's own `pyproject.toml`
  (auto-discovered from the package root): clean across the entire
  `topocore/gpkg/` package, not only `validation.py`.
- `mypy`, same real configuration, the 10 real `gpkg` source files
  checked together with `--follow-imports=silent`: clean --
  `validation.py` itself introduces no new finding.
- 6 new regression tests, covering the fixed defect from every angle
  described above (validator-level detection for both new checks, a
  no-false-positive check, the real `strict=False`/`strict=True`
  end-to-end behavior with a real queried `.gpkg`, and a fully valid
  collection) -- all pass.
- The full available test suite: 321 of 321 pass (315 before this
  fix, plus these 6).

## What this fix does not cover

No test file for `topocore.gpkg` existed anywhere available to this
audit, the same situation already noted for `18-features` -- the 6
tests added here are, as far as this audit could confirm, the first
tests this package has had in this audit's own working environment.
They were written specifically to prove this 1 defect stays fixed;
the extensive direct execution documented in
[`overview.md`](./overview.md) (the full export flow, CRS resolution,
atomic write, all 4 geometry families) provides real, additional
confidence for the rest of this package, but is not a substitute for
a maintained test suite.
