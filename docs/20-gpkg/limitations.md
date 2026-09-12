# Limitations

## `PointCloud -> GeoPackage` does not exist -- future work, not a gap in what exists

Confirmed directly, by the package's own module docstring and by
search across its complete source: `topocore.gpkg` exports
`FeatureCollection` only. Raw `PointCloud` export (with per-point
classification/intensity/RGB, referred to in the package's own
docstring as "PR17b") has zero implementation anywhere in this
codebase. This is not recorded as a defect or an incomplete
implementation -- there is nothing existing to be incomplete -- but
as a clearly-flagged scope boundary for whoever picks up that
future work, so it is never mistaken for something this audit found
broken. See [`overview.md`](./overview.md).

## Confirmed pre-existing, out of this audit's own 1-defect scope

- **1 `mypy` finding** (real configuration, `gpkg.geometry`'s own
  `build_gpb()`, `no-any-return` on `return header + wkb`): confirmed
  the same category of finding already documented multiple times
  elsewhere in this project (`16-terrain`, `18-features`) -- a
  third-party library's own imprecise type stubs (here, `shapely.wkb.dumps()`'s
  own return type) causing a locally-`Any`-typed value to flow into a
  function with a precise return annotation. Not a runtime defect;
  confirmed unrelated to this audit's own fix, which touched only
  `validation.py`. Not corrected here, consistent with this audit's
  own scope.
- **0 `ruff` findings**: confirmed directly, `ruff` (real
  `pyproject.toml`) reports nothing anywhere in `topocore/gpkg/`,
  including in the file this audit modified.

## Writing to an already-existing path overwrites it silently

Confirmed directly (see [`overview.md`](./overview.md)): calling
`GeoPackageExporter.export()` with a `path` that already exists
overwrites it with no warning or confirmation step, the same
behavior as `open(path, "w")` or virtually any explicit
"write-to-this-path" operation elsewhere in this codebase or in
Python's own standard library conventions. **Not classified as a
limitation or a defect** -- recorded here only so it is not
mistakenly rediscovered as a new finding in a future audit pass:
this was investigated, and no documentation, test, or convention
anywhere in this project establishes an expectation of
confirmation-before-overwrite for this or any other exporter
(`dxf`, for instance, follows the identical pattern).

## No test suite was available for this package

Confirmed directly, not assumed from an empty search: no
`tests/gpkg/` (or equivalent) directory was present in the
reconstructed repository export this audit worked from, and none was
otherwise provided -- the same situation already noted for
`18-features`. The 6 regression tests this audit added (see
[`validation.md`](./validation.md)) are, as far as this audit could
confirm, the first tests this package has had.

## What this audit did not attempt

A byte-level comparison of this project's own generated `.gpkg`
files against a GDAL/OGR-produced reference GeoPackage was not
performed -- this audit confirmed correctness by direct SQL
inspection of the real output (`PRAGMA application_id`, table
contents, row counts) and by reading the schema/geometry-encoding
code directly against the OGC GeoPackage specification's own
documented requirements, not by cross-validating against an external
tool. No evidence of an incompatibility was found during this
audit's own direct inspection, but this is a narrower form of
verification than an external-tool round-trip test would provide.
