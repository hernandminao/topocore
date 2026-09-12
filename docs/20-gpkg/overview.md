# `topocore.gpkg` — Overview

## Scope, confirmed directly from the package's own source

`topocore.gpkg` exports a `FeatureCollection` (the output of
`topocore.features`, just closed under `18-features`) to a real,
OGC-compliant GeoPackage (`.gpkg`) file. **It does not export a raw
`PointCloud`** -- confirmed directly, by the package's own module
docstring: "PR17b (raw PointCloud -> GeoPackage feature table, with
classification/intensity/RGB) is explicitly out of scope here." A
`PointCloud -> GeoPackage` exporter does not exist anywhere in this
codebase; building one is future work, not something this audit
found broken or incomplete within an existing implementation.
Recorded here plainly so this scope boundary is never mistaken for a
gap in the `FeatureCollection` exporter itself.

## Real inventory — 10 files, 1161 lines

```text
topocore/gpkg/
├── __init__.py       exports GeoPackageExporter, GPKGExportOptions,
│                       GPKGExportReport, and the 4 exceptions
├── config.py           GPKGExportOptions (epsg mandatory, strict flag)
├── exceptions.py         GPKGError -> GPKGExportError/GPKGGeometryError/
│                          GPKGValidationError
├── exporter.py             GeoPackageExporter -- the orchestrator
├── geometry.py               FeatureGeometry -> GeoPackage Binary (GPB)
├── metadata.py                 table naming, gpkg_contents/gpkg_geometry_columns rows
├── report.py                    GPKGExportReport / _ReportBuilder
├── schema.py                     DDL for the 4 mandatory system tables + per-category tables
├── spatial_ref.py                  EPSG -> gpkg_spatial_ref_sys, via pyproj
└── validation.py                    GPKGValidator -- pre-write, per-feature checks
```

No GDAL, fiona, or pyogrio dependency -- confirmed directly: geometry
encoding uses `shapely`'s own WKB output (already a mandatory project
dependency) wrapped in a hand-built GPB header; CRS resolution uses
`pyproj` (also already mandatory); the GeoPackage container itself is
built with stdlib `sqlite3`.

## Architecture -- export flow, confirmed by direct execution at every stage

```text
FeatureCollection
      |
      v
for each Feature:
   GPKGValidator.validate()  -- per-feature, before any disk write
      |
      +-- issues (ERROR severity) --> strict=True: abort whole export
      |                               strict=False: skip, record in report, continue
      v
group by (category, geometry family)
      |
      v
write to a temp file (same directory as the target path)
      |
      +-- system tables (gpkg_spatial_ref_sys, gpkg_contents,
      |    gpkg_geometry_columns, gpkg_extensions)
      +-- SRS rows (target EPSG + WGS84 courtesy row + OGC placeholders)
      +-- one feature table + R*Tree index per (category, family) group
      |
      v
os.replace(temp, final_path)  -- atomic; a failed write never leaves
                                  a corrupt .gpkg at the target path
```

**Confirmed directly, by reading `exporter.py`'s own code**: validation
happens entirely before any file is touched on disk (`export()`'s own
per-feature loop runs first; the temp file is only created afterward).
This is what makes the `strict=False` isolation contract meaningful --
a feature rejected at this stage never reaches the write transaction
at all.

## The 1 real defect found and fixed -- full account in `validation.md`

`GPKGValidator.validate()`'s own checks used to be a strict subset of
what `gpkg.geometry._to_shapely()` (called later, inside the write
transaction) actually required -- missing a finite-coordinate check
and a degenerate-MESH-triangle check. A `Feature` that only
`_to_shapely()` would reject reached the write transaction, where any
`GPKGGeometryError` aborts the WHOLE export regardless of
`GPKGExportOptions.strict` -- confirmed directly, with a real,
reproducible case: a collection containing one genuinely valid
feature and one with a degenerate MESH face produced **no `.gpkg`
file at all** under `strict=False`, silently violating that option's
own documented contract ("that feature is skipped ... instead").
Fixed by extending `GPKGValidator.validate()` with 2 new checks
(`GPKG006`, `GPKG007`) mirroring `_to_shapely()`'s own checks exactly,
so nothing reaches the write transaction that wasn't already accepted
during the earlier, `strict`-aware validation loop. Confirmed fixed
with a real, queried `.gpkg` file, not merely a caught exception --
see [`validation.md`](./validation.md).

## Investigated and confirmed correct, not defects

- **Empty `FeatureCollection`**: produces a valid, minimal `.gpkg`
  (system tables only, 0 feature tables) -- confirmed directly, not
  an error condition.
- **Invalid or placeholder EPSG** (including the GeoPackage spec's
  own `-1`/`0` "undefined" SRS ids): rejected cleanly with
  `GPKGExportError`, wrapping `pyproj`'s own `CRSError` where
  relevant -- confirmed no bare `pyproj` exception escapes.
  `GPKGExportOptions.epsg` has no default specifically to force this:
  "A GeoPackage without a real CRS is ambiguous in QGIS/ArcGIS and
  hostile to interoperability" (the option's own docstring).
- **Out-of-range MESH face indices**: confirmed protected even earlier
  than `gpkg` itself -- `topocore.features.models.FeatureGeometry`'s
  own construction-time validation already rejects this, before a
  `Feature` carrying one could ever reach `gpkg` at all.
- **Unclosed POLYGON geometry**: rejected cleanly, `GPKGValidationError`
  (`GPKG001`), confirmed via the existing, pre-audit validation path.
- **Atomic write**: confirmed directly -- a successful export leaves
  exactly the target file, no orphaned temp file; an export that
  fails partway through (an invalid feature under `strict=True`)
  leaves no file of any kind at the target path.
- **Writing to an already-existing path**: confirmed to overwrite
  silently, with no warning -- the same behavior as `open(path, "w")`
  or virtually any explicit "write to this path" operation. Not
  classified as a defect: no documentation or test anywhere in this
  project establishes an expectation of confirmation-before-overwrite,
  and an explicit destination path argument is ordinarily understood
  to mean "write here," not "write here only if nothing is there."

## Where to go next

- [`validation.md`](./validation.md) -- the 1 fixed defect, in full,
  with the same evidence chain used throughout this project.
- [`limitations.md`](./limitations.md) -- pre-existing, out-of-scope
  findings only; the `strict=False` defect is not repeated there,
  since it is fixed and verified, not outstanding.
