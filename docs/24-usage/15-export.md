# Export

Both exporters consume a `FeatureCollection` (see
[`features.md`](./features.md)) exclusively -- neither exports a raw
`PointCloud` directly; confirmed by each package's own module
docstring, this is a deliberate scope boundary, not a gap in either
implementation.

## `DXFExporter`

```python
from topocore.dxf import DXFExporter
from topocore.dxf.models import DXFExportOptions, ExportContext

options = DXFExportOptions(
    dxf_version: str = "R2010",
    units: DrawingUnits = DrawingUnits.METERS,   # only METERS supported today
    non_planar_polygon_mode: NonPlanarPolygonMode = NonPlanarPolygonMode.POLYLINE3D,
    tolerance: DXFTolerance = DXFTolerance(),
    strict: bool = True,
    index_contour_every: int = 5,
)
context = ExportContext(crs: str | None = None, options: DXFExportOptions = options)
report = DXFExporter(context).export(collection, path)
```

Confirmed directly: `units` other than `METERS` raises `DXFExportError`
immediately at construction -- `DXFExporter` only sets the DXF file's
`$INSUNITS` header (a metadata label), it does not convert coordinate
values, so declaring any other unit would silently mislabel
unconverted meter coordinates.

`ezdxf` is an optional dependency -- `DXFExporter()` raises
`DXFExportError` with a clear install message
(`pip install topocore[dxf]`) if it is not installed, checked before
anything else runs.

## `GeoPackageExporter`

```python
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions

options = GPKGExportOptions(
    epsg: int,             # mandatory -- no default
    strict: bool = True,
)
report = GeoPackageExporter(options).export(collection, path)
```

`epsg` has no default deliberately: a GeoPackage without a real CRS
is ambiguous in QGIS/ArcGIS -- `GeoPackageExporter` raises
`GPKGExportError` rather than falling back to a placeholder SRS.
No optional dependency is required -- geometry encoding uses
`shapely` (already mandatory) and the container itself is built with
stdlib `sqlite3`.

## The `strict` contract — confirmed identical in spirit, verified independently for each exporter

```text
strict=True   -- any feature that fails validation aborts the WHOLE
                 export; no file is written at the target path at all
strict=False  -- that one feature is skipped and recorded in the
                 report; every other valid feature is still written
```

Both exporters validate every feature *before* touching disk (a
per-feature validation loop runs first; the write transaction begins
only afterward) -- this is what makes `strict=False` isolation
possible without a corrupt partial file. Confirmed directly, with a
real, generated file, for both exporters: a collection with 1 valid
feature + 1 invalid one, under `strict=False`, produces a real,
valid output file containing only the valid feature; under
`strict=True`, no file is produced at all.

Both exporters' own validators were confirmed, during this project's
own audit, to cover every condition their own write path actually
requires (non-finite coordinates, a degenerate `MESH` triangle for
GeoPackage; non-finite `elevation`/`base`/`interval` for a DXF
contour's own layer resolution) -- a real, historical gap where an
unwrapped exception from deep inside the write path could bypass
`strict=False` entirely was found and fixed in each exporter
independently during this same audit.

## Atomic write — confirmed for both

Both write to a temporary file in the same directory as the target
path first, then atomically replace the target only on success
(`os.replace`). Confirmed directly: a failed export (under
`strict=True`) leaves no file at all at the target path -- never a
partial or corrupt one. Writing to an already-existing path
overwrites it silently, with no warning -- the same behavior as
`open(path, "w")` or any other explicit "write to this path"
operation; confirmed not a defect, since neither exporter documents
or implies confirmation-before-overwrite.

## Inspecting the report

Both `export()` calls return a real, structured report -- confirmed
fields, not a generic dict:

```python
report = DXFExporter(context).export(collection, path)
report.output_path        # Path
report.dxf_version         # str
report.feature_count        # int -- total features seen
report.entity_count          # int -- DXF entities actually written
report.skipped_features       # int -- 0 unless strict=False and something failed
report.point_count             # int, per representation...
report.lwpolyline_count
report.polyline3d_count
report.face3d_count
report.layer_count
report.features_by_type         # Mapping[FeatureType, int]
report.warnings                  # tuple[str, ...]
report.warning_count              # property, len(warnings)
```

```python
report = GeoPackageExporter(options).export(collection, path)
report.output_path        # Path
report.srid                # int
report.feature_count        # int
report.written_count         # int
report.skipped_count          # int -- 0 unless strict=False and something failed
report.table_count             # int
report.features_by_table        # Mapping[str, int]
report.warnings                  # tuple[str, ...]
```

## The real table schema — every column, where each one comes from

Confirmed directly from `GeoPackageExporter`'s own source, and with a
real generated file: every feature table it writes has exactly this
schema, regardless of `FeatureType`:

```text
fid                 -- SQLite rowid, assigned 1..N in insertion order
geom                -- the GPB/WKB blob (see build_gpb() in geodesy notes above)
feature_id          -- Feature.feature_id, as given
feature_type        -- Feature.feature_type.value
category            -- Feature.category.value
survey_code         -- ONLY IF Feature.attributes has this exact key
survey_name         -- ONLY IF Feature.attributes has this exact key
cad_layer           -- ONLY IF Feature.attributes has this exact key
confidence          -- Feature.confidence (always present, defaults to 1.0)
producer            -- Feature.metadata.detector, or NULL if metadata is None
producer_version    -- Feature.metadata.version, or NULL if metadata is None
attributes_json     -- every OTHER key in Feature.attributes, as one JSON object
```

**Only 3 attribute keys ever become their own column**
(`survey_code`, `survey_name`, `cad_layer`) -- confirmed directly,
`_PROMOTED_ATTRIBUTE_KEYS` is a fixed, hardcoded 3-item tuple, not
inferred from what a given `FeatureCollection` happens to contain.
Every other key you put in `Feature.attributes` -- `classification`,
`intensity`, or anything else -- lands inside `attributes_json` as a
single JSON text blob, not as its own queryable SQL column.

**This is a deliberate, reasonable design for `FeatureCollection`'s
own real use case** (a collection mixing buildings, trees, walls,
contours -- each with a genuinely different, unpredictable attribute
set): a fixed schema of 1 column per possible attribute would mean a
huge table full of columns that are `NULL` for every feature type
that doesn't use them. `attributes_json` avoids that without losing
any data.

**It becomes a real limitation for uniform, per-point data**
(classification/intensity/color on every point of an exported point
cloud, for example): a GIS user opening the file in QGIS cannot
filter or symbolize directly by `classification` -- it's confirmed
still there, correctly preserved, but only reachable via a JSON
expression (`json_extract(attributes_json, '$.classification')`) or
by parsing it in Python/SQL yourself, not as a native column. Turning
those into real, native columns would require a different, dedicated
writer -- not something `GeoPackageExporter` does today, and not
something built by simply wrapping it in a convenience function.

## Worked example — exporting many similar points (e.g. from a point cloud)

Confirmed real and runnable, using only the APIs documented above and
in [`features.md`](./features.md) -- but with 2 honest caveats
spelled out, since neither has a single correct answer TopoCore can
decide for you:

```python
from topocore.features.models import (
    Feature, FeatureCategory, FeatureCollection, FeatureGeometry, FeatureType, GeometryType,
)
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions

collection = FeatureCollection()
collection.crs = "EPSG:32618"

for i in range(len(xs)):
    geom = FeatureGeometry(geometry_type=GeometryType.POINT, vertices=[[xs[i], ys[i], zs[i]]])
    feature = Feature(
        feature_id=i,
        category=...,       # no single correct value exists for an
        feature_type=...,   # undifferentiated raw point -- see below
        geometry=geom,
        attributes={"classification": cls[i], "intensity": intensity[i]},  # -> attributes_json, not native columns
    )
    collection.add(feature)

options = GPKGExportOptions(epsg=32618)
GeoPackageExporter(options).export(collection, "salida.gpkg")
```

**Caveat 1 -- `category`/`feature_type`**: none of the 85 real
`FeatureType` values (see [`features.md`](./features.md)) has an
obvious semantic fit for "an undifferentiated raw point with no
detected meaning" -- they're all designed for something with real
significance (a tree, a control point, a building). Picking one
arbitrarily (e.g. `REFERENCE_POINT`/`CONTROL`) directly becomes the
real table name in the output file (confirmed: `CONTROL`+
`REFERENCE_POINT` produces a table literally named `control_point`),
which would mislead anyone opening the file in QGIS if the points
aren't actually control points. There is no default this guide can
responsibly recommend -- choose based on what the points actually
represent in your own workflow, or reconsider whether `GeoPackage`
via `Feature` is the right destination for genuinely undifferentiated
points at all.

**Caveat 2 -- extra attributes**: as documented above, anything you
put in `attributes` beyond `survey_code`/`survey_name`/`cad_layer`
lands in `attributes_json`, not a native column -- confirmed by
directly inspecting a real generated file's own schema. Correct and
lossless, but not directly filterable/symbolizable as a column in
QGIS without a JSON expression.

For thousands of points this runs in a few seconds with a modest
memory footprint; for hundreds of thousands or more, consider whether
the resulting per-feature-object memory cost and JSON-only attribute
storage still fit your use case, or whether the points belong in
[`point-clouds.md`](./point-clouds.md)'s own LAS/LAZ writers instead.

## Calling both on the same collection

```python
report_dxf = DXFExporter(context).export(collection, "output.dxf")
report_gpkg = GeoPackageExporter(options).export(collection, "output.gpkg")
```

Confirmed safe: neither exporter mutates the `FeatureCollection` it
is given, and `Workflow.export_dxf()`/`.export_gpkg()` (see
[`workflows.md`](./workflows.md)) are both documented to never write
to the artifact store -- either may be called any number of times,
including on the same collection.

## Where to go next

- [`workflows.md`](./workflows.md) -- `export_dxf()`/`export_gpkg()`'s
  own `Workflow`-level contract, including automatic EPSG resolution
  from a collection's own `crs`.
- [`features.md`](./features.md) -- building the `FeatureCollection`
  both exporters consume.
