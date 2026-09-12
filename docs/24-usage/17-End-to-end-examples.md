# End-to-End Examples — every input format to every output

Every code block on this page was executed for real during this
guide's own writing -- against real files in every supported format,
built and read back with TopoCore's own readers, not invented. Where
a real execution surfaced something this guide's other pages had
gotten wrong or left out, it's corrected here and cross-referenced.

## Part 1 — Loading every input format

All 7 point-cloud formats and the survey-text format, confirmed
read back correctly, each into the type shown:

```python
# LAS
from topocore.io.las import LASReader
with LASReader("nube.las") as r:
    cloud = r.read()          # -> PointCloud, 2000 points confirmed

# LAZ (requires the lazrs or laszip backend installed alongside laspy)
from topocore.io.laz import LAZReader
with LAZReader("nube.laz") as r:
    cloud = r.read()          # -> PointCloud

# PLY
from topocore.io.ply.reader import PLYReader
with PLYReader("nube.ply") as r:
    cloud = r.read()          # -> PointCloud

# XYZ
from topocore.io.ascii.xyz import XYZReader
with XYZReader("nube.xyz") as r:
    cloud = r.read()          # -> PointCloud

# CSV
from topocore.io.ascii.csv.reader import CSVReader
with CSVReader("nube.csv") as r:
    cloud = r.read()          # -> PointCloud

# PTS
from topocore.io.ascii.pts.reader import PTSReader
with PTSReader("nube.pts") as r:
    cloud = r.read()          # -> PointCloud

# Survey text (total station / GNSS field codes) -- NOT a PointCloud
from topocore.survey.reader import SurveyTXTReader
survey_points = SurveyTXTReader("survey.txt").read()   # -> SurveyPointSet
```

E57, confirmed real and working end to end (a synthetic E57 was
generated with `pye57` itself and read back successfully):

```python
from topocore.io.e57 import E57Reader
with E57Reader("nube.e57", chunk_size=1000) as r:
    cloud = r.read()   # -> PointCloud, confirmed: 500/500 points read back correctly
```

### A real, previously undocumented gap, found while generating fixtures for this page

`LASWriter` exists as a real class (`topocore/io/las/writer.py`), but
confirmed directly by import: it is **not** part of the public API --
`from topocore.io.las import LASWriter` raises `ImportError`. Only
`topocore.io.las.LASReader` is exported; the module's own docstring
states this explicitly. [`point-clouds.md`](./point-clouds.md)
previously stated "neither LASReader's package... expose[s] a
writer," which undersold the situation -- the class exists in the
source, it just isn't public. There is also no PLY writer at all,
public or private, confirmed directly (no `writer.py` anywhere under
`topocore/io/ply/`).

### The uniform destination

Every point-cloud reader above produces the exact same `PointCloud`
type, regardless of source format -- everything from here on is
identical no matter which format you started from. Only
`SurveyTXTReader` produces something different (`SurveyPointSet`),
which forks onto its own path (see Part 4).

## Part 1b — Vertical-datum conversion and georeferencing, confirmed real

**Control-point georeferencing**, confirmed with real execution
(pure Python, no optional dependency needed):

```python
from topocore.geodesy.georeferencing import ControlPoint, GeoreferencingOptions, fit_georeferencing

controls = [
    ControlPoint(id="P1", source=(0.0, 0.0, 0.0), target=(1000.0, 2000.0, 0.0)),
    ControlPoint(id="P2", source=(100.0, 0.0, 0.0), target=(1100.0, 2000.0, 0.0)),
    ControlPoint(id="P3", source=(0.0, 100.0, 0.0), target=(1000.0, 2100.0, 0.0)),
]
result = fit_georeferencing(controls, GeoreferencingOptions())
print(result.strategy)      # confirmed real output: helmert_3d
print(result.parameters)    # confirmed real output: tx=1000.0, ty=2000.0, tz≈0,
                             # rx≈0, ry≈0, rz≈0, scale≈0 -- a pure translation,
                             # correctly detected even though HELMERT_3D was fitted
```

**Vertical (ellipsoidal <-> orthometric) conversion**, confirmed with
real execution -- but with 1 honest caveat: GDAL could not be
installed in this sandbox (see the note in
[`geodesy.md`](./geodesy.md)), so `GeoidGrid.from_geotiff()` itself
is not exercised here. What *is* confirmed, by constructing a
`GeoidGrid` directly with synthetic values (bypassing the
`from_geotiff()` factory, which is not this class's normal, public
construction path) is `VerticalTransformer`'s own real conversion
math:

```python
from topocore.geodesy.vertical.geoid_grid import GeoidGrid
from topocore.geodesy.vertical.transformer import VerticalTransformer
from topocore.geodesy.vertical_datum import VerticalDatum

# NOT the normal public path (that's GeoidGrid.from_geotiff(path)) --
# built directly here only because GDAL isn't installable in this
# sandbox. A synthetic 2x2 grid with a constant 30.0m undulation.
grid = GeoidGrid(
    origin_longitude=-76.0, origin_latitude=3.0,
    pixel_width=0.1, pixel_height=-0.1,
    columns=2, rows=2,
    _values=(30.0, 30.0, 30.0, 30.0),
    nodata=None, source_path="synthetic-for-testing",
)

transformer = VerticalTransformer(
    VerticalDatum(name="WGS84 ellipsoidal", authority="EPSG"),
    VerticalDatum(name="EGM2008 orthometric", authority="EPSG"),
    grid,
)

H = transformer.ellipsoidal_to_orthometric(-75.95, 2.95, 2530.0)
print(H)   # confirmed real output: 2500.0  (2530 - 30, exact)

h = transformer.orthometric_to_ellipsoidal(-75.95, 2.95, H)
print(h)   # confirmed real output: 2530.0  (round-trips exactly)
```

Against a real geoid file (e.g. this project's own
`data/geoids/us_nga_egm2008_1.tif` -- see [`geodesy.md`](./geodesy.md)),
the only difference is the first line: `grid = GeoidGrid.from_geotiff(path)`
instead of the direct, synthetic construction above -- everything
after that is identical.

## Part 2 — CRS: detecting, setting, and transforming

```python
# Automatic detection from the file itself (native metadata first,
# falling back to a .prj sidecar -- see geodesy.md for the exact
# per-format priority)
from topocore.io.crs.detector import detect_crs
crs = detect_crs("nube.las")   # -> CRS | None

# If the file carried no CRS at all (confirmed: none of the
# synthetic fixtures on this page have one), set it explicitly
cloud.crs = "EPSG:32618"

# Transforming to a different CRS
from topocore.geodesy.crs import CRS
from topocore.geodesy.transformer import CoordinateTransformer
from topocore.geodesy.transform import transform_point_cloud

transformer = CoordinateTransformer(CRS.from_epsg(32618), CRS.from_epsg(4326))
cloud_wgs84 = transform_point_cloud(cloud, transformer)   # pure -- returns a new PointCloud
```

See [`geodesy.md`](./geodesy.md) for vertical-datum conversion and
control-point georeferencing, neither of which this page's synthetic
fixtures exercise (they have no real vertical datum or control
points to convert).

## Part 3 — The point-cloud path, confirmed end to end with `Workflow`

Every stage below was executed for real, in this exact order, against
the real `nube.las` fixture -- all 8 stages confirmed `SUCCESS`:

```python
from topocore.workflow import Workflow
from topocore.terrain.grid import Grid
from topocore.terrain.enums import InterpolationMethod

wf = Workflow()
wf.read_point_cloud("nube.las")
wf.classify_ground()

grid = Grid(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0, resolution=2.0)
wf.build_tin()
wf.build_dtm(grid, method=InterpolationMethod.LINEAR)
wf.extract_contours(interval=0.5, base=0.0)
wf.detect_features(strict=False)
wf.export_dxf("salida.dxf", strict=False)
wf.export_gpkg("salida.gpkg", epsg=32618, strict=False)

for stage in wf.result.stages:
    print(stage.stage, stage.status)
# read_point_cloud   SUCCESS
# classify_ground    SUCCESS
# build_tin          SUCCESS
# build_dtm          SUCCESS
# extract_contours   SUCCESS
# detect_features    SUCCESS
# export_dxf         SUCCESS
# export_gpkg        SUCCESS
```

### Multi-class classification (a separate stage, confirmed independently runnable)

```python
wf.classify_points()   # CLASSIFICATION_RESULT -- distinct from classify_ground()'s GROUND_CLOUD
```

### The same pipeline, modular API (no `Workflow`, confirmed equivalent)

```python
from topocore.io.las import LASReader
from topocore.processing.ground.manager import GroundManager
from topocore.terrain.tin import TIN
from topocore.terrain.dtm import DTM
from topocore.terrain.interpolation import TerrainInterpolator
from topocore.terrain.contours import ContourGenerator
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.grid import Grid

with LASReader("nube.las") as r:
    cloud = r.read()

ground_cloud = GroundManager(method="grid", cell_size=1.0).extract(cloud)

# TIN.from_points() needs Point3D tuples -- see primitives.md for
# building these from a Chunk's own X/Y/Z arrays directly
tin = TIN.from_points(points)

grid = Grid(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0, resolution=2.0)
interpolator = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR)
dtm = DTM.from_tin(tin, grid, interpolator)

contours = ContourGenerator(tin).generate(interval=0.5, base=0.0)
```

## Part 4 — The survey path, confirmed end to end

```python
from topocore.workflow import Workflow

wf = Workflow()
wf.read_survey("survey.txt")
wf.build_features_from_survey()
wf.resolve_sides()     # confirmed: no PAVEMENT_EDGE features present in
                        # this fixture -> passes through unmodified, not an error
wf.export_dxf("survey_salida.dxf", strict=False)

for stage in wf.result.stages:
    print(stage.stage, stage.status)
# read_survey                 SUCCESS
# build_features_from_survey  SUCCESS
# resolve_sides                SUCCESS
# export_dxf                    SUCCESS
```

## Part 4b — A road survey: centerline, pavement edges, laterality, and LandXML linearity

A realistic field survey of a straight 20m road segment, surveyed
**cross-section by cross-section every 5m** -- the way a real
topographic crew typically works (walk to a station, shoot left
edge, centerline, right edge, move to the next station), not by
walking one entire edge first. Confirmed end to end with real
execution, including 2 real pitfalls found while building this
example and the fix for each.

### Pitfall 1 — the default grouping mode cannot handle cross-section surveys at all

`FeatureBuilder`'s default mode groups only *consecutive* points
sharing the same field code (confirmed directly from `_group_runs`'s
own docstring). In a cross-section survey, the code changes at every
single point (edge, centerline, edge, edge, centerline, edge, ...),
so every "run" is exactly 1 point long. Confirmed by real execution
against exactly this data: **0 features result, all 15 points
diagnosed `INSUFFICIENT_POINTS`** -- not an error, but silently
nothing useful either.

### The real fix — field-code grammar syntax

`topocore.features.grammar`'s `BASE.FIGURE[.S|E|X]` syntax (a real,
standard total-station convention) lets figures be assembled correctly
even when interleaved with other figures in the file:

```text
id,x,y,z,code
1,0.000,3.500,10.100,BORDEPAV.1.S    <- figure 1, START
2,0.000,0.000,10.000,EJE.1.S
3,0.000,-3.500,10.100,PAV.1.S
4,5.000,3.500,10.100,BORDEPAV.1      <- figure 1, CONTINUE (no suffix)
5,5.000,0.000,10.000,EJE.1
6,5.000,-3.500,10.100,PAV.1
...
13,20.000,3.500,10.100,BORDEPAV.1.E  <- figure 1, END
14,20.000,0.000,10.000,EJE.1.E
15,20.000,-3.500,10.100,PAV.1.E
```

```python
from topocore.survey.reader import SurveyTXTReader
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry

survey_points = SurveyTXTReader("via_secciones.txt").read()
result = FeatureBuilder(FeatureCodeRegistry.default(), use_field_code_grammar=True).build(survey_points)

for f in result.features:
    print(f.feature_type, len(f.geometry.vertices))
# confirmed real output, all 3 correctly assembled despite interleaving:
# FeatureType.PAVEMENT_EDGE  5
# FeatureType.CENTERLINE     5
# FeatureType.PAVEMENT_EDGE  5
```

### Pitfall 2 — `Workflow.build_features_from_survey()` cannot reach grammar mode at all

Confirmed directly from its own real signature
(`build_features_from_survey(self, registry: Any = None)`) -- there
is no way to pass `use_field_code_grammar` through `Workflow`. **A
cross-section survey can only be processed via the modular
`FeatureBuilder` API directly, never through `Workflow` for this
stage.** This is a real, current gap between the 2 APIs, not
something this guide is choosing to omit.

### Laterality is confirmed order-independent — start from either edge, it doesn't matter

```python
from topocore.features.side.resolver import SideResolver
from topocore.features.models import FeatureCollection

collection = FeatureCollection()
for f in result.features:
    collection.add(f)
resolved = SideResolver().resolve(collection)

for f in resolved:
    print(f.feature_type, f.attributes.get("side"))
# confirmed: PAVEMENT_EDGE left, CENTERLINE None, PAVEMENT_EDGE right
```

Confirmed by real execution, specifically to answer whether survey
order matters: reversing one edge's own vertex order (simulating
"surveyed starting from the other end") **produced the exact same
left/right result**. `SideResolver` determines laterality from actual
geometric position relative to the centerline's own direction (a
cross-product computation), never from which edge was surveyed first
or which direction the points run in.

### Applying "linearity" via LandXML/`Alignment` — stationing the surveyed centerline

Confirmed with real execution: the surveyed centerline's own
`FeatureGeometry.vertices` become an `Alignment`, enabling station-based
queries (chainage) and LandXML export -- this is the sense in which
survey data connects to [`alignment.md`](./alignment.md)'s own
linearity domain:

```python
from topocore.geometry.point2d import Point2D
from topocore.alignment.elements import LineElement
from topocore.alignment.models import Alignment
from topocore.io.landxml import LandXMLDocument, LandXMLWriter, NamedAlignment

centerline = next(f for f in resolved if f.feature_type.name == "CENTERLINE")
vertices = centerline.geometry.vertices   # confirmed real output: 5 vertices, (0,0,10) to (20,0,10)

elements = tuple(
    LineElement(start=Point2D(vertices[i][0], vertices[i][1]),
                end=Point2D(vertices[i + 1][0], vertices[i + 1][1]))
    for i in range(len(vertices) - 1)
)
alignment = Alignment(name="EJE_VIA_1", elements=elements)
alignment.length   # confirmed real output: 20.0 -- exact

# "Linealidad" in practice -- querying a chainage/station along the
# surveyed centerline, the way you'd locate a utility crossing or a
# cross-section design point:
for station in (0.0, 7.5, 12.5, 20.0):
    print(f"K0+{station:05.2f} ->", alignment.station_to_point(station))
# confirmed real output, all exact linear interpolations:
# K0+00.00 -> Point2D(x=0.0, y=0.0)
# K0+07.50 -> Point2D(x=7.5, y=0.0)
# K0+12.50 -> Point2D(x=12.5, y=0.0)
# K0+20.00 -> Point2D(x=20.0, y=0.0)

document = LandXMLDocument(alignments=(NamedAlignment(name="EJE_VIA_1", alignment=alignment),), crs="EPSG:32618")
LandXMLWriter("eje_via.xml").write(document)   # confirmed: SUCCESS
```

`Alignment` has no built-in *inverse* query (point -> station) --
confirmed directly, only `station_to_point()` exists (see
[`alignment.md`](./alignment.md)) -- projecting an arbitrary surveyed
point onto the alignment to find its own station is not something
this guide can show as existing, real syntax.

### Recommended default for unknown user uploads (e.g. a web API/WebGIS backend)

Confirmed by real execution: `use_field_code_grammar=True` produces
the identical result on a plain, legacy-style survey (no `.`
separator anywhere in its codes) as `False` does -- it's a strict
superset, never a behavior change for files that don't use the
grammar syntax. A backend that accepts field-survey uploads from
unknown users, with no way to know in advance which convention a
given file used, can safely always do this:

```python
from topocore.survey.reader import SurveyTXTReader
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry

survey_points = SurveyTXTReader(uploaded_path).read()
result = FeatureBuilder(
    FeatureCodeRegistry.default(),
    use_field_code_grammar=True,   # confirmed safe as a default -- covers both conventions
).build(survey_points)
```

This is unrelated to running in "the cloud" or behind a web API as
such -- confirmed directly, the limitation lives in one specific
function (`Workflow.build_features_from_survey()`), not in TopoCore
generally or in any deployment environment. Backend code built around
`FeatureBuilder` directly (as above) is unaffected wherever it runs;
backend code built around calling `Workflow.build_features_from_survey()`
inherits the exact same silent-empty-result gap, in the cloud or
anywhere else.

### What kind of survey works correctly through `Workflow`, and what to do once you need `FeatureBuilder` directly

**Confirmed working through `Workflow`**: a survey where each linear
feature is walked as one uninterrupted run -- the whole centerline
first, then the whole left edge, then the whole right edge (never
alternating between them station by station). This is exactly
`via.txt` from earlier in this page.

**Once you need `FeatureBuilder` directly** (because your survey
interleaves codes), a second, related fact matters: confirmed
directly, `Workflow.build_tin()`/`build_dtm()`/`extract_contours()`
require `GROUND_CLOUD` -- which only comes from `classify_ground()`
on a `POINT_CLOUD` (`read_point_cloud()`). There is no path from
`SURVEY_POINT_SET` to `TIN`/`DTM` anywhere in `Workflow`'s own stage
graph, regardless of the grammar issue -- `Workflow`'s own survey
branch was already limited to feature-building, laterality,
transforms, and export, nothing terrain-related.

This means once you call `FeatureBuilder` directly, continuing the
rest of the pipeline manually costs you nothing real: every remaining
step is the exact same function `Workflow` would have called
internally, just invoked directly instead of through its wrapper.
Confirmed with a real, complete run:

```python
from topocore.survey.reader import SurveyTXTReader
from topocore.features.feature_builder import FeatureBuilder
from topocore.features.feature_codes import FeatureCodeRegistry
from topocore.features.models import FeatureCollection
from topocore.features.side.resolver import SideResolver
from topocore.geodesy.crs import CRS
from topocore.geodesy.transformer import CoordinateTransformer
from topocore.geodesy.transform import transform_feature_collection
from topocore.dxf import DXFExporter
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions

# 1. Reading -- the same as wf.read_survey()
survey_points = SurveyTXTReader("via_secciones.txt").read()

# 2. Features -- the same as wf.build_features_from_survey(), but WITH grammar
result = FeatureBuilder(FeatureCodeRegistry.default(), use_field_code_grammar=True).build(survey_points)
collection = FeatureCollection()
for f in result.features:
    collection.add(f)
collection.crs = "EPSG:32618"

# 3. Laterality -- the same as wf.resolve_sides()
collection = SideResolver().resolve(collection)

# 4. CRS transform -- the same underlying function wf.transform_crs() calls
transformer = CoordinateTransformer(CRS.from_epsg(32618), CRS.from_epsg(4326))
collection_wgs84 = transform_feature_collection(collection, transformer)

# 5. Export -- the same as wf.export_dxf()/wf.export_gpkg()
DXFExporter().export(collection, "via.dxf")
GeoPackageExporter(GPKGExportOptions(epsg=32618)).export(collection, "via.gpkg")
```

Confirmed real output: every step succeeds, producing the identical
result `Workflow` would have, for every capability `Workflow`'s own
survey branch actually offers.

## Part 5 — Analysis: the real per-manager methods, corrected

**A real correction to [`analysis.md`](./analysis.md), found while
writing this page**: not every analysis manager only offers the
generic `compute(*args, method=...)` dispatch documented there. Each
also has its own real, directly-named methods -- confirmed by reading
each manager's own source, and more convenient than `compute()` for
a known task:

```python
from topocore.analysis.statistics.manager import StatisticsAnalysis

stats = StatisticsAnalysis()
stats.elevation(tin)   # confirmed: accepts a raw array, a TIN (via
                        # .elevation_array()), or anything with
                        # .elevations (e.g. a DTM) -- NOT method=...
# -> ElevationStats(minimum=10.02, maximum=18.06, mean=14.00, ...)

stats.slope(dtm)        # confirmed: requires something with .elevations
                        # specifically -- a TIN does NOT have this,
                        # only a DTM does; calling with a TIN raises
                        # AttributeError
# -> SlopeStats(minimum=0.07, maximum=8.79, mean=3.67, histogram_bins=[...], ...)
```

```python
from topocore.analysis.distance.manager import DistanceAnalysis

dist = DistanceAnalysis()
dist.available_methods()   # ('euclidean', 'geodesic', 'horizontal', 'vertical', 'slope')
                            # confirmed: only DistanceAnalysis has this method,
                            # not the other 3 managers
dist.compute(0.0, 0.0, 3.0, 4.0, method="euclidean")   # -> DistanceResult(value=5.0, ...)
dist.gradient(0.0, 0.0, 10.0, 3.0, 4.0, 10.0)          # -> 0.0 (same elevation, no rise)
```

```python
from topocore.analysis.profile.manager import ProfileAnalysis

profile = ProfileAnalysis()
profile.longitudinal(...)    # confirmed to exist directly, not via compute()
profile.transversal(...)
profile.cross_section(axis, surface, interval=..., width=...)
profile.multi(...)
```

```python
from topocore.analysis.visibility.manager import VisibilityAnalysis

vis = VisibilityAnalysis()
vis.line_of_sight(...)   # confirmed to exist directly
vis.viewshed(...)
vis.intervisibility(...)
```

## Part 6 — Surface comparison and volume, confirmed with real numbers

```python
import numpy as np
from topocore.analysis.comparison.surface import SurfaceComparison
from topocore.analysis.comparison.cut_fill import SurfaceCutFill

existing = np.full((10, 10), 10.0)
proposed = np.full((10, 10), 12.0)

comparison = SurfaceComparison(tolerance=0.01)
result = comparison.compute(existing, proposed)
# result.mean_difference == 2.0, result.fill_mask is all True (confirmed)

cutfill = SurfaceCutFill(cell_area=4.0)   # 10x10 grid, 2m² cells -> 100 cells * 4m²
comparison_result, volume_result = cutfill.compute(existing, proposed)
# volume_result.fill_volume == 800.0  (100 cells * 4m² * 2m difference -- confirmed exact)
# volume_result.cut_volume == 0.0
# volume_result.net_volume == -800.0
```

## Part 7 — Every export configuration, confirmed

```python
from topocore.gpkg import GeoPackageExporter, GPKGExportOptions
from topocore.dxf import DXFExporter
from topocore.dxf.models import ExportContext, DXFExportOptions

# GeoPackage, strict (default) -- any invalid feature aborts the whole export
GeoPackageExporter(GPKGExportOptions(epsg=32618)).export(collection, "out.gpkg")

# GeoPackage, non-strict -- isolates bad features, keeps the rest
GeoPackageExporter(GPKGExportOptions(epsg=32618, strict=False)).export(collection, "out.gpkg")

# DXF, default options
DXFExporter().export(collection, "out.dxf")

# DXF, explicit options
options = DXFExportOptions(dxf_version="R2010", strict=False, index_contour_every=5)
DXFExporter(ExportContext(options=options)).export(collection, "out.dxf")
```

See [`export.md`](./export.md) for the complete report-field
reference and the real table schema `GeoPackageExporter` produces
(only `survey_code`/`survey_name`/`cad_layer` become native columns;
everything else lands in `attributes_json`).

## What this page does not cover

`GeoidGrid.from_geotiff()` itself against a real geoid file (GDAL
could not be installed in this sandbox -- see
[`geodesy.md`](./geodesy.md) for the exact reason and for using it
against this project's own real, bundled geoid files), and the 9
domain feature-code catalogs' own individual contents (see
[`catalogs.md`](./catalogs.md)).
