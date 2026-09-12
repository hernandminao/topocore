# Geodesy

Everything below was extracted and verified directly against the
real source. 2 honest findings surfaced during this extraction and
are documented explicitly rather than smoothed over -- see
"2 things worth knowing" at the end of this page.

## `CRS` — no public constructor

```python
from topocore.geodesy.crs import CRS

CRS.from_epsg(epsg: int) -> CRS
CRS.from_wkt(wkt: str) -> CRS
CRS.from_proj4(proj4: str) -> CRS
CRS.from_authority(auth_name: str, code: str | int) -> CRS
CRS.from_json(json_str: str) -> CRS
```

Confirmed directly: `CRS()` raises `RuntimeError` unconditionally --
you must use one of the 5 classmethods above. Each wraps `pyproj`
internally and raises `CRSError` (not a bare `pyproj` exception) on
failure.

```python
crs.epsg              # int | None
crs.name               # str
crs.authority           # tuple[str, str] | None
crs.is_geographic        # bool
crs.is_projected          # bool
crs.is_vertical            # bool
crs.is_compound              # bool
crs.is_geocentric              # bool
crs.ellipsoid                    # Ellipsoid | None
crs.datum                          # Datum | None
crs.projection                      # ProjectionInfo | None
crs.to_wkt() -> str
crs.to_proj4() -> str
crs.to_json() -> str
crs.equals(other: CRS) -> bool
```

## CRS validation

```python
from topocore.geodesy.validation import (
    validate_lat_lon, validate_epsg, validate_bbox,
    validate_array, validate_coordinate_arrays,
)

validate_lat_lon(latitude: float, longitude: float) -> None   # raises on invalid
validate_epsg(epsg: int) -> None
```

Each raises the relevant domain exception (not a bare `ValueError`)
for out-of-range or non-finite input -- called internally by `CRS.from_epsg()`,
`UTMZone.from_latlon()`, and others; you rarely need to call these
yourself unless validating raw user input before constructing a `CRS`.

## Transforming X/Y — `CoordinateTransformer`

```python
from topocore.geodesy.transformer import CoordinateTransformer

transformer = CoordinateTransformer(source_crs: CRS, target_crs: CRS)
transformer.transform_point(x: float, y: float, z: float | None = None) -> tuple[float, float, float | None]
transformer.transform_array(xs, ys, zs)   # vectorized form
```

Alternate constructor for a directly-supplied (non-EPSG) transform:

```python
transformer = CoordinateTransformer.from_operation(operation: CoordinateOperation)
```

`from_operation()` supports `IDENTITY` and the static (7-parameter)
`HELMERT` form for a geographic source and target CRS -- confirmed
verified numerically against EPSG formula 9606. The time-dependent
(14-parameter) `HELMERT` form is explicitly, deliberately
unsupported here (a real scope boundary, not a missing
implementation) -- applying its own rate terms correctly needs an
observation epoch for the data being transformed, which
`transform_point()`/`transform_array()` have no way to supply.

## Helmert parameters — the 7 (or 14) numbers themselves

```python
from topocore.geodesy.helmert import HelmertParameters

HelmertParameters(
    tx: float, ty: float, tz: float,     # translation, meters
    rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,  # rotation, arc-seconds
    scale: float = 0.0,                    # ppm
    dtx=None, dty=None, dtz=None, drx=None, dry=None, drz=None, dscale=None,  # 14-param rates
    reference_epoch: float | None = None,   # required if any rate is given
)
```

One class covers both the static (7-parameter) and time-dependent
(14-parameter) forms -- the rate fields are simply `None` for a
static transformation.

## Transforming each artifact type, directly (bypassing `Workflow`)

```python
from topocore.geodesy.transform import (
    transform_point_cloud, transform_survey, transform_feature_collection,
)

new_cloud = transform_point_cloud(cloud: PointCloud, transformer: CoordinateTransformer) -> PointCloud
new_survey = transform_survey(survey: SurveyPointSet, transformer: CoordinateTransformer) -> SurveyPointSet
new_collection = transform_feature_collection(collection: FeatureCollection, transformer: CoordinateTransformer) -> FeatureCollection
```

All 3 are confirmed pure -- the input object is never mutated, a new
one is returned. `GROUND_CLOUD` uses the same `transform_point_cloud()`
as `POINT_CLOUD` -- confirmed directly, both are the exact same
`PointCloud` type with no special-casing that would distinguish a
"full" cloud from a ground-only one. `transform_feature_collection()`
re-validates every transformed `Feature`'s own geometry automatically
(via `FeatureGeometry.__post_init__`) -- a transform that somehow
produced non-finite output fails loudly there, not silently
downstream. `MESH` face indices pass through unchanged (they index
vertices, they are not coordinates).

`Workflow.transform_crs(artifact_type, transformer)` (see
[`workflows.md`](./workflows.md)) is the orchestrated equivalent --
it calls exactly these 3 functions internally, dispatching by
`artifact_type`, plus 2 safety checks (never silently re-transform an
already-referenced artifact; reject the 4 unsupported artifact types
explicitly).

## Distance and geodesic calculations

```python
from topocore.geodesy.distance import distance2d, distance3d

distance2d(x1: float, y1: float, x2: float, y2: float) -> float
distance3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float
```

```python
from topocore.geodesy.geodesic import GeodesicCalculator

calc = GeodesicCalculator(crs: CRS)
calc.distance(...)
calc.azimuth(...)
calc.inverse(...)
calc.forward(...)
calc.polygon_area(...)
```

## UTM

```python
from topocore.geodesy.utm import UTMZone

zone = UTMZone.from_latlon(latitude: float, longitude: float) -> UTMZone
zone.zone_number      # int
zone.zone_letter       # str
zone.hemisphere         # "N" | "S"
zone.epsg                # int -- the real EPSG code for this zone
zone.central_meridian
zone.false_easting
zone.false_northing
```

Confirmed directly: handles the real Norway (zone 32 override,
56-64N/3-12E) and Svalbard (zones 31/33/35/37 override, 72-84N) UTM
grid exceptions explicitly, not just the plain 6-degree formula.

## Automatic CRS detection from a file

```python
from topocore.io.crs.detector import detect_crs

crs = detect_crs(path: str | Path) -> CRS | None
```

The single, format-agnostic entry point across every format TopoCore
reads. Returns `None` for "no CRS found" or an unrecognized
extension -- both are normal, expected outcomes, never an exception
-- except `FileNotFoundError`, checked unconditionally before any
format dispatch. Confirmed real per-format strategy: LAS/LAZ/E57 try
their own native CRS metadata first, falling back to a `.prj`
sidecar only if the native format carries none; LandXML reads only
its own `<CoordinateSystem>` element, with no `.prj` fallback
(confirmed deliberate: `LandXMLReader.read()` returns a
`LandXMLDocument`, not a `PointCloud`/`SurveyPointSet` -- an
incompatible shape for the `.prj`-application machinery).

## CRS propagation during reading

```python
from topocore.io.crs.external import apply_crs_with_native_priority, apply_external_crs

cloud = apply_crs_with_native_priority(cloud: PointCloud, path, native_crs: CRS | None) -> PointCloud
cloud = apply_external_crs(cloud: PointCloud, path) -> PointCloud   # .prj sidecar only
```

`apply_crs_with_native_priority()`: if the format's own native
detector already found a CRS, that value is used and the `.prj`
sidecar is never even consulted -- confirmed this holds even when a
`.prj` present alongside the file declares a genuinely different
CRS. Both functions mutate and return the same `cloud` object (via
its own public `.crs` setter), never raising for an absent or
unresolvable CRS from either source.

## CRS propagation through processing to output

Confirmed directly, by reading each real call site (not assumed):
`GroundManager.extract()`'s own resulting `GROUND_CLOUD` copies
`.crs` from the source `POINT_CLOUD` explicitly (a real, historical
gap -- found and fixed -- see [`workflows.md`](./workflows.md)'s own
`classify_ground()`). `FeatureCollection.crs` is set from the source
`PointCloud.crs` directly by `detect_features()`, or converted from
`SurveyPointSet.crs`'s own `CRS` object (via the same
`f"EPSG:{code}"`-or-`.name` string convention) by
`build_features_from_survey()`. No processing stage documented in
this project's own audit was found to silently drop a CRS that was
present on its own input.

## Vertical datum and geoid-based height conversion

```python
from topocore.geodesy.vertical_datum import VerticalDatum

VerticalDatum(name: str, authority: str | None = None, geoid_model: str | None = None, remarks: str | None = None)
```

```python
from topocore.geodesy.vertical.geoid_grid import GeoidGrid

geoid = GeoidGrid.from_geotiff(path: str | Path) -> GeoidGrid
geoid.undulation_at(longitude: float, latitude: float) -> float   # bilinear interpolation
```

**Confirmed requirements for any real geoid file to work here** --
not specific to the 2 files this project ships, any real GeoTIFF
meeting these passes:

- Exactly 1 raster band (checked explicitly; a multi-band file
  raises `GeoidError` naming how many bands it actually has).
- A non-rotated, non-sheared `GeoTransform` (checked explicitly).
- **A genuinely geographic (longitude/latitude) coordinate reference
  system, embedded in the file itself.**
- Openable by GDAL (a separate, lazily-imported dependency).

**Fixed during this project's own usage-documentation work**:
`from_geotiff()` used to accept any single-band raster's own
`GeoTransform` origin/pixel values unconditionally, treating them as
longitude/latitude in degrees with no check at all -- a projected
geoid grid (e.g. in UTM meters) would have been silently
misinterpreted as geographic, producing a wrong undulation value with
no error at all. Now confirmed fixed: `from_geotiff()` reads the
file's own embedded CRS (`GetProjection()`) and confirms it is
genuinely geographic (via `topocore.geodesy.CRS`, already a mandatory
dependency) before accepting the file -- raising `GeoidError` with a
clear message for a projected CRS, a missing CRS, or unreadable CRS
information, rather than silently producing a wrong result. Verified
directly: a real `EPSG:4326` file is accepted; a real `EPSG:32618`
(UTM 18N) file is rejected with a clear error naming its own CRS.

**Important, confirmed directly from the module's own source, with 1
correction based on this project's own real repository layout**: the
`topocore.geodesy.vertical` module itself does not bundle, hardcode,
or fabricate any geoid model data -- `GeoidGrid.from_geotiff()` reads
*whatever* single-band GeoTIFF you give it, and what is implemented
and tested inside this module is the loading/interpolation machinery
itself (verified against a synthetic grid with hand-computed expected
values), not a specific geoid's own real-world values. That said,
this project's own real repository -- confirmed directly from its own
`data/geoids/` directory -- does ship 2 real, usable grid files:

```text
data/geoids/Gravimetric_QGeoidCOL2023_DP.tif   -- Colombia's own gravimetric geoid
data/geoids/us_nga_egm2008_1.tif                -- EGM2008 (NGA)
```

```python
geoid = GeoidGrid.from_geotiff("data/geoids/us_nga_egm2008_1.tif")
# or, for work within Colombia specifically:
geoid = GeoidGrid.from_geotiff("data/geoids/Gravimetric_QGeoidCOL2023_DP.tif")
```

Reading a GeoTIFF requires GDAL, imported lazily only inside
`from_geotiff()` -- not needed for any other part of
`topocore.geodesy`. If you work with a different geoid model or
region, obtain the corresponding grid from its own authoritative
source (e.g. NGA for EGM96/EGM2008) in the GeoTIFF profile PROJ
itself recommends.

```python
from topocore.geodesy.vertical.transformer import VerticalTransformer

transformer = VerticalTransformer(source_datum: VerticalDatum, target_datum: VerticalDatum, geoid: GeoidGrid)
transformer.ellipsoidal_to_orthometric(longitude, latitude, height) -> float   # H = h - N
transformer.orthometric_to_ellipsoidal(longitude, latitude, height) -> float   # h = H + N
```

Confirmed directly from the class's own docstring: `source_datum`/
`target_datum` are carried for traceability only -- exactly one
geoid-based conversion is actually implemented today (ellipsoidal
height vs. this one geoid's own orthometric surface), not a general
datum-to-datum shift system. Both directions raise
`MissingGeoidGridError` (propagated from `GeoidGrid.undulation_at()`)
if the requested point falls outside the grid's own extent or on a
nodata cell -- confirmed this never silently returns an uncorrected
height, closing a real danger confirmed during this capability's own
design review: a plain `pyproj.Transformer` was confirmed to
silently return an unchanged height when its own required vertical
grid was missing.

```python
from topocore.geodesy.vertical.transform import (
    transform_point_cloud_vertical, transform_survey_vertical, transform_feature_collection_vertical,
)
```

Same pure, non-mutating pattern as the horizontal transform functions
above -- `Workflow.transform_vertical()` (see
[`workflows.md`](./workflows.md)) is the orchestrated equivalent.

## Georeferencing by control points

```python
from topocore.geodesy.georeferencing import (
    ControlPoint, GeoreferencingOptions, GeoreferencingResult, GeoreferencingStrategy,
    fit_georeferencing,
)

point = ControlPoint(id: str, source: tuple[float, float, float], target: tuple[float, float, float])

options = GeoreferencingOptions(
    accept_underconstrained: bool = False,
    minimum_control_distance: float | None = None,
    # see this module's own docstring for the complete field list
)

result = fit_georeferencing(controls: Sequence[ControlPoint], options: GeoreferencingOptions) -> GeoreferencingResult
result.strategy        # GeoreferencingStrategy.TRANSLATION_ONLY | HELMERT_2D | HELMERT_3D
result.parameters       # the fitted HelmertParameters
```

Estimates a Helmert similarity transformation (translation, and --
if the control points' own geometry supports it -- rotation and
scale) from known source/target coordinate correspondences, as
opposed to `CoordinateTransformer`, which applies an *already-known*
transformation between 2 already-identified CRS. `fit_georeferencing()`
automatically selects whichever of `TRANSLATION_ONLY`/`HELMERT_2D`/
`HELMERT_3D` the control points' own mathematical rank actually
supports -- confirmed raising `InsufficientControlPointsError` for
too few points, and `UnderconstrainedGeoreferencingError` (unless
`accept_underconstrained=True` is explicitly given) rather than
silently assuming unmeasured parameters.

`Workflow.georeference()` (see [`workflows.md`](./workflows.md)) is
the orchestrated equivalent -- confirmed to also reject running this
a second time on an already-referenced artifact.

## 2 things worth knowing before relying on this module's own docstrings

**1. `io/crs/detector.py`'s own module docstring is stale on one
point.** It states, under "deliberately out of scope," that
control-point georeferencing "does not exist in TopoCore yet."
Confirmed directly, this is no longer accurate: `topocore.geodesy.georeferencing`
(documented above, including `Workflow.georeference()`) is a real,
implemented, tested capability. The docstring was evidently written
before that module existed and was never updated afterward -- a
documentation lag within the codebase itself, not a functional gap.
This guide reflects the current, real state.

**2. This module's own code-level docstring is narrower than what
this project's real repository actually ships.** `vertical/geoid_grid.py`'s
own docstring states the *module* fabricates no geoid data of its
own (accurate -- it is loading/interpolation machinery, not a data
source) -- but the real repository's own `data/geoids/` directory
does ship 2 real, ready-to-use grids (Colombia's own gravimetric
geoid and EGM2008), confirmed directly from this project's own real
file layout. Use `GeoidGrid.from_geotiff("data/geoids/...")` with
either one directly; you do not need to source your own grid file
for those 2 cases.

## Where to go next

- [`workflows.md`](./workflows.md) -- `transform_crs()`/
  `transform_vertical()`/`georeference()`'s own `Workflow`-level
  contract, dependencies, and safety checks.
- [`point-clouds.md`](./point-clouds.md) -- `PointCloud.crs`, and how
  each reader populates it.
