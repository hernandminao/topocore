# Vertical Reference & Geoid Transformation

**This capability is implemented and tested.** `topocore.geodesy.vertical`
provides `GeoidGrid` (loads and bilinearly interpolates a real
geodetic GeoTIFF grid) and `VerticalTransformer` (applies it to
convert between ellipsoidal and orthometric height), plus a
`Workflow.transform_vertical()` stage. This page replaces an earlier
design-only proposal for this same capability — everything described
here is real, importable, and verified by direct execution.

## What is, and is not, included

**TopoCore does not bundle, fabricate, or couple itself to any one
geoid model.** Confirmed directly: `VerticalTransformer` never
branches its own logic on `source_datum`/`target_datum`'s specific
name — they're carried purely for traceability. The exact same
transformation code runs identically whether the geoid you supply is
EGM96, EGM2008, a national model such as Colombia's own GEOCOL2004,
or a project-specific grid — verified directly by running the same
transformation against 3 differently-named `VerticalDatum`s over the
same grid and confirming byte-identical results.

**What's missing is real geoid data, not the architecture or the
integration.** No geoid model file is bundled with TopoCore itself —
this is a genuine, ongoing data-provisioning requirement, not a gap
in the implementation. The capability has been validated end to end
against a real EGM2008 grid file: loading, bilinear interpolation,
and a full `Workflow.transform_vertical()` run all confirmed correct
against real, unmodified data. Bring your own real geoid GeoTIFF
(obtained via your national mapping agency, e.g. IGAC's own GEOCOL
model for Colombia, or NGA's own EGM2008/EGM96 distribution) and
point `GeoidGrid.from_geotiff()` at it — the path is fully
configurable, with no assumption anywhere about where the file
lives.

## Architecture

```text
topocore/geodesy/
├── crs.py
├── transformer.py           # horizontal (X/Y)
├── transform.py
│
└── vertical/                 # vertical (Z) -- kept separate, never
    ├── exceptions.py         # merged into CoordinateTransformer
    ├── geoid_grid.py
    ├── transformer.py
    └── transform.py
```

Horizontal and vertical transformation are deliberately independent,
composable components — a caller wanting both applies each
explicitly, never a single class trying to do both. See
[`transformer.md`](./transformer.md) for the horizontal side.

## `GeoidGrid`

```python
from topocore.geodesy.vertical import GeoidGrid

grid = GeoidGrid.from_geotiff("path/to/your_geoid.tif")
n = grid.undulation_at(longitude, latitude)
```

Loads any single-band GeoTIFF (the format PROJ itself recommends for
geodetic grids today) and bilinearly interpolates at an arbitrary
point. Requires GDAL — a lazily-imported optional dependency;
importing `topocore.geodesy` (or even `topocore.geodesy.vertical`
itself) never requires GDAL to be installed, only calling
`.from_geotiff()` does.

Verified against a hand-computable synthetic grid (interpolation
results checked against arithmetic, not merely "it runs"), and
separately against a deliberately non-square, asymmetric grid over
Colombia's own real geographic extent — constructed specifically to
catch any row/column transposition or north/south or east/west axis
inversion, which a symmetric grid could not reveal. Both confirmed
correct: axis order, coordinate units (decimal degrees, not
projected meters), and coverage rejection (a point genuinely outside
the grid's own region, tested with Madrid against a Colombia-extent
grid) all behave as expected.

### The one non-negotiable safety guarantee

**`GeoidGrid`/`VerticalTransformer` never return a height that was
left uncorrected because the required geoid data was unavailable.**
This directly closes a real, confirmed danger: plain
`pyproj.Transformer` was confirmed, during this capability's own
design review, to silently return an unchanged height when its own
required vertical grid is missing, rather than raising.

```python
>>> grid.undulation_at(100.0, 100.0)  # outside the grid's own extent
MissingGeoidGridError: Point (100.0, 100.0) is outside the geoid
grid's own extent (source: '...').
```

The same guarantee holds for a nodata cell, a missing file, and GDAL
not being installed — every one of these raises explicitly, verified
directly for each.

## `VerticalTransformer`

```python
from topocore.geodesy.vertical import VerticalTransformer
from topocore.geodesy import VerticalDatum

transformer = VerticalTransformer(
    source_datum=VerticalDatum(name="WGS84 ellipsoidal"),
    target_datum=VerticalDatum(name="EGM2008 orthometric", geoid_model="EGM2008"),
    geoid=grid,
)

H = transformer.ellipsoidal_to_orthometric(longitude, latitude, h)
h = transformer.orthometric_to_ellipsoidal(longitude, latitude, H)
```

`H = h - N` and its exact inverse — verified with an exact round
trip (`h → H → h` returns the original value precisely).

## Applying it to TopoCore's own models

```python
from topocore.geodesy.vertical.transform import (
    transform_point_cloud_vertical,
    transform_survey_vertical,
    transform_feature_collection_vertical,
)
```

Mirrors `topocore.geodesy.transform`'s own per-type functions, but
shifts only `Z`. **Caller responsibility, stated plainly**: `X`/`Y`
must already be geographic degrees matching the geoid grid's own CRS
— these functions do not reproject, and do not consult or validate
against whatever `.crs` value the artifact may already carry (see
[`../13-io/crs-detection.md`](../13-io/crs-detection.md) for how that
value gets there) — even a `PointCloud` with `.crs` correctly set to
a projected CRS is not checked against or reprojected here. If your
data is projected, run `transform_point_cloud()` (horizontal) to a
geographic CRS first.

**Fails the whole operation, never partially.** If any single point
in a `PointCloud`/`SurveyPointSet`/`FeatureCollection` falls outside
the geoid grid's own extent, the entire call raises
`MissingGeoidGridError` — never a result where some heights were
corrected and others silently were not.

## `Workflow.transform_vertical()`

```python
workflow.transform_vertical(
    ArtifactType.SURVEY_POINT_SET,
    source_datum=source_datum,
    target_datum=target_datum,
    geoid=geoid,
)
```

Same 4 supported artifact types as
[`Workflow.transform_crs()`](./workflow-integration.md)
(`POINT_CLOUD`, `GROUND_CLOUD`, `SURVEY_POINT_SET`,
`FEATURE_COLLECTION`); any other type is rejected explicitly.
Constructs a `VerticalTransformer` internally from the 3 named
arguments (unlike `transform_crs()`, which receives an already-built
`CoordinateTransformer` directly). A `MissingGeoidGridError` from the
underlying transform propagates as `WorkflowExecutionError`, exactly
like any other stage failure — confirmed directly that the artifact
remains at its pre-transform value when this happens, never silently
re-produced with an uncorrected height.

## What remains explicitly out of scope

- **Vertical (datum-level) CRS detection.** Horizontal CRS detection
  is implemented across every format (see
  [`../13-io/crs-detection.md`](../13-io/crs-detection.md)) — but
  nothing detects which *vertical* datum a file's own heights are
  expressed in. You must still supply `source_datum`/`target_datum`
  explicitly.
- **An `input`/`processing`/`output` CRS orchestration model on
  `Workflow`.** A genuine architectural decision touching every
  existing stage, not implemented here.
- **Bundled real geoid model data.** Genuinely a data-provisioning
  requirement, not an architectural gap — bring your own in
  deployment (see above).

These remain real, open product decisions — not limitations accepted
as permanent, and not blockers to using the vertical transformation
capability itself once a real geoid grid is available.
