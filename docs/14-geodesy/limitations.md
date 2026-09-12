# Limitations

Everything below was confirmed by direct execution during this
documentation's own preparation — not read off a comment and taken
on faith, and not inferred from what the PROJ/pyproj ecosystem
generally supports.

**TopoCore does not automatically georeference data, and does not
apply a geodetic transformation without explicit configuration.**
This is not the same as "TopoCore does not detect CRS" — horizontal
CRS detection from a file's own native metadata or a `.prj` sidecar
is implemented across every format (see
[`../13-io/crs-detection.md`](../13-io/crs-detection.md)). Nor is it
the same as "TopoCore cannot work with georeferenced data" — when the
source and target CRS are known, TopoCore provides everything needed
to build and execute the transformation, including as a normal
chained step inside `Workflow.transform_crs()` (see
[`workflow-integration.md`](./workflow-integration.md)). The
limitation is specifically that TopoCore will not decide *what
transformation to apply* for you, and will not transform anything
unless you ask it to — detecting a CRS and transforming coordinates
remain two separate, deliberate steps.

## CRS detection is automatic; transformation and georeferencing are not

- Every format `topocore.io` reads can resolve a real CRS
  automatically at read time — native metadata where the format has
  it, a `.prj` sidecar otherwise. See
  [`../13-io/crs-detection.md`](../13-io/crs-detection.md) for the
  complete architecture and per-format reliability (E57's own native
  mechanism, in particular, resolves a real CRS only rarely in
  practice).
- `LandXMLReader.read()` itself still only preserves
  `<CoordinateSystem>` as an opaque string on `LandXMLDocument.crs` —
  a **separate** function resolves that value into a real `CRS`; see
  [`../12-landxml/units-and-crs.md`](../12-landxml/units-and-crs.md).
- `Workflow.transform_crs()` lets you apply an already-built
  transform as a normal chained stage once you know the source and
  target CRS, but it is entirely opt-in and never decides the CRS
  itself. Once you provide that configuration explicitly, TopoCore
  can transform `PointCloud`, `GroundCloud`, `SurveyPointSet`, and
  `FeatureCollection` — see
  [`transform-functions.md`](./transform-functions.md).
- `Workflow.georeference()` estimates a transformation from local,
  unreferenced coordinates using known control-point correspondences
  — a genuinely different operation from both detection and
  transformation. See [`georeferencing.md`](./georeferencing.md).

## `CoordinateTransformer` scope, verified exhaustively

- `from_operation()` supports only `IDENTITY` and static
  (7-parameter) `HELMERT` between two geographic CRS. Time-dependent
  (14-parameter) Helmert and `GRID_SHIFT` both raise
  `TransformationError` explicitly — see
  [`transformer.md`](./transformer.md).
- `CoordinateTransformer` itself always passes `Z` straight through,
  never adjusted for a vertical datum difference — that's a separate,
  now-implemented capability, `topocore.geodesy.vertical` (see
  [`vertical-reference.md`](./vertical-reference.md)), not something
  `CoordinateTransformer` does itself.

## Vertical Reference & Geoid Transformation — implemented, requires a real geoid grid file

`GeoidGrid`/`VerticalTransformer`/`Workflow.transform_vertical()` are
real, tested, and confirmed working — including against a real
EGM2008 grid file, end to end — see
[`vertical-reference.md`](./vertical-reference.md) for the full
capability. The one genuine remaining limitation is data
availability, not architecture: **no geoid grid file (EGM96, EGM2008,
or otherwise) is bundled with TopoCore.** Vertical datum detection is
also not implemented — nothing determines which vertical datum a
file's own heights are expressed in; `source_datum`/`target_datum`
must always be supplied explicitly.

This has always failed loudly by design, never silently: confirmed
directly that plain `pyproj.Transformer` silently returns an
unchanged height when its own required vertical grid is missing,
rather than raising — `GeoidGrid`/`VerticalTransformer` never do
this, at any of the 3 layers (grid interpolation, the transformer
itself, the `Workflow` stage).

Bring your own real geoid GeoTIFF in your deployment environment to
use this for real work — confirmed directly, the implementation
places no restriction on which geoid model it represents (EGM96,
EGM2008, or a national model such as Colombia's own GEOCOL2004 all
run through the identical code path).

## Georeferencing by control points — implemented, with explicit authorization required for underdetermined geometry

`Workflow.georeference()` estimates a transformation from local,
unreferenced coordinates to a real CRS via known control-point
correspondences — see [`georeferencing.md`](./georeferencing.md).
Strategy (full Helmert, 2D Helmert, or translation-only) is selected
automatically by the control points' own geometric rank; an
underdetermined result (fewer control points than the full model
needs) is rejected unless explicitly authorized via
`GeoreferencingOptions(accept_underconstrained=True)` — never
silently assumed. Calling `georeference()` a second time on an
already-georeferenced artifact is rejected, not silently layered.

## Two types that describe capabilities TopoCore doesn't yet have

- `LocalCRS`: no transformation method exists at all — see
  [`declared-not-implemented.md`](./declared-not-implemented.md).
- `TransformationAccuracy`: confirmed zero construction sites
  anywhere in the codebase — never actually produced by any real
  transformation.

`VerticalDatum` used to be listed here; it now has a real consumer
(`VerticalTransformer`) — see
[`declared-not-implemented.md`](./declared-not-implemented.md) for
the updated detail.

## A documented, corrected discrepancy

`transform_point_cloud()`'s own source comments cite a historical
`PointCloud.clone()` defect ("TD-003") as the reason it builds
`Chunk`s manually rather than calling `.clone()`. Confirmed directly:
this defect no longer exists — `clone()` performs a genuine, complete
deep copy today. The function still works correctly; only the stated
rationale in its own comments is stale. See
[`transform-functions.md`](./transform-functions.md) for the full
verification.

## What's genuinely solid

None of the above should overshadow what's actually confirmed
working: `CRS` construction/properties/equality across 5 factory
methods, `CoordinateTransformer`'s supported operation types (with
Helmert geometry verified against hand-derivable math, not just "it
runs"), all three `transform_*` functions, `GeodesicCalculator`
(verified against a known real-world distance), `UTMZone` (including
the Norway/Svalbard special cases), automatic CRS detection across
every format TopoCore reads, vertical (geoid-based) transformation
including against real EGM2008 data, and georeferencing by control
points with automatic strategy selection. These are the parts of
`topocore.geodesy` (and its own IO-layer counterpart) that are ready
to build on today.
