# Geodesy — Overview

TopoCore's `geodesy` package is a thin, verified layer over `pyproj`:
Coordinate Reference Systems, coordinate transformations, geodesic
(ellipsoidal) calculations, vertical (geoid-based) height correction,
and georeferencing by control points.

## Four operations — do not conflate them

This section, together with
[`../13-io/crs-detection.md`](../13-io/crs-detection.md), covers 4
genuinely different operations:

| Operation | Question it answers | Automatic? |
|---|---|---|
| **Detection** | "What CRS does this file's own header/sidecar declare?" | Yes — every format `topocore.io` reads, at read time. See [`../13-io/crs-detection.md`](../13-io/crs-detection.md). |
| **Declaration / propagation** | "What CRS is this in-memory object currently expressed in?" | Set by a reader (from detection) or by an earlier transform/georeference stage; carried forward automatically from there. |
| **Transformation** | "Convert coordinates from a known source CRS to a known target CRS." | No — you build the `CRS`/`CoordinateTransformer` yourself; nothing decides this for you. |
| **Georeferencing** | "Estimate a transformation from local/arbitrary coordinates to a real CRS, using control-point correspondences." | No — you supply the `ControlPoint`s yourself; strategy selection *within* the fit is automatic. |

```text
        IO                          Workflow                    GEODESY
┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│ LAS / LAZ / E57 /    │    │                     │    │ CRS                  │
│ PLY / ASCII / XML    │──► │ PointCloud          │──► │ CoordinateTransformer│
│                      │    │ SurveyPointSet      │    │ transform_point_cloud│
│ .crs detected         │    │ FeatureCollection   │    │ transform_survey     │
│ automatically where   │    │                     │    │ transform_feature_   │
│ the file declares one │    │  .artifact(type)    │    │   collection         │
│ (native or .prj)      │    │                     │    │ georeference()       │
└─────────────────────┘    └─────────────────────┘    └──────────┬───────────┘
                                                                   │
                                                                   ▼
                                                        new, transformed/
                                                        georeferenced artifact
```

Reading a file now gives you a `PointCloud`/`SurveyPointSet` whose
`.crs` is already populated whenever the source file declares one
(natively, or via a `.prj` sidecar) — see
[`../13-io/crs-detection.md`](../13-io/crs-detection.md) for the
complete, per-format architecture. What TopoCore still never does
automatically is decide to *transform* those coordinates, or to
*georeference* local ones: you construct the source and target `CRS`
yourself and call `transform_crs()` explicitly, or supply
`ControlPoint`s yourself and call `georeference()` explicitly.
Nothing runs either of those two for you.

## What's genuinely functional here — verified, not assumed

| Component | Status |
|---|---|
| `CRS` | Functional — construction, properties, equality, round-trips through 4 formats |
| CRS detection (`topocore.io.crs.detect_crs()`) | Functional — native detection for LAS/LAZ/E57/LandXML, `.prj` fallback for every format; see [`../13-io/crs-detection.md`](../13-io/crs-detection.md) |
| `CoordinateTransformer` | Functional — point/array transforms, Helmert operations |
| `transform_point_cloud`/`transform_survey`/`transform_feature_collection` | Functional — apply an existing transformer to TopoCore's own models, updating each model's own `.crs` |
| `GeodesicCalculator` | Functional — distance/azimuth/area on the ellipsoid, verified against a known real-world distance |
| `UTMZone` | Functional — including the Norway/Svalbard special cases, verified |
| `GeoidGrid`/`VerticalTransformer`/`Workflow.transform_vertical()` | Functional — real GeoTIFF loading, bilinear interpolation, never silently returns an uncorrected height; validated end to end against a real EGM2008 grid file. Requires a real geoid grid file (not bundled) — see [`vertical-reference.md`](./vertical-reference.md) |
| Georeferencing by control points (`fit_georeferencing()`, `Workflow.georeference()`) | Functional — automatic strategy selection by geometric rank, explicit authorization required for underdetermined geometry; see [`georeferencing.md`](./georeferencing.md) |

## What's declared but not implemented — see `declared-not-implemented.md`

| Component | What it claims to be | What it actually does |
|---|---|---|
| `LocalCRS` | A project-local engineering coordinate system | Pure data. No transformation method exists — its own docstring states this belongs to a future PR. |
| `TransformationAccuracy` | The accuracy of an executed transformation | Pure data. Confirmed zero construction sites anywhere in the codebase — never actually produced by any transformation. |

## Closing status

This is a genuinely solid, working part of TopoCore — not an
incomplete one with a few gaps. The line between what's implemented
and what's future work is deliberate and clearly marked, not
accidental:

| Capability | Status |
|---|---|
| `CRS` / coordinate reference systems | Implemented |
| CRS detection (native + `.prj`, every format) | Implemented |
| Horizontal coordinate transformation | Implemented |
| `Workflow.transform_crs()` | Implemented |
| `Workflow.artifact()` (safe artifact retrieval) | Implemented |
| Geodesic calculations (`GeodesicCalculator`) | Implemented |
| UTM zone handling (`UTMZone`) | Implemented |
| Laterality resolution (`SideResolver`, `Workflow.resolve_sides()`) | Implemented |
| Vertical (geoid-based) transformation (`GeoidGrid`, `VerticalTransformer`, `Workflow.transform_vertical()`) | Implemented — requires a real geoid grid file, not bundled |
| Georeferencing by control points (`Workflow.georeference()`) | Implemented |
| Vertical datum detection from any reader | Not implemented — `source_datum`/`target_datum` must be supplied explicitly |
| Input/processing/output CRS orchestration model on `Workflow` | Future product decision, not started |

## A note on discrepancies

Where this documentation's own verification contradicted a code
comment, the executed behavior is what's documented, with the
discrepancy noted explicitly — see `transform-functions.md` for one
concrete example (a stale comment referencing an already-fixed
`PointCloud.clone()` limitation).

## In this section

- [`crs.md`](./crs.md) — `CRS`: construction, properties, equality.
- [`transformer.md`](./transformer.md) — `CoordinateTransformer` and
  `CoordinateOperation`/`HelmertParameters`.
- [`transform-functions.md`](./transform-functions.md) —
  `transform_point_cloud`/`transform_survey`/
  `transform_feature_collection`, and the TD-003 discrepancy.
- [`workflow-integration.md`](./workflow-integration.md) —
  `Workflow.transform_crs()`, the one explicit, opt-in bridge between
  `Workflow` and `geodesy` for applying an already-known
  transformation.
- [`georeferencing.md`](./georeferencing.md) — estimating a
  transformation from local coordinates via control points, and
  `Workflow.georeference()`.
- [`geodesic.md`](./geodesic.md) — `GeodesicCalculator`.
- [`utm.md`](./utm.md) — `UTMZone`.
- [`declared-not-implemented.md`](./declared-not-implemented.md) —
  `LocalCRS`, `TransformationAccuracy`.
- [`io-boundary.md`](./io-boundary.md) — what CRS detection resolves
  automatically versus what still requires explicit action, and how
  LandXML's own behavior differs.
- [`limitations.md`](./limitations.md) — consolidated list.
- [`vertical-reference.md`](./vertical-reference.md) — real,
  implemented, tested vertical (geoid-based) transformation:
  `GeoidGrid`, `VerticalTransformer`,
  `Workflow.transform_vertical()`. Requires a real geoid grid file,
  not bundled with TopoCore.
