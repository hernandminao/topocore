# Quick Reference

Every signature below is the real, current one, confirmed directly
against the source during PR22's own audit.

## `Workflow` — high-level API

| Task | Call |
|---|---|
| Create a workflow | `Workflow(*, progress: ProgressObserver \| None = None)` |
| Read a point cloud (LAS/LAZ/E57/PLY/XYZ/CSV/PTS, by extension) | `.read_point_cloud(path: str \| Path, **reader_kwargs)` |
| Read a total-station/GNSS survey text file | `.read_survey(path: str \| Path, **reader_kwargs)` |
| Transform CRS in place | `.transform_crs(artifact_type: ArtifactType, transformer)` |
| Transform vertical datum in place | `.transform_vertical(artifact_type, *, source_datum, target_datum, geoid)` |
| Control-point georeference in place | `.georeference(artifact_type, controls, target_crs, options)` |
| Classify ground vs. non-ground | `.classify_ground(**manager_kwargs)` |
| Multi-class point classification | `.classify_points(*, method: str = "auto", **manager_kwargs)` |
| Build a TIN (from `GROUND_CLOUD`) | `.build_tin()` |
| Build a DTM (from `TIN`) | `.build_dtm(grid, *, method: InterpolationMethod, power: float = 2.0)` |
| Extract contours (from `TIN`, not `DTM`) | `.extract_contours(interval: float, *, base: float = 0.0)` |
| Detect features from a point cloud | `.detect_features(*, strict: bool = True)` |
| Build features from a surveyed `SurveyPointSet` | `.build_features_from_survey(registry=None)` |
| Resolve left/right side for pavement-edge features | `.resolve_sides(**resolver_kwargs)` |
| Export to DXF | `.export_dxf(path: str \| Path, **exporter_kwargs)` |
| Export to GeoPackage | `.export_gpkg(path: str \| Path, *, epsg: int \| None = None, **exporter_kwargs)` |
| Read the current value of an artifact | `.artifact(artifact_type: ArtifactType)` |
| Inspect full execution history | `.result` (a `WorkflowResult`) |
| Which artifact type this run started from | `.input_kind` |
| Branch into an independent copy | `.snapshot()` |

## Point-cloud readers — modular API

All implement the same `PointCloudReader` base contract
(`.read() -> PointCloud`); use as a context manager (`with Reader(...) as r:`).

| Format | Reader | Real constructor |
|---|---|---|
| LAS | `topocore.io.las.LASReader` | `(path, *, chunk_size: int = DEFAULT_CHUNK_SIZE)` |
| LAZ | `topocore.io.laz.LAZReader` | `(path, *, chunk_size: int = DEFAULT_CHUNK_SIZE)` |
| E57 | `topocore.io.e57.E57Reader` | `(path, *, chunk_size: int)` -- **required**, no default |
| PLY | `topocore.io.ply.reader.PLYReader` | `(path, *, chunk_size: int = 1_000_000)` |
| XYZ | `topocore.io.ascii.xyz.XYZReader` | `(path, *, chunk_size: int = 1_000_000, encoding: str = "utf-8")` |
| CSV | `topocore.io.ascii.csv.reader.CSVReader` | `(path, *, chunk_size: int = 1_000_000, encoding: str = "utf-8", has_header: bool \| None = None)` |
| PTS | `topocore.io.ascii.pts.reader.PTSReader` | `(path, *, chunk_size: int = 1_000_000, encoding: str = "utf-8")` |

**Survey text (total station / GNSS), not a point cloud:**

| Reader | Real constructor | Returns |
|---|---|---|
| `topocore.survey.reader.SurveyTXTReader` | `(path, *, format: SurveyFormat \| None = None, delimiter: str \| None = None, encoding: str = "utf-8")` | `.read() -> SurveyPointSet` |

`SurveyTXTReader` resolves column layout by explicit `format` preset
first, then by auto-detecting a recognized header row; if neither
applies, it raises `SurveyFormatError` rather than guessing. If a
`.prj` sidecar exists next to the file, `SurveyPointSet.crs` is set
from it automatically (coordinates are never transformed by this).

## Terrain — modular API

| Task | Call |
|---|---|
| Build a TIN from points | `topocore.terrain.tin.TIN.from_points(points: tuple[Point3D, ...])` |
| Build a TIN from an existing mesh | `TIN.from_mesh(vertices, simplices)` |
| Define a grid | `topocore.terrain.grid.Grid(min_x, min_y, max_x, max_y, resolution)` |
| Interpolate a TIN onto a grid | `topocore.terrain.dtm.DTM.from_tin(tin, grid, interpolator)` |
| Generate contours from a TIN | `topocore.terrain.contours.ContourGenerator(tin).generate(interval, base=0.0)` |

## Feature extraction — modular API

| Task | Call |
|---|---|
| Run every registered detector | `topocore.features.manager.FeatureExtractionManager(strict=True).detect_all(context)` |
| Build features from survey field codes | `topocore.features.feature_builder.FeatureBuilder(registry).build(survey_points)` |
| Default feature-code registry | `topocore.features.feature_codes.FeatureCodeRegistry.default()` |

## Export — modular API

| Task | Call |
|---|---|
| Export to DXF | `topocore.dxf.DXFExporter(context: ExportContext \| None = None).export(collection, path)` |
| Export to GeoPackage | `topocore.gpkg.GeoPackageExporter(options: GPKGExportOptions).export(collection, path)` |

See [`export.md`](./export.md) for the full `strict`/error contract
of both exporters.

## Geodesy — modular API

| Task | Call |
|---|---|
| Build a CRS | `CRS.from_epsg(epsg)` / `.from_wkt(...)` / `.from_proj4(...)` / `.from_authority(...)` / `.from_json(...)` |
| Build a coordinate transformer | `CoordinateTransformer(source_crs, target_crs)` |
| Transform a point cloud directly | `topocore.geodesy.transform.transform_point_cloud(cloud, transformer)` |
| Detect a UTM zone | `UTMZone.from_latlon(latitude, longitude)` |
| Auto-detect a file's own CRS | `topocore.io.crs.detector.detect_crs(path)` |
| Load a real geoid grid | `GeoidGrid.from_geotiff(path)` -- **you must supply your own real EGM96/EGM2008/other file** |
| Convert ellipsoidal <-> orthometric height | `VerticalTransformer(...).ellipsoidal_to_orthometric(...)` / `.orthometric_to_ellipsoidal(...)` |
| Fit a georeferencing from control points | `fit_georeferencing(controls, options)` |

See [`geodesy.md`](./geodesy.md) for the complete API, including 2
things worth knowing before relying on this module's own internal
docstrings.

## Catalogs — modular API

| Task | Call |
|---|---|
| Default registry (built-in catalog) | `FeatureCodeRegistry.default()` |
| Look up a code | `registry.get(code: str) -> FeatureCodeDefinition \| None` |
| Register one definition | `registry.register(definition, *, overwrite: bool = False)` |
| Register many | `registry.register_many(definitions, *, overwrite: bool = False)` -- **not atomic**, see [`catalogs.md`](./catalogs.md) |
| Load an external JSON catalog | `topocore.features.catalogs.loaders.json_loader.load_json(path, *, encoding="utf-8")` |
| Load an external CSV catalog | `topocore.features.catalogs.loaders.csv_loader.load_csv(path, *, encoding="utf-8-sig")` |
| Load an external YAML catalog | `topocore.features.catalogs.loaders.yaml_loader.load_yaml(path, *, encoding="utf-8")` -- requires `pyyaml` |

## Where to go next

Each of the above is covered in full, with error contracts and
confirmed audit findings, in its own guide:
[`workflows.md`](./workflows.md), [`point-clouds.md`](./point-clouds.md),
[`primitives.md`](./primitives.md), [`geodesy.md`](./geodesy.md),
[`landxml.md`](./landxml.md), [`alignment.md`](./alignment.md),
[`features.md`](./features.md), [`feature-builder-cases.md`](./feature-builder-cases.md), [`terrain.md`](./terrain.md),
[`analysis.md`](./analysis.md), [`surface.md`](./surface.md),
[`export.md`](./export.md), [`catalogs.md`](./catalogs.md).
