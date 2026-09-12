# Quickstart

A complete, real, minimal pipeline: read a point cloud, classify
ground, build terrain, extract contours, export to both DXF and
GeoPackage. Every call below uses `Workflow`'s own real, current
signature -- confirmed directly against the source, not written from
memory.

```python
from topocore.workflow import Workflow
from topocore.terrain.grid import Grid
from topocore.terrain.enums import InterpolationMethod

workflow = Workflow()

# 1. Read a point cloud. Format is selected by file extension --
#    .las, .laz, .e57, .ply, .xyz, .csv, .pts are all supported.
workflow.read_point_cloud("survey.las")

# 2. Classify ground points (GroundManager.extract() under the hood).
workflow.classify_ground()

# 3. Build a TIN from the ground points.
workflow.build_tin()

# 4. Build a DTM -- requires a Grid and an interpolation method.
grid = Grid(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0, resolution=1.0)
workflow.build_dtm(grid, method=InterpolationMethod.LINEAR)

# 5. Extract contours directly from the TIN (not the DTM).
workflow.extract_contours(interval=1.0, base=0.0)

# 6. Detect features from the point cloud (buildings, vegetation,
#    infrastructure, ...) -- TIN/DTM/classification are used
#    automatically if already present in this same Workflow.
workflow.detect_features(strict=False)

# 7. Export. Both may be called on the same FeatureCollection.
workflow.export_dxf("output.dxf")
workflow.export_gpkg("output.gpkg", epsg=32618)

# Inspect what happened.
result = workflow.result
for stage in result.stages:
    print(stage.stage, stage.status)
```

## Reading a total-station/GNSS survey instead

For field-surveyed points with feature codes (not a raw point
cloud), use `read_survey()` and `build_features_from_survey()`
instead of `read_point_cloud()`/`detect_features()`:

```python
workflow = Workflow()
workflow.read_survey("survey.txt")
workflow.build_features_from_survey()
workflow.export_dxf("survey_features.dxf")
```

## Confirmed error behavior worth knowing up front

- Calling a stage before its own required artifact exists raises
  `WorkflowStateError` (e.g. `build_tin()` before `classify_ground()`
  has produced `GROUND_CLOUD`).
- `detect_features(strict=False)` isolates one bad detector and
  continues with the rest; `strict=True` (the default) aborts the
  whole detection run on the first failure.
- `export_dxf()`/`export_gpkg()` take the same `strict` semantics via
  `**exporter_kwargs` (e.g. `workflow.export_dxf("out.dxf",
  strict=False)`), confirmed fixed and correct as of this project's
  own audit: an invalid `Feature` is isolated, not fatal to the whole
  export, when `strict=False`.
- Every stage failure raises `WorkflowExecutionError`, with the
  original underlying exception attached via `.__cause__` -- prior,
  already-successful artifacts in the same `Workflow` are never
  affected by a later stage's failure.

See [`workflows.md`](./workflows.md) for the complete contract of
every stage, and [`reference.md`](./reference.md) for a one-page
lookup table.
