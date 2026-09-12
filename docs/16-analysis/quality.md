# Quality

10 accuracy/quality metrics between point clouds, surfaces, or scalar
observations -- the largest submodule in `topocore.analysis`, all
taking raw `NDArray` inputs (never `PointCloud`/`TIN` objects
directly, except `CloudToMeshDistance`'s own `tin` parameter), plus a
dispatcher (`QualityAnalysis`).

## Cloud/point comparison metrics

```python
from topocore.analysis.quality import CloudToCloudDistance, HausdorffDistance, ChamferDistance, CloudToMeshDistance

CloudToCloudDistance(max_distance=0.0).compute(reference, compared)   # -> CloudToCloudResult: distances, mean, median, min, max, std
HausdorffDistance(percentile=100.0).compute(reference, compared)      # -> HausdorffResult: directed_forward, directed_backward, hausdorff, symmetric
ChamferDistance().compute(set_a, set_b)                                 # -> ChamferResult: directed_forward, directed_backward, chamfer
CloudToMeshDistance().compute(pointcloud, tin)                          # point cloud vs. a TriangulatedSurface, not another cloud
```

Verified directly with a uniform `0.1`-unit offset between 2 identical
point sets: `CloudToCloudDistance.mean == 0.1`,
`HausdorffDistance.hausdorff == HausdorffDistance.directed_forward ==
HausdorffDistance.directed_backward == 0.1`, `ChamferDistance.chamfer
== 0.1` -- all 3 agree exactly for this simple, symmetric case, as
expected (they only diverge from each other for genuinely asymmetric
point distributions).

## `RMSEAnalysis` -- expects point coordinates, not scalar values

```python
from topocore.analysis.quality import RMSEAnalysis

RMSEAnalysis().compute(reference, observed)   # both (N, 3) -- NOT flat scalar arrays
# -> RMSEResult: rmse_3d, horizontal, vertical, total, count
```

**Passing flat 1D arrays "works" (no error) but the result's own
attribute names won't mean what you'd expect** -- confirmed directly:
a 1D call produces a result with no `.rmse` attribute at all (only
`horizontal`/`vertical`/`rmse_3d`/`total`), since this class always
treats its input as `(N, 3)` point coordinates. Verified directly
with a `+0.1` uniform Z offset: `rmse_3d == vertical == 0.1`,
`horizontal == 0.0` -- confirming the 3 components decompose
correctly.

## Statistical/QA-style metrics

```python
from topocore.analysis.quality import PrecisionAnalysis, CompletenessAnalysis, CorrectnessAnalysis, GPSControlAnalysis, RegistrationQuality

PrecisionAnalysis(tolerance_x=0.02, tolerance_y=0.02, tolerance_z=0.04, confidence_level=0.95).compute(observations)
    # -> PrecisionResult: std_dev, confidence_interval, count, confidence_level
    # Repeated observations of the SAME point -- measures repeatability/scatter, not accuracy against a known reference.

CompletenessAnalysis(resolution=1.0).compute(points, reference_bbox)
    # -> CompletenessResult: coverage_ratio, total_area, uncovered_area
    # How much of a reference bounding box's own area has any data at all -- detects coverage gaps.

CorrectnessAnalysis(tolerance=0.05).compute(model_grid, reference_grid)
    # -> CorrectnessResult: true_positives, false_positives, false_negatives, precision, recall, f1_score
    # A confusion-matrix-style classification metric, not a distance metric.

GPSControlAnalysis(tolerance_x=0.05, tolerance_y=0.05, tolerance_z=0.1).compute(reference, observed)
    # -> GPSControlResult: mean_deviation, max_deviation, points_passing, points_failing, control_point_count, tolerance

RegistrationQuality(tolerance=0.05).compute(residuals)
    # -> RegistrationQualityResult: rmse, mean_distance, max_distance, fitness
    # Takes ALREADY-COMPUTED residuals (e.g. from an ICP run), not raw point clouds.
```

`PrecisionAnalysis` and `GPSControlAnalysis` are easy to confuse:
`PrecisionAnalysis` measures how consistent repeated measurements of
the *same* point are with each other (no reference coordinate
needed); `GPSControlAnalysis` measures how far *known* control points
deviate from their *observed* GPS coordinates, against explicit
per-axis tolerances, reporting a pass/fail count.

## `QualityAnalysis` -- the dispatcher

```python
QualityAnalysis(method="c2c").compute(reference, compared)
```

Dispatches to all 10 above by name. `QualityMethod` (the
method-selector constant) is a plain class, not a `StrEnum` -- see
[`contracts.md`](./contracts.md).

## Invalid input, verified

An empty reference cloud is rejected explicitly, not silently
producing `NaN`/`0`-valued statistics:

```python
>>> CloudToCloudDistance().compute(empty_array, some_points)
QualityError: Reference cloud must not be empty.
```

A single-point comparison is accepted and computes a real, meaningful
distance (verified directly: `sqrt(3)` for `(0,0,0)` vs `(1,1,1)`) --
not a degenerate/rejected case here, unlike the near-zero-range
histogram issue found in [`statistics.md`](./statistics.md).
