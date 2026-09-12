# Ground Classification

5 genuinely different ground-classification algorithms, unified
behind one manager class:

```python
from topocore.processing.ground import GroundManager

ground_cloud = GroundManager(method="grid", cell_size=1.0).extract(cloud)
```

Verified directly, on a synthetic cloud of ~200 ground points
(`Z` in `[0, 0.3]`) plus ~50 elevated points simulating vegetation
(`Z` in `[3, 8]`): `grid`, `adaptive_grid`, `progressive_tin`, and
`pmf` all correctly separate the two groups (each keeping the ~200
low points, with `grid`/`progressive_tin` including a small number of
additional points in this particular synthetic scenario — the exact
count is algorithm- and parameter-dependent, not a fixed guarantee).

## The 5 methods

```python
GroundManager(method="grid", ...)            # GridGroundClassifier/Extractor
GroundManager(method="adaptive_grid", ...)   # AdaptiveGridGroundClassifier/Extractor
GroundManager(method="progressive_tin", ...) # ProgressiveTINGroundClassifier/Extractor
GroundManager(method="pmf", ...)             # PMFGroundClassifier/Extractor -- Progressive Morphological Filter
GroundManager(method="csf", ...)             # CSFGroundClassifier/Extractor -- Cloth Simulation Filter
```

Each method has its own dedicated classifier/extractor pair under
`topocore.processing.ground` (`GridGroundClassifier`,
`AdaptiveGridGroundClassifier`, `ProgressiveTINGroundClassifier`,
`PMFGroundClassifier`, `CSFGroundClassifier`, and their matching
`*Extractor` counterparts) — `GroundManager` is a convenience
dispatcher over all 5, not a wrapper adding its own logic.

## `csf` requires an optional dependency the other 4 do not

**Confirmed directly**: `method="csf"` requires the separate
`cloth-simulation-filter` package. The other 4 methods run on NumPy/
SciPy alone.

```python
>>> GroundManager(method="csf").extract(cloud)
GroundError: CSF ground filtering requires the optional
'cloth-simulation-filter' package. Install it with:
pip install cloth-simulation-filter
```

## `.classify()` vs. `.extract()`

```python
mask = manager.classify(cloud)     # BoolArray1D -- True where a point is ground
ground_cloud = manager.extract(cloud)   # PointCloud -- only the ground points
```

`extract()` does not call `classify()` and then filter separately
internally in a way that computes the classification twice — it runs
the underlying algorithm once. If you need both the mask and the
extracted cloud, deriving the ground/non-ground counts from
`extract()`'s own result (`ground_cloud.point_count` vs. the original
`cloud.point_count`) avoids a second, redundant classification pass.

## A note on this module's own test coverage

Confirmed directly: `ground`'s own test suite (254 lines) is smaller
than every other Processing submodule's (587–1715 lines elsewhere).
This is a genuine QA-coverage observation, not a statement that any
of the 5 methods is unimplemented or unreliable — each classifier and
extractor is real, complete code with its own passing tests; the
relative line count simply means less of the parameter space (edge
cases, degenerate inputs, parameter combinations) has dedicated test
coverage compared to modules like `sampling` or `segmentation`.

## Integration with `Workflow`

`Workflow.classify_ground()` wraps `GroundManager.extract()` (never
`.classify()`) and additionally propagates the source `PointCloud`'s
own `.crs` onto the resulting `GroundCloud` — ground classification
is a pure geometric subset (points removed, none moved or
reprojected), so the CRS is unchanged, but `GroundManager` itself
does not carry `.crs` forward on its own. See
[`workflow-integration.md`](./workflow-integration.md).
