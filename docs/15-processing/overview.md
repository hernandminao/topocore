# Processing — Overview

`topocore.processing` is where TopoCore does point-cloud-level
computation: spatial neighbor search, downsampling, outlier/spatial
filtering, normal and curvature estimation, geometric feature
computation, ground classification, multi-class point classification,
segmentation, and point-cloud-to-point-cloud registration (ICP).

## A correction worth stating plainly

The package's own module docstring (`topocore/processing/__init__.py`)
currently states registration (ICP) is "Not yet implemented." **This
is incorrect and does not reflect the actual codebase** —
`topocore.processing.registration` is fully implemented
(`PointToPointICP`, `PointToPlaneICP`, a shared `ICPBase`, and a
`RegistrationManager`), with 874 lines of its own tests, including a
regression test for a real historical defect. See
[`registration.md`](./registration.md). This is flagged here as a
stale code comment to be corrected in the source, not treated as a
real limitation of TopoCore.

## What's implemented — verified against the current code and tests

| Submodule | What it does | Status |
|---|---|---|
| [`neighbors`](./neighbors.md) | KD-tree KNN / radius search | Implemented |
| [`normals`](./normals.md) | PCA-based normal and curvature estimation | Implemented |
| [`filters`](./filters.md) | Statistical/radius outlier, pass-through, crop box, clip polygon | Implemented |
| [`ground`](./ground.md) | Ground classification — 5 selectable methods (CSF, grid, adaptive grid, PMF, progressive TIN) | Implemented — see [`ground.md`](./ground.md) for a note on its own test coverage |
| [`features`](./features.md) | Geometric feature computation (height, density, PCA-derived shape descriptors) | Implemented |
| [`sampling`](./sampling.md) | Random, uniform, voxel, stratified, density-based sampling | Implemented |
| [`segmentation`](./segmentation.md) | DBSCAN, region growing, connected components, tree/building segmenters | Implemented |
| [`classification`](./classification.md) | Rule-based ASPRS classification, plus 4 ML backends (Random Forest, Gradient Boost, XGBoost, LightGBM) | Implemented |
| [`registration`](./registration.md) | Point-to-point and point-to-plane ICP | Implemented |
| [`backend` / `cache`](./backend-and-cache.md) | Shared computation backend and LRU caching infrastructure | Implemented |

## A recurring pattern: an abstract class and a concrete class sharing the same name

Across several submodules, `base.py` defines an **abstract** class,
and a separate file defines the **concrete** implementation under the
exact same class name (importing the abstract one under a different
local name to avoid a clash within that one file). Confirmed directly
in 2 places:

```text
topocore/processing/segmentation/base.py
    class RegionGrowingSegmenter(Segmenter):        # abstract
topocore/processing/segmentation/region_growing.py
    class RegionGrowingSegmenter(BaseRegionGrowingSegmenter):  # concrete

topocore/processing/classification/base.py
    class RuleBasedClassifier(Classifier):          # abstract
topocore/processing/classification/rules.py
    class RuleBasedClassifier(_RuleBasedClassifier): # concrete
```

This is a deliberate, repeated pattern — not a naming collision or a
defect. What it means practically: **which `RuleBasedClassifier` (or
`RegionGrowingSegmenter`) you get depends on which module you import
it from.** `from topocore.processing.classification import
RuleBasedClassifier` (the package's own public `__all__`) gives you
the concrete, usable one; reaching into `.base` directly gives you
the abstract one, which cannot be instantiated. Each submodule's own
page notes this again only where it applies, rather than assuming
you've read this section first.

## Shared infrastructure

`backend/` (`NumPyBackend`) and `cache/` (`LRUCache`) are not
processing algorithms themselves — they're infrastructure several of
the algorithms above build on (caching neighbor queries, PCA results,
etc.). See [`backend-and-cache.md`](./backend-and-cache.md), including
a real, confirmed test-coverage asymmetry between the two.

## `topocore.processing.features` is not `topocore.features`

These are two genuinely different packages with similar names.
`topocore.processing.features` (documented in
[`features.md`](./features.md)) computes **geometric** properties of
a point cloud — height above ground, local point density, PCA-derived
shape descriptors (planarity, linearity, sphericity) — used as
*input* to classification and segmentation algorithms.
`topocore.features` (documented separately) is TopoCore's semantic
feature-detection package — buildings, terrain breaklines, vegetation,
etc., producing `Feature`/`FeatureCollection` objects. Neither wraps
the other; do not conflate them.

## Integration with `Workflow` and the rest of TopoCore

Only 2 of the 10 submodules above are wired into `Workflow` as their
own chainable stages: ground classification
(`Workflow.classify_ground()`) and multi-class point classification
(`Workflow.classify_points()`). The other 8 (neighbors, normals,
filters, features, sampling, segmentation, registration, and the
shared backend/cache infrastructure) are standalone utilities you call
directly on a `PointCloud` — some of them are used internally by
ground classification or the ML classifiers, but none has its own
`Workflow` stage. See [`workflow-integration.md`](./workflow-integration.md)
for the complete picture, including how this module relates to the
CRS/geodesy and vertical-reference capabilities documented in
[`../14-geodesy/`](../14-geodesy/).

## In this section

- [`neighbors.md`](./neighbors.md)
- [`normals.md`](./normals.md)
- [`filters.md`](./filters.md)
- [`ground.md`](./ground.md)
- [`features.md`](./features.md)
- [`sampling.md`](./sampling.md)
- [`segmentation.md`](./segmentation.md)
- [`classification.md`](./classification.md)
- [`registration.md`](./registration.md)
- [`backend-and-cache.md`](./backend-and-cache.md)
- [`workflow-integration.md`](./workflow-integration.md)
