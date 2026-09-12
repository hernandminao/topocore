# Geometric Feature Computation

**This is `topocore.processing.features` — geometric properties of a
point cloud, used as input to classification and segmentation.** It
is not `topocore.features` (semantic feature detection: buildings,
breaklines, vegetation, producing `Feature`/`FeatureCollection`
objects), documented separately. The two packages share no code and
serve different purposes; a "feature" here is a per-point numeric
value (height, density, planarity), never a detected object.

```python
from topocore.processing.features import HeightFeatureComputer

heights = HeightFeatureComputer().compute(cloud)   # FloatArray1D, shape (N,)
```

## The computers

```python
HeightFeatureComputer()                            # Z relative to the cloud's own minimum Z
RelativeHeightFeatureComputer(ground_class=2, k=1)  # height above the nearest ground-classified point
DensityFeatureComputer(radius=1.0)                  # local point count within radius, per point
DistanceToNeighborFeatureComputer(k=1)              # distance to the k-th nearest neighbor
PCAFeatureComputer(feature_name, k=10)               # one named PCA-derived shape descriptor
```

Every computer implements `.compute(cloud) -> FloatArray1D` (shape
`(N,)`) — a `ScalarFeatureComputer`. Verified directly on a 30-point
synthetic cloud: `HeightFeatureComputer` and `DensityFeatureComputer`
both return an array shaped `(30,)`, one value per point.

## `PCAFeatureComputer` — 10 named descriptors from one local covariance analysis

```python
PCAFeatureComputer(feature_name="planarity", k=10)
```

`feature_name` must be one of: `"eigenvalues"`, `"eigenvectors"`,
`"omnivariance"`, `"anisotropy"`, `"linearity"`, `"planarity"`,
`"sphericity"`, `"surface_variation"`, `"verticality"`,
`"eigenentropy"` — all derived from the same local PCA (the 3
eigenvalues of each point's own local covariance matrix), just
different combinations of them. `linearity`/`planarity`/`sphericity`
are also directly reachable as their own dedicated methods on
`PCAFeatures` (`.linearity()`, `.planarity()`, `.sphericity()`) if you
need those 3 specifically without naming them as a string.

## `FeatureManager`

A caching entry point selecting between the computers above by name,
matching the manager pattern used throughout this section
(`neighbors.NeighborhoodManager`, `ground.GroundManager`,
`normals.NormalManager`).

## Used by

`classification`'s rule-based and ML classifiers both consume these
computers as their own input features (e.g. relative height above
ground, planarity, verticality) — see
[`classification.md`](./classification.md). `segmentation`'s
`TreeSegmenter`/`BuildingSegmenter` use relative height similarly.
