# Normals

PCA-based surface normal and curvature estimation over a local
neighborhood of each point.

```python
from topocore.processing.normals import PCANormalEstimator

estimator = PCANormalEstimator(k=10)
normals = estimator.estimate(cloud)              # (N, 3)
normals, curvature = estimator.estimate_both(cloud)  # (N, 3), (N,)
```

Verified directly on a perfectly flat synthetic plane (points
scattered in X/Y, `Z = 0`): every normal resolves to `[0, 0, 1]`, and
curvature resolves to `0.0` at every point — exactly the expected
result for a locally flat surface.

## 3 separate interfaces, one class implementing all of them

`NormalEstimator.estimate()` returns normals only (`FloatArray2D`,
shape `(N, 3)`). `CurvatureEstimator.estimate()` returns curvature
only (`FloatArray1D`, shape `(N,)`) — the same method name, on a
different interface, with a different return shape; don't confuse
the two when reading the base classes directly.
`NormalAndCurvatureEstimator.estimate_both()` returns both together
in one pass, more efficiently than calling both separately.
`PCANormalEstimator` implements all 3 interfaces at once — it's the
one class you construct, and which of its methods you call decides
what you get back.

```python
estimator.estimate(cloud)          # normals only, (N, 3)
estimator.estimate_at(cloud, indices)  # normals for specific points only
estimator.estimate_both(cloud)     # (normals, curvature) in one pass
```

## `viewpoint` — orienting normals consistently, and a real validation gotcha

```python
PCANormalEstimator(k=10, viewpoint=np.array([x, y, z]))
```

PCA alone produces a normal direction with an arbitrary sign — without
a viewpoint, two adjacent points on the same surface can end up with
oppositely-flipped normals. Passing `viewpoint` (a sensor/scanner
position) orients every normal to point toward it.

**`viewpoint` must be a `numpy.ndarray`, not a plain Python tuple or
list.** Confirmed directly:

```python
>>> PCANormalEstimator(k=10, viewpoint=(5.0, 5.0, 100.0))
NormalError: viewpoint must be a numpy array, got tuple.
```

Wrap it explicitly: `viewpoint=np.array([5.0, 5.0, 100.0])`.

## `WeightedPCANormalEstimator`

A second implementation, weighting each neighbor's contribution
(e.g. by inverse distance) rather than treating every neighbor in the
local PCA equally. Same interface shape as `PCANormalEstimator`.

## `NormalManager`

A higher-level entry point selecting between the 2 estimators (and
their own configuration) by name, mirroring the manager pattern used
by `ground`/`classification`/`sampling` elsewhere in this section.

## Used internally by

`features` (PCA-derived shape descriptors reuse the same local
covariance computation) and `classification`/`segmentation`
algorithms that use surface orientation as an input feature.
