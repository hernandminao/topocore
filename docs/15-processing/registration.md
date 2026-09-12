# Registration (ICP)

Point-cloud-to-point-cloud alignment via Iterative Closest Point —
**fully implemented**, despite what the package's own module
docstring currently says (see [`overview.md`](./overview.md) for that
correction).

```python
from topocore.processing.registration import PointToPointICP

icp = PointToPointICP(max_correspondence_distance=5.0)
result = icp.register(source_cloud, target_cloud)
```

Verified directly: registering a 200-point cloud against a copy of
itself translated by a known `(2.0, 1.0, 0.5)` offset recovers that
exact translation (`result.transformation.translation ==
[2.0, 1.0, 0.5]`), with `fitness=1.0` and an RMSE of
`~6e-15` — floating-point-exact, for this pure-translation case.

## The 2 algorithms

```python
PointToPointICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=1.0, use_adaptive_distance=True)
PointToPlaneICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=1.0, use_adaptive_distance=True, normal_k=10)
```

`PointToPointICP` minimizes point-to-point distance directly.
`PointToPlaneICP` minimizes the distance along each target point's
own local surface normal instead (estimated internally via `normals`,
controlled by `normal_k`) — typically converges faster and more
robustly on smooth, planar surfaces, at the cost of needing a
meaningful local normal at each target point (noisy or very sparse
data can make this less reliable than the simpler point-to-point
criterion).

## `RegistrationResult`

```python
result.transformation        # Transformation -- the estimated rigid transform
result.converged             # bool
result.iterations            # int -- how many ICP iterations actually ran
result.fitness                # float -- fraction of source points with a valid correspondence
result.rmse                   # float -- root-mean-square error over matched correspondences
result.source_transformed     # PointCloud -- source, with the estimated transformation applied
result.has_source_transformed # bool
```

## `max_correspondence_distance` and `use_adaptive_distance`

ICP only matches a source point to a target point within
`max_correspondence_distance` — too small, and genuinely corresponding
points in a poorly-initialized alignment never get matched at all,
so the algorithm cannot converge. `use_adaptive_distance=True` (the
default) lets this threshold shrink automatically across iterations,
starting more permissive and tightening as the alignment improves — a
different strategy from fixing one value for every iteration.

## `RegistrationManager`

Selects between the 2 algorithms by name, matching the manager
pattern used throughout this section.

## What registration does not do

It never assumes a known correspondence between specific points in
the 2 clouds — correspondences are found by nearest-neighbor search
(via `neighbors`) at every iteration, which is why a reasonable
initial alignment matters: ICP finds a *local* optimum, not a global
one, and a source/target pair that starts too far apart or too
rotated can converge to the wrong alignment entirely.
