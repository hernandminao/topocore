# Visibility

3 genuinely different visibility computations over a `TriangulatedSurface`
-- one line between 2 points, one point-to-everywhere raster, and one
all-pairs network -- plus a dispatcher (`VisibilityAnalysis`).

## `LineOfSight` -- one observer, one target

```python
from topocore.analysis.visibility import LineOfSight

los = LineOfSight(observer_height=1.7, target_height=0.0, earth_curvature=True, num_samples=100)
result = los.compute(observer=(10.0, 50.0), target=(90.0, 50.0), tin=tin)
```

```python
result.visible               # bool
result.distance                # float -- observer to target
result.obstacles               # tuple -- terrain points blocking the line, if any
result.elevation_at_target     # float
result.clearance                # float -- how much margin the line has over the terrain (negative if blocked)
```

Verified directly on a flat terrain surface (no obstruction possible):
`visible=True`, with a small positive `clearance` matching the
observer/target height offsets used.

## `Viewshed` -- one observer, everywhere

```python
from topocore.analysis.visibility import Viewshed

viewshed = Viewshed(observer_height=1.7, resolution=5.0, max_distance=0.0, earth_curvature=True, num_samples=50)
result = viewshed.compute(observer=(50.0, 50.0), tin=tin)
```

```python
result.visibility_map     # per-sampled-point visibility
result.visible_points      # the subset actually visible
result.visible_count       # int
result.total_count          # int
result.visibility_ratio     # visible_count / total_count
```

Verified directly on a flat, unobstructed surface (`resolution=25.0`
over a 100x100 extent): `visible_count == total_count == 25`,
`visibility_ratio == 1.0` -- full visibility, as expected with nothing
to block the line of sight anywhere.

`max_distance=0.0` (the default) means unbounded -- every sampled
point on the surface is checked, regardless of distance from the
observer.

## `Intervisibility` -- all pairs among several points, not one observer to many targets

**This computes something different from repeatedly calling
`LineOfSight`, and different from `Viewshed`.** Given a list of
points, it checks visibility between **every pair** of them --
confirmed directly with 3 points on flat terrain: `visible_pairs=3`,
`total_pairs=3` (all 3 possible pairs among 3 points), `visibility_ratio=1.0`.

```python
from topocore.analysis.visibility import Intervisibility

inter = Intervisibility(observer_height=1.7, num_samples=100, earth_curvature=True)
result = inter.compute(points=[(10,10), (90,10), (50,90)], tin=tin)
```

```python
result.visibility_matrix   # pairwise visibility
result.visible_pairs        # int
result.total_pairs           # int
result.visibility_ratio      # visible_pairs / total_pairs
```

`.visibility_network(points, tin) -> list[tuple[int, int]]` is a
separate, related method returning only the index pairs that ARE
mutually visible (not a full matrix) -- confirmed directly, `[(0,1),
(0,2), (1,2)]` for the same 3-point, fully-visible case above.

## `VisibilityAnalysis` -- the dispatcher

```python
VisibilityAnalysis(method="los").line_of_sight(observer, target, tin)
VisibilityAnalysis(method="los").compute(observer, target, tin)  # generic dispatch, same *args pattern as other managers
```

**Confirmed directly: the dispatcher's own string value for
line-of-sight is `"los"`, not `"line_of_sight"`** -- a real,
easy-to-guess-wrong mismatch between `VisibilityMethod.LINE_OF_SIGHT`'s
own value and its matching dedicated method's name.

3 dedicated methods (`.line_of_sight()`, `.viewshed()`,
`.intervisibility()`) alongside the generic `.compute()` -- same
pattern already confirmed for `ProfileAnalysis` in
[`profile.md`](./profile.md). `VisibilityMethod` (the method-selector
constant here) is a plain class, not a `StrEnum` -- see
[`contracts.md`](./contracts.md).

## Obstruction, confirmed with a real blocking ridge

A 20-meter ridge placed directly between observer and target (both at
1.7m height, ridge running perpendicular to the line of sight) is
correctly detected as blocking:

```python
>>> los.compute(observer=(10.0, 50.0), target=(90.0, 50.0), tin=tin_with_ridge)
LOSResult(visible=False, clearance=-14.3, obstacles=(0, 1, 2, 3), ...)
```

`clearance` goes negative when blocked (confirmed: `-14.3` here,
versus a small positive value in the unobstructed flat-terrain case
above) and `obstacles` becomes a non-empty tuple (versus the empty
`()` seen in the unobstructed case) -- the exact meaning of the
values inside `obstacles` was not independently confirmed during this
audit.
