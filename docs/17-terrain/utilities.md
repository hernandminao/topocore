# Shared utilities

## `weights.py` -- the low-level math `barycentric.py`/`idw.py` build on

`triangle_area2()`/`triangle_area()`: signed (twice) and unsigned
triangle area, confirmed exact for a known right triangle (legs
`1`/`1` -> area `0.5`). `barycentric_weights(x, y, p1, p2, p3)`:
confirmed exact at each of the 3 vertices (weights `(1,0,0)`,
`(0,1,0)`, `(0,0,1)`), at the centroid (`(1/3, 1/3, 1/3)`), always
summing to `1.0`, and confirmed to reproduce a known linear plane's
own Z exactly when used to blend it. `inside_triangle(weights,
tolerance=1e-9)`: confirmed `True`/`False` correctly for an interior
vs. exterior point, with a small negative-weight tolerance to absorb
floating-point noise at a triangle's own edge. `point_distance()`:
plain Euclidean distance, confirmed with a 3-4-5 case.
`inverse_distance_weight(distance, power)`: confirmed `inf` at zero
distance (the short-circuit `IDWInterpolator` itself relies on), and
confirmed to decrease monotonically with distance.

**Found and fixed during this audit**: `barycentric_weights()`'s own
degenerate-triangle check used to raise a plain `ValueError`,
inconsistent with the rest of this package's own domain-exception
convention -- the identical underlying problem (a degenerate
triangle) raises `TriangulationError` via `TIN.from_points()` and
`TerrainValidationError` via `validate_triangle()` elsewhere in this
same package. Now raises `TerrainValidationError`, matching
`validate_triangle()`'s own exception type for what is, at heart,
the same check. Confirmed directly: the exception type changed; the
valid-triangle case is confirmed byte-for-byte unaffected. See
[`validation.md`](./validation.md).

## `nodata.py` -- the shared `NaN`-as-NoData vocabulary

`is_nodata(value)` is exactly `not math.isfinite(value)` -- confirmed
`True` for `NaN`, `+inf`, and `-inf` alike, not only `NaN`.
`valid_mask()`/`nodata_mask()` are confirmed exact complements of
each other over a mixed array. `valid_count()`/`nodata_count()`
confirmed to sum to the array's own total size. `replace_nodata(array,
value)`/`fill_nodata(array, value=0.0)` (the latter simply calling
the former with a default) confirmed to touch only NoData cells,
never mutating the caller's own original array.

**Confirmed by search, not assumed**: none of this module's own 7
functions has a real caller anywhere in the codebase beyond its own
tests. `DEFAULT_NODATA` (`float("nan")`) is similarly unreferenced
anywhere real. This is the same category as `constants.DEFAULT_NO_DATA`
(a separate, also-unused declaration of the identical concept) --
`Raster.nodata` (a real dataclass field, defaulting to `NaN`) is the
one place this convention is actually enforced in working code. See
[`limitations.md`](./limitations.md).

## `sampling.py` -- `RasterSampler`, independent of `Raster.elevation()`

`RasterSampler.nearest(raster, x, y)`/`.cell(raster, x, y)` -- each
confirmed to validate `raster.contains(x, y)` *before* computing a
row/column and indexing, raising `ValueError` for an out-of-bounds
coordinate rather than indexing at all. This is the exact pattern
`Raster.elevation()` itself was missing until this audit's own fix
(see [`grid-raster-cell.md`](./grid-raster-cell.md) and
[`validation.md`](./validation.md)) -- `RasterSampler` never had the
bug; it's the class whose own existing, correct pattern the fix was
modeled on. `.index(grid, x, y)` converts a coordinate to
`(row, column)`, same bounds check. `.coordinate(grid, row, column)`
is a thin pass-through to `grid.coordinate()`.

`.bilinear(raster, x, y)`: confirmed to reproduce a genuinely linear
surface (`z = x + 2y`) exactly at non-grid-aligned query points, not
merely approximately -- the defining property bilinear interpolation
must have on a truly planar surface. Confirmed to return `NaN`
whenever any of the 4 surrounding cells is itself NoData (a single
`NaN` neighbor is enough), rather than silently ignoring it or
propagating a partial result.

## `conversion.py` -- the one bridge from `PointCloud`

```python
pointcloud_to_points(cloud: PointCloud) -> tuple[Point3D, ...]
```

Converts every point of a `PointCloud` into `Point3D`, preserving
chunk order then within-chunk order -- confirmed directly with a
2-chunk cloud, checked against the exact expected X sequence. Raises
`ConversionError` for an empty cloud, confirmed directly. This is
deliberately NOT owned by any orchestrator -- its own module
docstring states it's domain logic belonging to `terrain`, reusable
independently of how the caller obtained the `PointCloud`.

## `validation.py` -- confirmed real callers vs. confirmed orphans

`validate_resolution()`/`validate_interval()` (both `_validate_positive()`
under the hood: rejects non-numeric, non-finite, or non-positive
values -- confirmed for `0.0`, negative, `NaN`, and `inf`),
`validate_points()`, `validate_triangle()` (see
[`slope-aspect-hillshade.md`](./slope-aspect-hillshade.md) for this
audit's own fix extending it to collinearity), and `validate_breakline()`
are confirmed to have real callers (`Grid.__post_init__`,
`ContourGenerator`, `_geometry.oriented_normal()`, `breaklines.py`).
`validate_grid_definition()` and `validate_tin()` are confirmed, by
search, to have zero callers anywhere beyond their own declaration
and tests -- see [`contracts.md`](./contracts.md) and
[`limitations.md`](./limitations.md).

## `models.py` -- the plain geometry types

`Edge` (`start`, `end`, `is_breakline`): `.length` is full 3D
distance including `Z` -- confirmed with a 3-4-5-equivalent case
including a Z component. `Triangle` (`p1`, `p2`, `p3`): `.area`
(projected XY, confirmed independent of vertex winding order),
`.centroid`, `.bbox` (confirmed exact for a known asymmetric
triangle). `Breakline` (`id`, `name`, `points`, `type`):
`.is_closed` requires the first and last point to match AND more
than 2 points -- confirmed a 2-point "loop" (first==last trivially)
is correctly NOT considered closed. `ContourLine` (`elevation`,
`points`, `closed`): `.vertex_count`.

`GridDefinition` is confirmed, independently of its own already-
self-documented legacy status (see [`contracts.md`](./contracts.md)),
to be internally correct on its own terms -- `.width`/`.height`
confirmed exact for a known extent/resolution -- it is simply
disconnected from the rest of the package's real, working code path,
not internally broken.
