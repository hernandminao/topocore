# `CoordinateTransformer`

```python
from topocore.geodesy import CRS, CoordinateTransformer

wgs84 = CRS.from_epsg(4326)
utm17n = CRS.from_epsg(32617)
transformer = CoordinateTransformer(wgs84, utm17n)
```

## Two construction paths

```python
CoordinateTransformer(source_crs, target_crs)     # direct, most common
CoordinateTransformer.from_operation(operation)    # explicit CoordinateOperation
```

The direct constructor works for any two `CRS` objects, EPSG-backed
or custom (verified with a PROJ4-only Lambert Azimuthal CRS as
target). `from_operation()` is for the narrower, explicitly-scoped
cases below.

## `from_operation()` — support table, verified exhaustively

| `OperationType` | Supported? |
|---|---|
| `IDENTITY` | Yes |
| `HELMERT`, static (7-parameter), between two geographic CRS | Yes |
| `HELMERT`, time-dependent (14-parameter) | No — rejected, `TransformationError` |
| `HELMERT`, either CRS non-geographic | No — rejected, `TransformationError` |
| `GRID_SHIFT` | No — explicitly not yet implemented, `TransformationError` |

```python
>>> CoordinateTransformer.from_operation(time_dependent_helmert_op)
TransformationError: CoordinateTransformer.from_operation(): time-dependent
(14-parameter) Helmert transformations are not yet supported -- applying
the rate terms correctly requires an observation epoch for the data being
transformed, which transform_point()/transform_array() don't accept
today. Use a static (7-parameter) HelmertParameters instead.
```

### Helmert geometry — verified against hand-derivable math, not just "it runs"

At `(lon=0, lat=0)`, geocentric X is the local vertical direction.
Confirmed by direct execution: a pure `tx=100` translation resolves
to `h ≈ +100` (within floating-point precision) at that exact point,
and a pure `rz=1` arcsecond rotation shifts longitude by exactly
`1/3600` degrees. This matches
`topocore.analysis.profile.transversal.TransversalProfile`'s own
perpendicular-offset sign convention — both derive from the same
underlying cross-product relationship, not two independently-invented
conventions.

## `transform_point()` / `transform_array()`

```python
x, y, z = transformer.transform_point(-80.0, 25.0)          # z is None if not passed
x, y, z = transformer.transform_point(-80.0, 25.0, 100.0)   # z passed through unchanged
x_arr, y_arr, z_arr = transformer.transform_array(xs, ys)         # z_arr is None
x_arr, y_arr, z_arr = transformer.transform_array(xs, ys, zs)     # zs passed through
```

`transform_array()` validates its inputs before transforming:

```python
>>> transformer.transform_array([1.0, 2.0], [1.0, 2.0, 3.0])
ValidationError: Coordinate arrays must have identical length.

>>> transformer.transform_array([1.0, float("nan")], [1.0, 2.0])
ValidationError: Coordinate arrays contain NaN or infinite values.
```

## `transform_bbox()`

```python
minx, miny, maxx, maxy = transformer.transform_bbox((-82.0, 25.0, -80.0, 27.0))
```

```python
>>> transformer.transform_bbox((10.0, 10.0, 5.0, 5.0))   # min > max
ValidationError: min values cannot be greater than max values.
```

## What `CoordinateTransformer` does *not* do

It never applies a unit conversion and never touches `Z` beyond
passing it straight through unmodified — `z` is treated as an
independent axis, not reprojected along with `x`/`y`. If your source
and target CRS have different vertical references (e.g. ellipsoidal
vs. orthometric height), `CoordinateTransformer` itself will not
correct for that — but TopoCore does provide this, via a separate,
dedicated mechanism: `GeoidGrid`/`VerticalTransformer`/
`Workflow.transform_vertical()`, documented in
[`vertical-reference.md`](./vertical-reference.md). The two remain
deliberately independent — a caller wanting both a horizontal
reprojection and a vertical datum shift applies each explicitly,
never a single class trying to do both.
