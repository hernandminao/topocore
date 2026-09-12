# Distance

5 distance calculations, sharing a common `DistanceResult` return
type, plus a dispatcher (`DistanceAnalysis`) that selects between
them by name -- with one contract discrepancy between the dispatcher
and one of the concrete classes that must not be confused.

```python
from topocore.analysis.distance import EuclideanDistance

result = EuclideanDistance().compute(0.0, 0.0, 3.0, 4.0, z1=0.0, z2=5.0)
result.value      # 7.0710678118654755 -- verified: 5*sqrt(2), the 3-4-5-in-Z right triangle
result.method      # 'euclidean'
result.dimension   # '3d'
```

`DistanceResult` has exactly 3 fields: `value`, `method`, `dimension`.

## The 5 calculations, verified against hand-computable results

```python
EuclideanDistance(*, dimension="3d").compute(x1, y1, x2, y2, z1=0.0, z2=0.0)
HorizontalDistance.compute(x1, y1, x2, y2)                       # static method -- XY only
VerticalDistance.compute(z1, z2)                                  # static method -- Z only
SlopeDistance().compute(x1, y1, z1, x2, y2, z2)
GeodesicDistance(crs).compute(lon1, lat1, lon2, lat2)
```

Verified directly, all against the same `(0,0,0)` -> `(3,4,5)` point
pair (or its 2D projections): `EuclideanDistance` gives `7.0711`
(`5*sqrt(2)`); `HorizontalDistance` gives `5.0` (XY only, Z ignored
entirely); `VerticalDistance` gives `5.0` (Z only); `SlopeDistance`
matches `EuclideanDistance` exactly for the same 3 coordinates, and
additionally exposes `slope_angle()`/`slope_gradient()`/
`from_horizontal_and_vertical()` -- `slope_angle(0,0,0, 3,4,5)` gives
exactly `45.0` degrees for a horizontal run of `5.0` and a vertical
rise of `5.0`.

**`HorizontalDistance.compute()` and `VerticalDistance.compute()` are
`@staticmethod`s** -- confirmed directly -- callable directly on the
class without constructing an instance first, unlike the other 3.

## Invalid input -- rejected explicitly, never silently coerced

Every one of the 5 validates coordinate finiteness before computing.
Confirmed directly:

```python
>>> EuclideanDistance().compute(0.0, 0.0, float("nan"), 0.0)
DistanceError: Coordinate must be finite; got nan.
>>> EuclideanDistance().compute(0.0, 0.0, float("inf"), 0.0)
DistanceError: Coordinate must be finite; got inf.
```

## `GeodesicDistance` requires a real `CRS`, and validates coordinate range

```python
from topocore.geodesy import CRS
GeodesicDistance(CRS.from_epsg(4326)).compute(-74.0721, 4.7110, -75.5636, 6.2476)
```

Verified directly against
[`../14-geodesy/geodesic.md`](../14-geodesy/geodesic.md)'s own
`GeodesicCalculator`: the exact same Bogotá-to-Medellín distance
(`237043.87134926795` m) comes back from both -- `GeodesicDistance`
is built on the same underlying geodesic machinery, not a separate or
duplicated implementation.

Passing coordinates that aren't valid longitude/latitude (e.g. large
projected-CRS-scale values) is rejected, not silently miscomputed --
confirmed directly:

```python
>>> GeodesicDistance(CRS.from_epsg(3116)).compute(1000.0, 2000.0, 1001.0, 2001.0)
DistanceError: Longitude out of range: 1000.0
```

This catches values clearly outside the valid range; it does not
independently verify that the `CRS` you passed is itself geographic
-- a projected CRS whose numeric values happen to fall inside valid
longitude/latitude bounds is not something this was confirmed to
catch.

## `DistanceAnalysis` -- the dispatcher, and a real argument-order discrepancy

```python
from topocore.analysis.distance import DistanceAnalysis

DistanceAnalysis(method="euclidean").compute(*args, method=None)
```

An unrecognized `method`, whether at construction or per-call, is
rejected the same way in both places:

```python
>>> DistanceAnalysis(method="not_a_method")
DistanceError: Unsupported distance method 'not_a_method'. Available: ('euclidean', 'geodesic', 'horizontal', ...)
```

`DistanceAnalysis` also exposes 3 members beyond `compute()`, not
reachable through the `method=` dispatch:

```python
DistanceAnalysis.available_methods() -> tuple[str, ...]       # static method -- same 5 strings accepted by method=
DistanceAnalysis(...).geodesic         # property -- the underlying GeodesicDistance instance, or None if not constructed with a crs
DistanceAnalysis(...).gradient(x1, y1, z1, x2, y2, z2) -> float   # slope gradient, as a percentage -- calls SlopeDistance.slope_gradient() directly
DistanceAnalysis(...).angle(x1, y1, z1, x2, y2, z2) -> float      # slope angle, in degrees -- calls SlopeDistance.slope_angle() directly
```

Verified directly: `.gradient(0,0,0, 100,0,10)` gives exactly `10.0`
(a 10-unit rise over a 100-unit run is a 10% grade);
`.angle(0,0,0, 100,0,10)` gives `5.7106` degrees
(`atan(10/100)`). Both take the same natural, point-grouped
`(x1,y1,z1,x2,y2,z2)` order as `compute()`'s own 6-argument form (see
below) -- **not** `EuclideanDistance`'s own raw order. Both return a
plain `float` directly, not a `DistanceResult` -- a different return
shape from every method reachable via `compute()`/`method=`,
reflecting that these 2 are a convenience shortcut to
`SlopeDistance`'s own static helpers, not additional dispatch targets.

**`DistanceAnalysis.compute()`'s own 6-argument order is
`(x1, y1, z1, x2, y2, z2)` -- the natural, point-grouped order.
`EuclideanDistance.compute()`'s own order is
`(x1, y1, x2, y2, z1, z2)` -- do not assume they match.** Confirmed
directly, and confirmed deliberate: a real, severe defect existed
here previously (before this documentation effort's own audit,
already fixed in the code you have) where `DistanceAnalysis` forwarded
6 arguments straight through to `EuclideanDistance` unreordered,
silently computing a wrong distance with no error at all (a 3-4-12
right triangle, expected distance `13.0`, silently returned `8.544`
instead). The fix adopted the natural point-grouped order as
`DistanceAnalysis`'s own public contract, matching what
`SlopeDistance.compute()`/`GeodesicDistance.compute()` already
expose. This means the 2 classes now have **genuinely different**
argument orders for what looks like the same 6 numbers -- confirmed
directly with the exact 3-4-12 triangle:

```python
>>> EuclideanDistance().compute(0.0, 0.0, 3.0, 4.0, z1=0.0, z2=12.0)   # (x1,y1,x2,y2,z1,z2)
DistanceResult(value=13.0, ...)
>>> DistanceAnalysis(method="euclidean").compute(0.0, 0.0, 0.0, 3.0, 4.0, 12.0)  # (x1,y1,z1,x2,y2,z2)
DistanceResult(value=13.0, ...)
```

Both give the correct `13.0` -- but only because each was called with
**its own** expected order. Passing `EuclideanDistance`'s own argument
order to `DistanceAnalysis.compute()` (or vice versa) silently
computes a different, wrong point pair, with no error, since both
signatures accept exactly 6 floats positionally. There is no runtime
check that could catch this mistake -- read this section again before
choosing which of the 2 you call, if you're ever unsure.

**Only the `euclidean` branch validates its own argument count.**
Confirmed directly: `compute(1.0, 2.0, 3.0, method="euclidean")` (3
args, neither 4 nor 6) raises a clear
`DistanceError: Euclidean requires 4 or 6 coordinates.`; the same
kind of mistake against `horizontal`/`vertical`/`slope`/`geodesic`
instead raises a raw Python `TypeError` from the underlying class's
own positional-argument checking (e.g. `HorizontalDistance.compute()
missing 1 required positional argument: 'y2'`), never translated into
a `DistanceError`. See
[`contracts.md`](./contracts.md#every-manager-leaks-a-raw-typeerror-for-wrongmissing-arguments)
for this same pattern confirmed across every manager in this section,
not unique to `DistanceAnalysis`.

## `DistanceConfig.ellipsoid` -- a known limitation, not a usable setting

```python
DistanceConfig(default_method="euclidean", ellipsoid="WGS84", default_precision=6)
```

`ellipsoid` is validated at construction (rejects an empty string) but
**is never read or consumed anywhere else in this codebase** --
confirmed directly by searching every use of `.ellipsoid` in
`topocore.analysis`. Setting it to anything has no effect on any
computation; `GeodesicDistance`'s own ellipsoid comes entirely from
whatever `CRS` you construct it with, not from this field. Treat this
as a known limitation of `DistanceConfig`, not a working way to
configure the ellipsoid used for geodesic calculations.
