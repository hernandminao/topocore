# Statistics

Descriptive statistics over elevation, slope, area, point density,
and general scalar distributions -- 5 computers, sharing no common
base result type (each has its own dataclass), plus a dispatcher
(`StatisticsAnalysis`).

## `ElevationStatistics`

```python
from topocore.analysis.statistics import ElevationStatistics

ElevationStatistics.compute(values)      # raw NDArray
ElevationStatistics.from_tin(tin)         # from a TIN
ElevationStatistics.from_dtm(dtm)         # from a GriddedSurface
```

All 3 are `@staticmethod`s. Verified directly on a 3-vertex TIN with
elevations 0/10/20: `minimum=0.0`, `maximum=20.0`, `mean=10.0`,
`median=10.0`, `count=3`.

**A real, confirmed defect existed in `from_tin()` before this
documentation effort's own audit** (see
[`validation.md`](./validation.md) for the full account): it passed
`tin.elevation_array` without calling it -- a bound method, not the
array -- raising `TypeError` unconditionally. Already fixed in the
code you have; `from_tin()` now works correctly, verified above. The
same defect existed a second time in `StatisticsAnalysis`'s own
duck-typed elevation dispatch, also fixed.

**This also surfaced a related contract defect, found and fixed as
part of this same audit**: `from_tin()`'s own parameter is typed as
`PointCloudData`, which used to declare `elevation_array` as a
`@property`, while `TIN` -- the sole concrete implementation anywhere
in the codebase -- exposes it as a plain method. `PointCloudData` is
now corrected to type `elevation_array` (and `xy_array`, the same
mismatch, found on the same protocol -- see below) as a method,
matching `TIN`'s own real, working, tested behavior. See
[`contracts.md`](./contracts.md) for the full account, including why
`TIN` itself was never the one considered wrong here.

## `SlopeStatistics`

```python
from topocore.analysis.statistics import SlopeStatistics

stats = SlopeStatistics(num_bins=10)
stats.slope_array(elevations, resolution)        # raw slope-in-degrees array
stats.compute_from_array(elevations, resolution)  # SlopeStats
stats.compute_from_dtm(dtm)                        # SlopeStats
```

Slope is computed via `np.gradient` (finite differences) ->
`np.hypot` -> `np.arctan` -> `np.degrees`. Requires at least a 3x3
elevation grid -- confirmed directly, smaller grids raise
`StatisticsError`.

**A real, confirmed defect existed here too, now fixed** (see
[`validation.md`](./validation.md)): a genuinely uniform slope --
computed via real floating-point arithmetic, not constructed
artificially -- differs across the grid by an amount as small as
`~1e-15`, not exactly `0`. `np.histogram()` only self-widens a
degenerate range when the min and max are **exactly** equal; this
tiny nonzero spread crashed with `"Cannot create N finite-sized
bins"`. The fix detects a near-degenerate range using a *relative*
tolerance (not an absolute epsilon, since slope magnitudes vary
widely) and widens the histogram's own binning range explicitly --
`SlopeStats.minimum`/`.maximum` are still reported exactly as
computed; only the internal histogram bin edges are stabilized.

## `AreaStatistics`

```python
from topocore.analysis.statistics import AreaStatistics

AreaStatistics.triangle_area(p1, p2, p3)            # real 3D surface area
AreaStatistics.projected_triangle_area(p1, p2, p3)   # flat XY (horizontal) area
AreaStatistics().compute(tin)                         # AreaStats over a whole TIN
```

Verified directly with an inclined triangle (base 10, height 10 in
XY, one vertex raised 10 units in Z): `projected_triangle_area` gives
exactly `50.0` (plain 2D triangle area, ignoring the slope entirely);
`triangle_area` gives `70.71` (the real, larger surface area of the
tilted triangle). Confusing the two silently gives a plausible but
wrong number for any sloped terrain -- always be certain which one a
given computation needs.

`AreaStatistics().compute(tin)` returns both, plus `count`, on an
`AreaStats` result -- verified directly on a flat, non-tilted
triangle (base 10, height 10, no slope), where `total_area` came back
`50.0` exactly, matching the expected flat-triangle area.

## `DensityStatistics`

```python
from topocore.analysis.statistics import DensityStatistics

DensityStatistics(resolution=1.0).compute(points)   # points: list[tuple[float,float]] | NDArray -- XY only, NOT a PointCloud
DensityStatistics(resolution=1.0).compute_from_tin(tin)   # from a TIN's own XY coordinates
```

`compute()` **takes raw XY coordinates, not a `PointCloud`.** Confirmed
directly: passing a `PointCloud` object raises `TypeError` inside its
own internal coordinate conversion -- extract the XY array yourself
first. `compute_from_tin()` accepts a `TIN` directly, extracting its
XY coordinates internally.

**A related contract defect was found and fixed in `compute_from_tin()`
during this same audit**: it called `tin.xy_array` without invoking
it -- a bound method, not the array -- raising `TypeError`
unconditionally, exactly the same underlying cause as the
`elevation_array` defects above (`PointCloudData.xy_array` was
declared as a `@property`, while `TIN`'s own implementation is a
method). Fixed together with `elevation_array` by correcting
`PointCloudData` -- see [`contracts.md`](./contracts.md). Verified
directly: `compute_from_tin()` now works correctly for a real `TIN`.

## `DistributionStatistics`

```python
from topocore.analysis.statistics import DistributionStatistics

dist = DistributionStatistics(num_bins=10)
dist.compute(values)                    # DistributionStats -- histogram, skewness, kurtosis
dist.percentiles(values, [25, 50, 75])   # dict[float, float]
dist.quantile_ranges(values)              # box-plot-style whiskers
```

Requires at least 4 finite values -- confirmed directly, fewer raises
`StatisticsError`. Verified directly: `percentiles([1,2,3,4,5,100],
[25,50,75])` gives `{25.0: 2.25, 50.0: 3.5, 75.0: 4.75}`, matching
NumPy's own linear-interpolation convention exactly.

**A real historical defect was found and fixed here in an earlier PR
(not this documentation effort's own audit, but worth knowing):**
skewness/kurtosis used to mix sample standard deviation (Bessel's
correction) in the denominator with population-style moments in the
numerator -- a non-standard hybrid matching neither `scipy.stats`
convention. For a small sample (n=6), this flipped the reported
distribution shape's own sign entirely. Now consistently
population-based throughout.

**This class also handles an exactly-uniform input correctly**
(`np.full(N, constant)` -- confirmed directly, `np.histogram()`'s own
automatic range-widening covers the exact-equality case on its own).
Whether it handles the *near*-degenerate case (a genuinely computed,
not artificially constructed, near-uniform array -- the actual defect
found and fixed in `SlopeStatistics`, see above) was not verified
during this audit and is not assumed to be safe -- see
[`limitations.md`](./limitations.md).

## `StatisticsAnalysis` -- the dispatcher

```python
StatisticsAnalysis(method="elevation").compute(tin)
```

Dispatches to the 5 computers above by name, via `*args`/`**kwargs`
forwarding (same generic pattern as `DistanceAnalysis`/
`ProfileAnalysis`). `StatisticsMethod` (the method-selector constant)
is a plain class, not a `StrEnum` -- see [`contracts.md`](./contracts.md).

### `.slope()` -- now a full facade, matching `.elevation()`'s own established pattern

```python
StatisticsAnalysis(method="slope").slope(elevations, resolution=1.0)   # raw array -- resolution required
StatisticsAnalysis(method="slope").slope(dtm)                            # GriddedSurface -- resolution taken from the DTM itself
```

**Found and fixed during this project's own manager/dispatcher
audit**: `.slope()` previously accepted only a `GriddedSurface`,
calling `SlopeStatistics.compute_from_dtm()` internally no matter
what was passed -- a narrower facade than `SlopeStatistics` itself
(which publicly supports both a raw array and a gridded surface), and
inconsistent with this same manager's own `.elevation()` method,
which already branches on input type via `isinstance`/`hasattr`
checks. Passing a raw array used to raise a bare `AttributeError`
(`'numpy.ndarray' object has no attribute 'elevations'`), not a
`StatisticsError`.

Corrected to branch on input type exactly like `.elevation()` does:
a raw `NDArray` now requires an explicit `resolution` argument
(matching `SlopeStatistics.compute_from_array()`'s own contract,
since a bare array carries no cell-size information of its own) and
raises a clear `StatisticsError` if omitted; anything else is passed
to `compute_from_dtm()` unchanged, exactly as before this fix.
Verified directly: the raw-array path now works
(`.slope(elevations, resolution=1.0)`), the missing-`resolution` case
now raises `StatisticsError` instead of `AttributeError`, and the
pre-existing `GriddedSurface` path is byte-for-byte unaffected
(confirmed against calling `SlopeStatistics.compute_from_dtm()`
directly on the same object).
