# Analysis

5 independent analysis managers, confirmed to share the same real
dispatch pattern: construct with a config/method, call `.compute()`
with method-specific positional arguments.

```python
from topocore.analysis.distance.manager import DistanceAnalysis
from topocore.analysis.profile.manager import ProfileAnalysis
from topocore.analysis.visibility.manager import VisibilityAnalysis
from topocore.analysis.statistics.manager import StatisticsAnalysis
```

## `DistanceAnalysis`

```python
DistanceAnalysis(
    *,
    config: DistanceConfig | None = None,
    method: str | None = None,     # defaults to config's own default_method
    crs: CRSType | None = None,     # required only if method="geodesic"
)

analysis.compute(*args: float, method: str | None = None) -> DistanceResult
analysis.available_methods() -> tuple[str, ...]   # discover real method names
analysis.method     # property -- currently active method
```

Confirmed real, distinct methods (via `available_methods()`):
`euclidean` (2D or 3D, selected by how many coordinate values you
pass), `horizontal`, `vertical`, `slope`, `geodesic` (requires `crs`
at construction). Confirmed a real, historical fix in this exact
class: the `euclidean` branch used to forward its own `*args`
directly into `EuclideanDistance.compute()`, whose own parameter
order (`x1, y1, x2, y2, z1, z2`) differs from every sibling method's
natural point-grouped order (`x1, y1, z1, x2, y2, z2`) -- silently
computing a wrong distance for any 3D call. Fixed by reordering
before delegating; confirmed correct as of this project's own audit.

## `ProfileAnalysis`, `VisibilityAnalysis`, `StatisticsAnalysis`

**Corrected during a later revalidation pass**: each of these also
has its own real, directly-named methods -- confirmed by reading each
manager's own source with a real execution check -- not only the
generic `compute(method=...)` dispatch below. Prefer the named method
when you already know which calculation you want; it's more direct
and, for `StatisticsAnalysis`, sometimes the *only* way to reach a
particular calculation family cleanly:

```python
StatisticsAnalysis().elevation(values)   # accepts a raw array, a TIN (via its own
                                          # elevation_array()), or anything with
                                          # .elevations (e.g. a DTM)
StatisticsAnalysis().slope(dtm)          # confirmed: requires .elevations specifically --
                                          # a TIN does NOT have this attribute; passing a
                                          # TIN here raises AttributeError, not a clean error
StatisticsAnalysis().area(...)
StatisticsAnalysis().density(...)
StatisticsAnalysis().distribution(...)
StatisticsAnalysis().percentiles(...)

ProfileAnalysis().longitudinal(...)
ProfileAnalysis().transversal(...)
ProfileAnalysis().cross_section(axis: list[tuple[float, float]], surface, *, interval=None, width=None)
ProfileAnalysis().multi(...)

VisibilityAnalysis().line_of_sight(...)
VisibilityAnalysis().viewshed(...)
VisibilityAnalysis().intervisibility(...)
```

`DistanceAnalysis.available_methods()` is confirmed unique to that
one manager -- `StatisticsAnalysis`/`ProfileAnalysis`/
`VisibilityAnalysis` do not have an equivalent method; consult each
one's own source or docstring instead. See
[`end-to-end-examples.md`](./end-to-end-examples.md) for these
running against real data, including the exact numeric result of
`elevation()`/`slope()` on a real generated TIN/DTM.

The generic dispatch shape also still works, for every manager:

```python
ProfileAnalysis(...).compute(*args, method: str | None = None, **kwargs) -> ProfileResult | list[ProfileResult]
VisibilityAnalysis(...).compute(*args, method: str | None = None, **kwargs) -> VisibilityResult
StatisticsAnalysis(...).compute(*args, method: str | None = None, **kwargs) -> StatisticsResult
```

Confirmed real underlying calculators, one per family:

```text
profile:      CrossSectionProfile, LongitudinalProfile, MultiProfile
visibility:   LineOfSight, Intervisibility, Viewshed
statistics:   ElevationStatistics, AreaStatistics, DensityStatistics,
              DistributionStatistics
quality:      CloudToCloudDistance, CloudToMeshDistance, ChamferDistance,
              CompletenessAnalysis
```

Each of these can also be constructed and used directly, bypassing
the manager entirely, when you already know exactly which calculation
you want and don't need the method-name dispatch layer.

## Where to go next

- [`terrain.md`](./terrain.md) -- `TIN`/`DTM` are the typical real
  input to profile/visibility/statistics computations.
- [`surface.md`](./surface.md) -- comparing 2 surfaces and cut/fill
  volumes (`topocore.analysis.comparison`/`topocore.analysis.volume`)
  -- a related but distinct set of managers from the 4 above.
