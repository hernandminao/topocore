# Profile

Terrain profiles (elevation along a path) over any surface
implementing the `TerrainSurface` protocol -- 4 profile generators,
sharing a common `ProfileResult`, plus a dispatcher
(`ProfileAnalysis`).

```python
from topocore.analysis.profile import LongitudinalProfile

profile = LongitudinalProfile(interval=10.0)
result = profile.generate(origin=(0.0, 50.0), target=(100.0, 50.0), surface=tin)
```

Verified directly on a simple inclined plane (`Z = X * 0.1`):
`result.points[0]` is `ProfilePoint(station=0.0, x=0.0, y=50.0, z=0.0,
offset=0.0)`; `result.points[-1]` is `ProfilePoint(station=100.0,
x=100.0, y=50.0, z=10.0, offset=0.0)` -- the elevation profile
correctly matches the plane's own slope.

## `ProfileResult` -- 2 real fields, 3 derived methods that look like properties but aren't

```python
result.points        # tuple[ProfilePoint, ...] -- a real field
result.profile_type   # a real field
result.axis_length     # a real field
result.distances()    # method, not a property -- station of each point
result.elevations()   # method, not a property -- z of each point
result.offsets()       # method, not a property -- perpendicular offset of each point
```

Confirmed directly: `distances`, `elevations`, and `offsets` are
plain methods (`result.offsets()`), not properties -- calling
`result.offsets` without `()` gives you the method object itself, not
an array. Do not assume every convenience accessor here follows the
same access pattern.

## The 4 generators

```python
LongitudinalProfile(interval=1.0).generate(origin, target, surface)
    -> ProfileResult
TransversalProfile(interval=1.0, width=10.0).generate(axis_origin, axis_target, station, surface)
    -> ProfileResult
CrossSectionProfile(interval=1.0, width=10.0).generate(axis, surface)
    -> list[ProfileResult]
MultiProfile(interval=1.0, offsets=None).generate(origin, target, surface)
    -> list[ProfileResult]
```

`LongitudinalProfile` follows a single straight line from `origin` to
`target`. `TransversalProfile` cuts perpendicular to a given axis, at
one specific `station` along it. `MultiProfile` generates several
`LongitudinalProfile`-like cuts in parallel to the main axis, one per
value in `offsets` (verified directly: `offsets=[-5.0, 0.0, 5.0]`
produces exactly 3 results, each correctly shifted -- `y=45`, `y=50`,
`y=55` for an axis running along `Y=50`). `CrossSectionProfile`
generates a `TransversalProfile`-style cut at **every vertex you
provide** in its own `axis` list.

## `width` is a half-extent, not a total width

`TransversalProfile(width=20.0)` produces offsets ranging from `-20.0`
to `+20.0` -- a total span of 40 units, not 20. Verified directly:
`width=20.0` gives `ProfilePoint(offset=-20.0)` at one end and
`ProfilePoint(offset=+20.0)` at the other. Read `width` as "extent on
each side of the axis," not "total cross-section width."

## `CrossSectionProfile`'s own `interval` does not subdivide your axis for you

**This is worth stating precisely, since it is not obvious from the
constructor's own parameter name.** `CrossSectionProfile(interval=X)`
does not mean "generate a section every `X` units along the
corridor." Confirmed directly: passing `axis=[(0,50), (100,50)]` (2
vertices, a single 100-unit segment) produces exactly **2** sections
-- one per vertex given -- regardless of `interval`'s own value.
`interval` here only controls point spacing *within* each individual
transversal cut (forwarded to an internal `TransversalProfile`), not
spacing *along* the corridor. If you want sections every 25 units
over a 100-unit corridor, you must supply the axis as
`[(0,50), (25,50), (50,50), (75,50), (100,50)]` yourself --
`CrossSectionProfile` will not densify a 2-point line for you.

## Interpolation outside the surface fails loudly, not silently

Querying a point outside a `TIN`'s own triangulated extent raises
`ProfileError`, wrapping the underlying interpolation failure --
confirmed directly, this never returns `NaN` or an extrapolated guess:

```python
ProfileError: Surface interpolation failed at (125.0, 40.0).
```

This matters in practice for `CrossSectionProfile` specifically: a
corridor whose end vertices sit near the edge of your surface can
have its own perpendicular cut extend past the surface's boundary,
even if every vertex you supplied was itself inside the surface --
confirmed directly, this is a real, reachable failure mode with a
corridor near a TIN's own edge, not a hypothetical one.

## `ProfileAnalysis` -- the dispatcher

**There is no `.generate()` method here** -- `ProfileAnalysis` exposes
`.compute(*args, method=None, **kwargs)` (generic dispatch, matching
the same `*args`-forwarding pattern used by `DistanceAnalysis` and
`StatisticsAnalysis` -- see [`contracts.md`](./contracts.md)), plus 4
dedicated, individually-named methods with their own explicit keyword
arguments: `.longitudinal(origin, target, surface, *, interval=None)`,
`.transversal(...)`, `.cross_section(...)`, `.multi(...)`. Verified
directly, both routes give identical results for the same inputs:

```python
from topocore.analysis.profile import ProfileAnalysis

analysis = ProfileAnalysis(method="longitudinal")
analysis.longitudinal(origin=(0.0, 50.0), target=(100.0, 50.0), surface=tin)  # explicit, self-documenting
analysis.compute((0.0, 50.0), (100.0, 50.0), tin)                              # positional, via the generic dispatcher
```

Method selection (`ProfileMethod.LONGITUDINAL`/`TRANSVERSAL`/
`CROSS_SECTION`/`MULTI`) is a real `StrEnum` -- see
[`contracts.md`](./contracts.md) for which of `analysis`'s several
dispatcher-selector enums are and aren't real `StrEnum`s.
