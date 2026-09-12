# Interpolation — 4 algorithms, 1 facade

## `BaseInterpolator` -- the real, confirmed contract

`interpolate(x, y) -> float` and `interpolate_many(x, y) -> ndarray`.
All 4 concrete interpolators below are confirmed `issubclass()` of
this ABC (see [`contracts.md`](./contracts.md)) -- this project's own
regression suite exists specifically because, in an earlier version
(TD-001), none of the 4 inherited from it at all; that gap is fixed
and re-confirmed here. Every `interpolate_many()` implementation is
confirmed, by this project's own dedicated test suite, to match
calling `interpolate()` once per point exactly -- not approximately,
not "usually agrees" -- for all 4 classes, over the same query
points.

## `LinearInterpolator` / `BarycentricInterpolator` -- mathematically identical, confirmed

`LinearInterpolator` wraps `BarycentricInterpolator` internally and
delegates every one of its own methods (`interpolate()`,
`interpolate_triangle()`, `interpolate_many()`, `interpolate_point()`)
directly to it -- confirmed by direct execution that both give
byte-identical results for the same query, not merely "close." This
is not 2 independent implementations that happen to agree: `Linear`
exists as a separately-named class purely so a caller can state
"I want linear interpolation" explicitly, matching its own class
docstring's stated reasoning (each TIN triangle defines a plane, so
linear and barycentric interpolation are the same computation).

`BarycentricInterpolator.interpolate(x, y)`: locates the containing
triangle via `TIN.find_triangle()`, raises `InterpolationError` if
outside the hull, then computes barycentric weights
(see [`utilities.md`](./utilities.md) for `weights.barycentric_weights()`)
and blends the 3 vertices' `z` values. `interpolate_triangle(index, x, y)`
does the same directly against a known triangle index, skipping the
point-location step -- confirmed to raise `InterpolationError` if the
weights fall outside the triangle for the *given* index (a caller's
own responsibility to supply a triangle that actually contains the
point).

`interpolate_many()` here is a plain Python loop, not a vectorized
NumPy computation -- confirmed deliberate, not an oversight: its own
docstring explains that `find_triangle()`'s own spatial index still
requires locating each query point individually, so there's no
broadcasting shortcut available the way there is for `Nearest`/`IDW`
(which only ever need vertex coordinates, never triangle
containment).

## `IDWInterpolator` -- Inverse Distance Weighting

```python
IDWInterpolator(tin, *, power: float = 2.0, epsilon: float = 1e-12)
```

`power` must be strictly positive -- confirmed rejected with
`ValueError` for `power <= 0`. Uses **every** TIN vertex for each
query (no spatial subsetting) -- confirmed directly by the
implementation, and documented in the class's own docstring as a
known scaling characteristic ("later PRs may introduce KD-Tree
acceleration"). A query exactly at (or within `epsilon` of) an
existing vertex short-circuits to that vertex's own `z` exactly --
confirmed `10.0` exactly, not a weighted blend that happens to be
close to it. `interpolate_many()` is a genuinely vectorized
computation (not a Python loop), confirmed to reproduce the same
exact-match short-circuit for the corresponding rows, and confirmed
to handle a batch containing BOTH an exact-match row and an ordinary
weighted row correctly, independently -- the exact-match case does
not corrupt the ordinary row's own result or vice versa.

## `NearestInterpolator` -- nearest-vertex lookup

No interpolation math at all: returns the `z` of whichever TIN
vertex is planar-closest to the query point. Ties resolve to the
FIRST vertex encountered in `tin.vertices` order -- confirmed for
both the scalar path (Python's own `min()`) and the vectorized
`interpolate_many()` path (`np.argmin()`), and confirmed the 2 paths
agree with each other on an exact, constructed tie (2 vertices
equidistant from the query point).

## `TerrainInterpolator` -- the facade

```python
TerrainInterpolator(tin, *, method: InterpolationMethod = InterpolationMethod.LINEAR, power: float = 2.0)
```

Constructs all 4 concrete interpolators up front (so switching
`.method` later never reconstructs anything), and dispatches
`.interpolate()` to whichever one `.method` currently selects.
Confirmed NOT itself a `BaseInterpolator` subclass (see
[`contracts.md`](./contracts.md)) -- architecturally correct, since
it delegates rather than implements interpolation itself.

### 2 real, historical dispatch bugs -- both now fixed, confirmed by this audit

**Bug 1 (already fixed before this audit, re-confirmed still fixed)**:
`enums.py` and `interpolation.py` used to each declare their own,
separately-defined `InterpolationMethod` (same name, different
members -- one had `NATURAL_NEIGHBOR`, never implemented anywhere;
the other had `NEAREST`, which is). Since the dispatch compared
method values with `is` rather than `==`, passing the "natural"
import (`from topocore.terrain.enums import InterpolationMethod`,
alongside every other Terrain enum) silently fell through to
`NEAREST` instead of the intended method -- confirmed by this
project's own regression test with a real, asymmetric TIN: `7.5`
(correct `LINEAR`) vs. `10.0` (silently `NEAREST`) for the identical
query. Fixed by consolidating to one `InterpolationMethod`
definition, in `enums.py`; `interpolation.py` now imports it rather
than declaring its own. Confirmed still fixed: `enums.InterpolationMethod`
and `interpolation.InterpolationMethod` are confirmed to be the
exact same class object (`is`, not just equal), not 2 classes that
happen to share values.

**Bug 2 (found and fixed during this specific audit)**: that first
fix addressed the specific duplicate-enum symptom, but not this
dispatch's own structural cause -- it still compared with `is`. Since
`InterpolationMethod` is a `StrEnum`, `"linear" == InterpolationMethod.LINEAR`
is `True` while `"linear" is InterpolationMethod.LINEAR` is `False`
-- so the natural, easy mistake of writing `method="linear"` (a
plain string, not the enum member) silently produced `NEAREST`'s own
result instead, with no error. Confirmed directly, before the fix:
`TerrainInterpolator(tin, method="linear").interpolate(0.5, 0.5)`
gave `10.0`; after the fix, `7.5` (correct). The dispatch now compares
with `==` and raises `InterpolationError` explicitly for any value
that isn't one of the 4 known `InterpolationMethod` members, rather
than silently defaulting to `NEAREST` for anything unrecognized. See
[`validation.md`](./validation.md) for the complete account.

### Confirmed unaffected by either fix

`.method` can be changed after construction (confirmed: switching
from `LINEAR` to `NEAREST` on the same query gives a different,
correct result for each). `.interpolate_point()`/`__call__` are
confirmed to match `.interpolate()` exactly. `power` is confirmed to
actually affect `IDW`'s own result at an off-center query point
(default power vs. `power=10.0` give genuinely different answers, as
expected -- the equidistant-from-all-corners case was deliberately
avoided in this check, since power cannot matter there regardless of
its value). `LINEAR` and `BARYCENTRIC`, dispatched through the
facade, are confirmed to still agree exactly with each other,
matching the direct-class finding above.
