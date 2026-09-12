# `TIN` and its construction algorithms

## 2 constructors, 2 different contracts

```python
TIN.from_points(points: tuple[Point3D, ...] | list[Point3D]) -> TIN
TIN.from_mesh(vertices: tuple[Point3D, ...], simplices: NDArray[np.int32]) -> TIN
```

`from_points()` computes a fresh Delaunay triangulation
(`algorithms.delaunay.DelaunayTriangulator`) from the given points.
`from_mesh()` accepts an already-known mesh and preserves it exactly
-- confirmed directly with the decisive case: for a perfect square,
2 diagonals are equally valid Delaunay choices, and `from_mesh()`
was confirmed to keep exactly the diagonal it was given, never
silently re-triangulating to whichever one a fresh
`scipy.spatial.Delaunay` call would have picked. `from_points()` is
untouched by `from_mesh()`'s own existence -- confirmed directly,
same points still produce the same triangulation via `from_points()`
regardless.

### `from_points()` -- validation, confirmed for all 4 documented rejection paths

- Fewer than 3 points, or an empty list -- `TriangulationError`
  ("At least three...").
- Duplicated XY coordinates (even with different Z) --
  `TriangulationError` ("Duplicated...").
- All points collinear -- `TriangulationError` ("collinear"),
  confirmed for both an axis-aligned line and an arbitrary diagonal
  (the check is matrix-rank based, not axis-specific), and confirmed
  NOT to over-trigger on a genuinely 2D but nearly-collinear point
  set (a tiny real deviation, `0.001` in one coordinate, was
  correctly accepted, both at small coordinate scale and at real
  UTM scale, ~500000/4500000 -- centering before the rank check makes
  this scale-independent).

### `from_mesh()` -- validation, confirmed for all 5 documented rejection paths

Empty vertices, empty simplices, wrong simplex shape (not `(n, 3)`),
an out-of-range or negative vertex index, and a duplicated vertex
within one triangle -- each confirmed to raise `TriangulationError`
independently. A degenerate (collinear) triangle within an otherwise
valid mesh is also rejected -- confirmed with a simple 3-point
collinear case. `from_mesh()` is confirmed to defensively copy the
given `simplices` array: mutating the array returned by
`tin.simplices` afterward does not affect the caller's own original
array.

## Core accessors -- confirmed by direct execution against known geometry

`vertex_count`/`triangle_count`, `vertices` (the original `Point3D`
tuple, order preserved exactly -- confirmed with tuple identity, not
just equal values), `simplices` (vertex-index triples), `neighbors`
(adjacent triangle indices per edge, `-1` for a hull boundary --
confirmed symmetric: if triangle A lists B as a neighbor, B lists A
back). `bounds` returns the XY bounding box as a plain property (not
a method) -- confirmed `(0.0, 0.0, 10.0, 10.0)` for a known 4-corner
square.

`elevation_array()`/`xy_array()`/`vertex_array()` are **plain
methods, not properties** -- confirmed directly, and this is the
exact contract whose mismatch with `analysis.protocols.PointCloudData`
(there declared as `@property`) was found and fixed during
`16-analysis`'s own audit. `TIN` itself was never changed for that
fix -- it was already correct; only the `Protocol` was. This session
re-confirmed the real `TIN` source still matches: `elevation_array()`
returns a 1D array of every vertex's `z` in vertex order;
`xy_array()` returns an `(n, 2)` array; `vertex_array()` returns the
full `(n, 3)` XYZ array both `NearestInterpolator`/`IDWInterpolator`
use for their own vectorized `interpolate_many()`.

`triangle(index)` returns a `Triangle` (see [`utilities.md`](./utilities.md)
for the `Triangle`/`Edge`/`Breakline` model definitions) built from
that simplex's own 3 vertices -- confirmed the vertex objects are
identical to the ones in `vertices`, not copies. `triangles()`
iterates every triangle this way; `triangle_vertices(index)` returns
just the 3 `Point3D` without building a `Triangle` wrapper. `edges`
is confirmed cached -- 2 successive accesses return the identical
object (`is`, not just `==`), not recomputed each time.

`len(tin)` is `triangle_count`; `tin[i]` is `triangle(i)`; `tin` is
directly iterable over its own triangles -- all confirmed directly,
including that `len(tin)` and iteration agree with `triangle_count`.
`repr(tin)` includes vertex, triangle, and edge counts, confirmed by
direct string inspection.

## `find_triangle()` / `interpolate()` / `contains()` / `locate()`

`find_triangle(x, y)` returns the containing triangle's index, or
`-1` if the point lies outside the TIN's convex hull. As of PR21.5,
this is backed by a uniform-grid spatial index rather than a
brute-force `O(triangle_count)` scan -- **confirmed correct, not
merely faster**, by this project's own dedicated regression suite:
an independent brute-force reimplementation of the exact pre-PR21
algorithm is run against the indexed version over thousands of
random query points (including points outside the hull, on a shared
triangle edge, exactly on a TIN vertex, and exactly on the spatial
index's own internal cell boundaries) -- 0 mismatches. The one
branch of the index-building code confirmed structurally unreachable
from either public constructor (`triangle_count == 0` or a
zero-area bounding box) is explicitly documented as such in the
project's own test suite, with the reasoning spelled out: both
constructors already reject the inputs that would produce it.

`interpolate(x, y)` uses barycentric interpolation directly (not
delegated through a separate interpolator class) -- confirmed exact
for a tilted plane (`z=5.0` at the centroid of a plane tilted `0` to
`10`, not merely "close"), constant for a flat plane, and raises
`ValueError` ("outside TIN") for a point beyond the hull.
`contains(x, y)` and `locate(point)` both confirmed to delegate to
`find_triangle()` internally, agreeing with it exactly in every case
checked.

## `algorithms.delaunay.DelaunayTriangulator` -- the backend `from_points()` calls

A separate, lower-level module (see [`contracts.md`](./contracts.md)
for why it's deliberately not part of the public API). Wraps
`scipy.spatial.Delaunay` (`qhull_options="Qbb Qc Qz"`), converting
`QhullError` into `TriangulationError`. Its own validation
(`_validate_duplicate_points`, `_validate_collinear_points`) is what
`TIN.from_points()` actually relies on -- confirmed identical
rejection behavior between calling `DelaunayTriangulator.triangulate()`
directly and via `TIN.from_points()`. `compute_bbox()` and
`validate_result()` are small standalone utilities; `validate_result()`
is confirmed, by search, to have no real caller anywhere beyond its
own tests -- its own docstring already states it's "primarily
intended for internal use and unit testing," so this is not treated
as a finding, unlike the orphaned functions in
[`limitations.md`](./limitations.md) that make no such admission.

## `algorithms.constrained_delaunay.ConstrainedDelaunayTriangulator` -- breakline-aware triangulation

Used by `breaklines.apply_breaklines()` (see
[`filters-breaklines.md`](./filters-breaklines.md)), not called
directly by ordinary `TIN` construction. Given a base point set and
a list of `(index_a, index_b)` constraint edges, forces each
constraint into the final mesh and re-legalizes the rest of the
triangulation around it. Confirmed by this project's own dedicated
test suite (geometric invariants, not just "ran without error"):

- A constraint requiring a genuine edge flip is confirmed present in
  the final mesh, and confirmed to **survive** the global
  legalization sweep that runs afterward (the sweep is confirmed to
  skip constrained edges specifically, not just coincidentally leave
  this one alone -- verified with enough additional points that the
  sweep has real work to do elsewhere).
- Multiple, non-conflicting constraints are all honored
  simultaneously.
- A constraint that's already a natural edge in the base
  triangulation is a confirmed no-op.
- Determinism: identical input produces byte-for-byte identical
  output.
- Documented error paths, each confirmed: an out-of-range constraint
  index, a self-loop (a point connected to itself), a constraint
  passing exactly through an existing third vertex, and 2 constraints
  that geometrically cross each other (the 2 diagonals of a square)
  -- all raise `BreaklineError`, not a silent, geometrically wrong
  result.

The resulting `TIN` is constructed via `TIN`'s own private
`_result: DelaunayResult` constructor -- confirmed by real execution
that `ConstrainedDelaunayTriangulator.triangulate()`'s own return
value is structurally accepted there, the same way `from_points()`/
`from_mesh()` themselves are built.
