# Filters and Breaklines

## `LaplacianSmoother` / `laplacian_smooth()` -- Z-only smoothing

```python
laplacian_smooth(tin, *, iterations: int = 1, factor: float = 0.5) -> TIN
```

Nudges each vertex's elevation toward the average `Z` of its
triangulation neighbors (`_vertex_adjacency()`, built once from
`tin.simplices`), for `iterations` passes, blended by `factor` each
pass (`new_z = z + factor * (mean_neighbor_z - z)`). `iterations`
must be `>= 1`; `factor` must be in `(0, 1]` -- both confirmed
rejected with `TerrainValidationError` outside range, both confirmed
accepted at the `factor=1.0` boundary.

Confirmed exact against a fully-controlled adjacency graph (a
center vertex with exactly 4 known neighbors): `factor=1.0` snaps
the center exactly to the neighbor mean; `factor=0.5` blends exactly
halfway. Confirmed X/Y are left bit-for-bit unchanged -- only `Z`
moves. Confirmed the original `TIN` is never mutated (a new `TIN` is
always returned). Confirmed flat terrain is unaffected regardless of
iteration count (nothing to smooth toward).

**Confirmed, by direct execution, that the resulting TIN's own
triangulation connectivity is genuinely identical** -- not merely
claimed by the docstring. `laplacian_smooth()` calls `TIN.from_points()`
on the smoothed vertices, which re-triangulates from scratch; since
Delaunay triangulation depends only on XY position (confirmed
elsewhere in this documentation set -- see [`tin.md`](./tin.md)) and
smoothing never touches XY, the new triangulation is deterministically
identical to the original. Verified directly on a random 40-point
TIN, smoothed over multiple iterations: `simplices` arrays compare
exactly equal.

## `SpikeDetector` / `detect_spikes()` / `remove_spikes()`

```python
detect_spikes(tin, *, threshold: float) -> NDArray[np.bool_]
remove_spikes(tin, *, threshold: float) -> TIN
```

Flags a vertex whose elevation deviates from its own neighbors'
average by more than `threshold` (absolute Z units) -- there is no
default; the right value depends entirely on the source survey's own
noise characteristics (LiDAR blunders vs. total-station precision
differ by orders of magnitude, per the module's own docstring).
`threshold` must be strictly positive, confirmed rejected otherwise.
Confirmed correct on a dense, realistic grid with one true moderate
outlier: only that one vertex is flagged, not its neighbors.

**A documented, confirmed characteristic, not a bug**: an extreme
outlier with very few neighbors (a sparse configuration) skews those
neighbors' own local averages enough that they get flagged too --
confirmed directly with a 5-vertex "fan" (1 center, 4 neighbors):
an extreme center spike flags all 5 vertices, not just the center.
This project's own test suite documents this as an inherent property
of neighbor-mean-based detection, confirmed NOT to happen in the
denser, more realistic grid case above.

`remove_spikes()` drops every flagged vertex and re-triangulates the
survivors; raises `TerrainError` (not `TerrainValidationError`) if
fewer than 3 vertices would remain. A separate, purely theoretical
inconsistency was noted but not corrected during this audit: if
exactly the *surviving* vertices happened to be collinear (a
different, unrelated failure mode from "too few remain"), the
underlying `TIN.from_points()` call would raise `TriangulationError`
unwrapped, not `TerrainError` -- but reaching this specific
configuration in practice requires an initial point cloud close
enough to already-degenerate that it would typically fail earlier
construction anyway; not pursued further given its low practical
likelihood. See [`limitations.md`](./limitations.md).

## `BreaklineEnforcer` / `apply_breaklines()`

```python
apply_breaklines(tin: TIN, breaklines: Sequence[Breakline]) -> TIN
```

An empty `breaklines` sequence is confirmed a true identity: the
exact same `TIN` object is returned (`is`, not just `==`), no
triangulation performed. Otherwise: merges every breakline's own
points into the existing vertex set (`_merge_points()`, deduplicating
by XY -- a breakline point whose XY coincides with an existing
vertex reuses that vertex's own index and, confirmed directly, its
own existing `Z`, silently discarding the breakline's own Z for that
point), builds the constraint-edge list for `HARD` breaklines only
(`SOFT` breaklines contribute vertices for elevation accuracy without
forcing a rigid edge), and calls
`ConstrainedDelaunayTriangulator.triangulate()`.

**Worth knowing deliberately, not a defect**: the XY-coincidence Z
behavior above means a breakline cannot be used to *correct* an
existing point's own elevation at the same XY -- confirmed directly,
constructing a breakline point at a coinciding XY with a different Z
and observing the original Z survive, not the breakline's own. This
matches the function's own docstring exactly; documented here because
it's the kind of behavior that could genuinely surprise a caller
expecting the opposite.

Confirmed directly: a `HARD` breakline forces the non-default
diagonal of a simple square into the final triangulation (the exact
edge that plain Delaunay wouldn't have chosen). A breakline with
fewer than 2 points raises `BreaklineError`. 2 `HARD` breaklines that
geometrically cross each other (confirmed with the 2 diagonals of a
square) raise `BreaklineError` ("cross each other; not supported
yet"), propagated correctly from `ConstrainedDelaunayTriangulator`
-- not silently resolved one way or the other. The original `TIN` is
confirmed never mutated.

`TIN(result)` inside `apply_breaklines()` -- constructing a `TIN`
directly from a `ConstrainedDelaunayTriangulator.triangulate()`
result -- was checked specifically during this audit and confirmed
NOT an error: this is `TIN`'s own private constructor, the same one
`from_points()`/`from_mesh()` are themselves built on (see
[`tin.md`](./tin.md)).
