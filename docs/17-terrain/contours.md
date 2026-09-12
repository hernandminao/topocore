# Contours — marching triangles

## The algorithm

`ContourGenerator(tin).generate_at(elevation)` intersects every
triangle in the TIN against a horizontal plane at `elevation`,
producing short segments (`_triangle_segment()`), then welds and
stitches those segments into continuous `ContourLine` polylines
(`_stitch()`, via a tolerance-quantized node-welding graph:
`_build_graph()` → `_walk()` → `_open_polylines()`/`_closed_polylines()`).
`generate(interval, base)` is the higher-level entry point: computes
the set of levels to contour within the TIN's own elevation range
(`_levels()`) and calls `generate_at()` for each.

## The critical historical bug -- exact-elevation levels, already fixed, re-confirmed

**Confirmed still fixed by this audit**: a contour level landing
exactly on vertex elevations -- a flat pad/plateau, or a graded
slope's toe at a round design elevation -- used to disappear from
the output entirely. This is a common REAL construction-topography
shape (this project's own regression suite specifically escalated it
to CRITICAL after reproducing it not just on a synthetic 5-vertex
pyramid but on a realistic 25-vertex graded building pad, with a
genuine Delaunay triangulation, not hand-picked simplices).

The fix lives in `_triangle_segment()`'s own explicit handling of
vertices classified exactly `ON` the level (within `EPSILON`):

- 2 vertices `ON`, 1 off -- the edge between the 2 `ON` vertices IS
  the contour segment for that triangle, taken directly (no
  interpolation, since both endpoints already sit at `level`).
- 3 vertices `ON` -- the whole triangle is coplanar with `level`;
  it contributes no segment of its own, deferring to whichever
  neighboring, non-coplanar triangles supply the real boundary via
  the 2-`ON` case above.
- 1 vertex `ON`, the other 2 on the SAME side -- confirmed a
  genuinely different, still-excluded case: the contour only
  touches this triangle at a single point, not a real line through
  its interior, and correctly produces no segment. Verified directly
  with the pyramid's own peak vertex at its own exact elevation: `0`
  contours, not `1` degenerate one.

Re-confirmed by this audit, including at real UTM-scale coordinates
(the exact pyramid reproduction, shifted to ~500000/4500000): the
exact-level case still correctly produces its own real, closed
contour.

## `_levels()` -- a second, separate defect found and fixed during this audit

**Found and fixed**: the level sequence used to accumulate via
repeated `level += interval` inside its own loop. For an `interval`
with no exact binary floating-point representation (`0.1`, common in
real survey work), this drifts with every iteration -- over a
realistic range (1000 levels), the level meant to be exactly `100.0`
came out as `99.9999999999986`. The drift was always far smaller
than `EPSILON`, so it never changed *which* triangle edges got
contoured -- but the elevation value itself is used verbatim as each
contour's own label in real output (legends, exported layer names),
where it would have displayed this floating-point noise instead of a
clean number. Fixed by computing each level directly as
`first + n * interval` for an integer step count `n` (no
accumulation possible), then rounding to 9 decimal places. Confirmed
directly: the same 1000-level case now ends at exactly `100.0`;
every previously-passing case (a 900-level, exactly-representable
`0.5` interval; a non-zero `base` offset) confirmed unchanged. See
[`validation.md`](./validation.md).

## Multiple, disjoint rings at the same level -- confirmed correctly separated

A TIN with 2 separate peaks, each producing its own closed ring at
the same contour level, is confirmed to yield 2 separate
`ContourLine` objects, not one incorrectly-merged polyline -- the
graph-based stitching (`_stitch()`) correctly keeps disconnected
components apart.

## Node welding -- confirmed scale-independent

`_node_key()` quantizes a point's own coordinates by dividing by
`EPSILON` (`1e-9`) and rounding, to weld the 2 independently-computed
endpoints 2 adjacent triangles produce for the shared boundary
between them. Confirmed to still work correctly at real UTM-scale
coordinates (~500000, ~4500000): welding depends on *consistency*
(the same floating-point division applied to the same logical point
from both sides), not on the absolute magnitude of the coordinate
values, so precision loss in the division itself doesn't break it.

## What this module does not yet do

Its own module docstring states plainly: contouring operates on the
unconstrained TIN produced by `DelaunayTriangulator`; breakline-aware
contouring (honoring hard/soft breaklines as triangulation
constraints during contour generation itself, not merely during TIN
construction) "will be introduced once the constrained triangulation
backend is available" -- which it now is
(`algorithms.constrained_delaunay`, see [`tin.md`](./tin.md) and
[`filters-breaklines.md`](./filters-breaklines.md)), but `contours.py`
itself has not yet been updated to take advantage of it. Contours
generated over terrain with sharp breaklines may still cross them.
This is the module's own stated limitation, not a gap this audit
discovered independently.
