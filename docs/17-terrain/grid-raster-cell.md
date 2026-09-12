# `Grid` / `Raster` / `Cell` / `DTM`

## `Grid` — pure geometry, no elevations

A frozen, slotted dataclass: `min_x`, `min_y`, `max_x`, `max_y`,
`resolution`. Stores no elevation values at all -- confirmed by
reading every field and method; elevations live only in `Raster`.

```python
Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)
```

### `width`/`height` vs `columns`/`rows` -- 2 different things, confirmed distinct by real execution

`width`/`height` are the nominal, requested extent
(`max_x - min_x`, `max_y - min_y`) -- exactly `10.0`, `10.0` for the
example above. `columns`/`rows` round the extent divided by
`resolution` UP (`math.ceil`) and add 1, to guarantee full coverage
when `resolution` doesn't evenly divide the extent -- `5`, `5` for
the same example (`ceil(10/3) + 1 = 5`), not `4` (`10/3` truncated).
This rounding is deliberate and correct, and was never the bug (see
below) -- confirmed unchanged by this audit's own fix.

### `bounds`/`contains()`/`actual_max_x`/`actual_max_y` -- the real, generated extent, not the nominal one (PR19, already fixed)

Because `columns`/`rows` round up, the last generated column/row can
sit past the nominal `max_x`/`max_y` whenever `resolution` doesn't
evenly divide the extent. For the example above: `actual_max_x` /
`actual_max_y` are `12.0`, `12.0` (the last column/row's own
coordinate: `0 + 4*3`), not the nominal `10.0`. `bounds` reports
`(0.0, 0.0, 12.0, 12.0)` -- the real generated extent -- and
`contains()` checks against this real extent, not the nominal one.
This was a real, historical bug (documented directly in `Grid`'s own
class docstring, PR19): `bounds`/`contains()`/`Raster.transform` used
to still describe the nominal `max_x`/`max_y`, so a grid's own last
row/column could fail its own `contains()` check, and GDAL/GeoTIFF
export could misalign whenever the resolution didn't evenly divide
the requested extent. Fixed via the `actual_max_x`/`actual_max_y`
properties; re-confirmed correct by this audit, including at real
UTM-scale coordinates (~500000, ~4500000) with millimeter-level
deviations -- no precision issue at that scale, since welding/bounds
math is internally consistent regardless of absolute coordinate
magnitude.

### Coordinate/index conversion

`x(column)`/`y(row)` return the coordinate of a column/row.
`coordinate(row, column)` returns both as a tuple. `row(y)`/`column(x)`
invert this via `round((coord - min) / resolution)` -- Python's own
banker's rounding applies at exact `.5` ties (confirmed directly:
`round(0.5)` gives `0`, not `1` -- standard Python semantics, not a
`terrain`-specific choice). `index(row, column)` returns the
flattened `row * columns + column` index.

### `cell(row, column)` -- confirmed working; `point(row, column)` -- removed, was permanently broken

`Grid.cell()` returns a `Cell` at the given row/column, with
`z=float("nan")` -- confirmed working, since `Cell` (unlike `Point3D`)
freely accepts `NaN`.

**`Grid.point()` no longer exists.** It used to return a `Point3D`
with `z=float("nan")`, but `Point3D`'s own validation rejects `NaN`
unconditionally -- confirmed directly: calling `Grid.point()` always
raised `MathError`, with zero possible successful calls, ever. No
caller anywhere in the codebase (confirmed by search) or in the
project's own test suite exercised it -- the real project's own
coverage report showed its exact body as never executed. Removed
during this audit rather than fixed, since `Grid.cell()` already
provides the same "point at this grid location, elevation not yet
known" need correctly. See [`validation.md`](./validation.md).

## `Cell` — the shared per-point unit `Grid` and `Raster` both return

A frozen, slotted dataclass: `row`, `column`, `x`, `y`, `z`.
`has_data`/`is_nodata` are exactly `math.isfinite(z)` /
`not math.isfinite(z)` -- confirmed directly, including for `+inf`/
`-inf`, not just `NaN`. `xy`/`xyz` return coordinate tuples.
`distance_to()` is planar (X/Y only, `Z` ignored) -- confirmed with a
3-4-5 triangle giving exactly `5.0` regardless of the 2 cells'
different `Z` values. `translated()` returns a modified copy,
explicit `None` for any field meaning "keep this field's current
value" (confirmed directly, not merely documented) -- distinct from
simply omitting the argument, which has the identical effect but
lets a caller be explicit either way. `Cell` iterates as
`(row, column, x, y, z)`, confirmed directly.

## `Raster` — a `Grid` paired with a `values` array

A frozen, slotted dataclass: `grid: Grid`, `values: NDArray[np.float64]`,
`nodata: float = NaN`. `__post_init__` rejects a `values` shape that
doesn't match `grid.shape` exactly -- confirmed directly.

### Statistics and shape accessors -- all confirmed against known values

`shape`/`rows`/`columns`/`size` mirror `values.shape`. `minimum`/
`maximum`/`mean`/`std` use `np.nanmin`/`nanmax`/`nanmean`/`nanstd`
-- NoData cells are excluded, confirmed directly. `valid_count`/
`nodata_count` sum to `size`, confirmed directly. `statistics()`
returns all of these together as one dict:
`{rows, columns, cells, valid, nodata, minimum, maximum, mean, std}`
-- confirmed the exact key set by direct execution. `bounds`/`extent`
are identical (`extent` is a plain alias, confirmed by direct
comparison) and both delegate to `grid.bounds` -- so `Raster.bounds`
already reflects the PR19 actual-extent fix above, confirmed
directly propagated through, not just at the `Grid` level.

### `transform` -- the GDAL/GeoTIFF affine transform, confirmed to use the real extent

`(origin_x, pixel_width, 0.0, origin_y, 0.0, -pixel_height)`, with
`origin_y = grid.actual_max_y` (the real, generated top edge), not
the grid's own nominal `max_y` -- confirmed directly, both for a
grid where `resolution` evenly divides the extent (nominal and
actual coincide) and one where it doesn't (they differ, and
`transform` correctly uses the actual value). This is the same PR19
fix propagated one level up from `Grid` itself.

### Cell-level access, masking, and windows

`value(row, column)`/`cell(row, column)`/`is_nodata(row, column)`
take raw integer indices -- the caller's own responsibility to keep
in range, matching ordinary array-indexing convention (unlike the
coordinate-based methods below). `copy()` deep-copies the `values`
array (confirmed: mutating the copy leaves the original untouched).
`fill(value)`/`mask(bool_array)` each return a new `Raster` (the
original is never mutated, confirmed directly) -- `fill()` replaces
every NoData cell; `mask()` turns every `False` cell into NoData,
confirmed to reject a mismatched mask shape with a clear `ValueError`.
`valid_cells()` iterates only non-NoData cells as `Cell` objects;
plain iteration (`for cell in raster`) visits every cell, NoData
included, in row-major order -- confirmed directly, first and last
cell indices as expected. `window(row, column, radius=1)` returns a
copy of the `(2*radius+1)`-ish neighborhood, clipped (not padded) at
the raster's own edges -- confirmed directly: a corner cell's own
window is `2x2`, not `3x3` with fill values.

### `elevation(x, y)` -- coordinate-based lookup, fixed to respect the raster's own bounds

Returns the value of the nearest cell to a given `(x, y)` coordinate,
via `grid.row(y)`/`grid.column(x)`. **Found and fixed during this
audit**: this used to compute row/column and index directly into
`values` with no bounds check at all. For a coordinate just outside
the raster's own negative edge (e.g. `x` one cell below `min_x`),
the computed column was `-1`, and NumPy's own negative-index
wraparound silently returned the value from the OPPOSITE edge of the
raster -- a plausible-looking but entirely wrong number, not an
error. `raster.contains(x, y)` already correctly reported `False`
for this exact coordinate; `elevation()` simply never checked it.
Confirmed directly, both before the fix (reproduced the wraparound)
and after (now returns `self.nodata`, matching this class's own
NoData convention, for any out-of-bounds coordinate; every in-bounds
case confirmed byte-for-byte unaffected). `RasterSampler.nearest()`
(a separate utility, see [`utilities.md`](./utilities.md)) already
validated `.contains()` first -- this fix brings `Raster.elevation()`
in line with that already-established, safer pattern. See
[`validation.md`](./validation.md) for the complete account.

## `DTM` — a `TIN` sampled onto a `Grid`, producing a `Raster`

A frozen, slotted dataclass: `tin: TIN`, `grid: Grid`, `raster: Raster`.
Confirmed by reading every method: `DTM` reimplements none of
`Raster`'s own logic -- `.elevation()`, `.cell()`, `.statistics()`,
`.array()`, `__getitem__`, `__iter__`, `__len__` are each a one-line
delegation to `self.raster`; `.rows`/`.columns`/`.width`/`.height`/
`.resolution`/`.bounds` delegate to `self.grid`. The one piece of
real logic `DTM` owns is its own classmethod:

```python
DTM.from_tin(tin: TIN, grid: Grid, interpolator: BaseInterpolator) -> DTM
```

For every cell of `grid`, query `interpolator.interpolate(x, y)`;
catch `InterpolationError` (raised for any query point outside the
TIN's convex hull) and leave that cell as `NaN` rather than aborting
the whole DTM -- confirmed directly, both for cells inside the hull
(exact expected elevation, verified against a known tilted plane)
and outside it (confirmed `NaN`, for a grid deliberately extending
past a small triangular TIN's own hull). This is documented in
`DTM.from_tin()`'s own docstring as the expected, common case for a
rectangular grid over an irregular real survey boundary, not a
failure mode to guard against.
