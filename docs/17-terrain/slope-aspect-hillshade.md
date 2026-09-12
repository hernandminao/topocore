# Slope / Aspect / Hillshade

## Per-triangle, not raster-based -- a deliberate design choice

All 3 modules compute directly from the exact plane each TIN triangle
defines, not from a resampled raster with a finite-difference (Horn's
method) approximation. This is stated explicitly in each module's
own docstring, and `slope.py`'s own docstring names it as the same
approach "used by professional TIN-based earthwork engines" -- a
raster-based approximation would introduce error a TIN-native
computation doesn't have to accept.

## `_geometry.py` -- the one shared source of truth (private)

`oriented_normal(triangle)`, `slope_radians(normal)`,
`aspect_radians(normal)` -- confirmed the single place all 3 public
modules derive their own math from, not merely by convention: each
module's own docstring states it, and `aspect_radians()`'s own
docstring is explicit that this is deliberate: `slope.py`/`aspect.py`/
`hillshade.py` "can never disagree at the tolerance boundary" as a
structural consequence of sharing this one flatness check
(`is_zero(normal.x) and is_zero(normal.y)`), not a coincidence of 3
independently-written checks that happen to agree.

`oriented_normal()` always flips the triangle's own cross-product
normal to `z >= 0` -- confirmed directly: 2 triangles with opposite
vertex winding order give the identical normal, and therefore
identical slope/aspect.

### Degenerate triangles -- confirmed unified to one exception, fixed during this audit

`oriented_normal()` calls `validate_triangle()` first, then
separately checks whether its own computed cross-product is zero.
**Found and fixed during this audit**: `validate_triangle()` used to
check only for duplicated vertices, missing the broader case of 3
genuinely distinct but collinear vertices (also zero-area, equally
degenerate) -- that case fell through to `oriented_normal()`'s own
separate zero-normal check instead, which raised a *different*
exception (`TerrainError`, not `TerrainValidationError`). Confirmed
directly, before the fix: duplicated vertices raised
`TerrainValidationError`; collinear-but-distinct vertices raised
`TerrainError` -- 2 different exceptions for what a caller would
reasonably consider the same category of problem. Both now raise
`TerrainValidationError`, confirmed directly. See
[`validation.md`](./validation.md).

## `slope.py` -- `triangle_slope()` / `SlopeCalculator`

Returns degrees in `[0, 90]`: `0` for a horizontal facet, `90` for a
vertical one. Confirmed exact against hand-derived analytic cases:
a `z=x` plane gives exactly `45.0`; a `1000`-over-`1` drop gives
`~90.0` (confirmed within `0.1`, matching the near-vertical
asymptotic case); a flat triangle gives exactly `0.0`.
`SlopeCalculator(tin).compute()` returns one value per triangle, in
`tin.simplices` order; `.at(index)` returns a single triangle's own
value, confirmed to match the corresponding entry from `.compute()`
exactly.

## `aspect.py` -- `triangle_aspect()` / `AspectCalculator`

Returns degrees in `[0, 360)`, clockwise from North -- and
**specifically the DOWNSLOPE direction**, not the direction of
ascent. Confirmed against all 4 cardinal cases (a triangle
descending toward East/North/South/West gives aspect
90/0/180/270 respectively) and against the easy-to-get-backward
case: a `z=x` plane ascends toward East, and its own aspect is
confirmed to be `270` (West -- the downslope direction), not `90`.
A flat triangle returns `nan` (confirmed, not an exception) --
consistent with this module's own stated NoData convention.
`AspectCalculator` mirrors `SlopeCalculator`'s own interface exactly
(`.compute()`, `.at(index)`, `__call__`).

## `hillshade.py` -- `triangle_hillshade()` / `HillshadeCalculator`

Implements the standard Lambertian illumination model (matching
GDAL's `gdaldem hillshade`):

```text
cos(incidence) = cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
illumination = 255 * max(0, cos(incidence))
```

`azimuth` (sun compass direction, `[0, 360]`) and `altitude` (sun
elevation above horizon, `[0, 90]`) are both validated, both at
`triangle_hillshade()`'s own call site and at
`HillshadeCalculator`'s own construction -- confirmed rejected
outside range, confirmed accepted at both inclusive boundaries
(`0`/`360` for azimuth, `0`/`90` for altitude). Defaults
(`DEFAULT_AZIMUTH=315.0`, `DEFAULT_ALTITUDE=45.0`) match the standard
GDAL/ArcGIS convention (a Northwest sun), confirmed by the module's
own named constants.

Confirmed exact against 4 hand-derived analytic cases: a flat facet
under a directly-overhead sun (`altitude=90`) gives exactly `255.0`;
a flat facet at `altitude=45` gives exactly `255 * sin(45deg)`; a
steep facet facing away from a low sun gives exactly `0.0` -- clamped,
confirmed never negative even though the underlying cosine is
mathematically negative there; a facet whose own slope/aspect exactly
matches the sun's own zenith/azimuth (perpendicular incidence) gives
exactly `255.0`, confirmed by direct construction (using
`altitude = 90 - triangle_slope(triangle)` and
`azimuth = triangle_aspect(triangle)`). A flat facet (undefined
aspect) is confirmed handled correctly without special-casing needed
at the call site -- `sin(slope_rad)` is itself `0` for a flat facet,
so the aspect-dependent term vanishes mathematically regardless of
what `aspect_radians()` would have returned; the explicit `if aspect_rad
is not None` guard exists to avoid `cos(azimuth - None)` crashing,
not because the math would otherwise be wrong.

`HillshadeCalculator.azimuth`/`.altitude` are confirmed read-only
(no setter exists) -- unlike `TerrainInterpolator.method`
(see [`interpolation.md`](./interpolation.md)), there is no way to
mutate this class's own state past its constructor's validation.
