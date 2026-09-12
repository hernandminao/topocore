# Terrain

## `TIN` — Triangulated Irregular Network

```python
from topocore.terrain.tin import TIN
from topocore.geometry.point3d import Point3D

tin = TIN.from_points(points: list[Point3D] | tuple[Point3D, ...])
```

Builds a Delaunay triangulation from the given points. Rejects
duplicated XY coordinates and fully collinear point sets with a clear
`TriangulationError`, rather than producing a degenerate mesh.

```python
tin = TIN.from_mesh(vertices: tuple[Point3D, ...], simplices: NDArray[np.int32])
```

Builds a `TIN` from an already-known mesh -- preserves the exact
triangulation given, never re-triangulates. Useful when you already
have a mesh from another source (e.g. a constrained triangulation
honoring breaklines).

```python
tin.vertex_count
tin.triangle_count
tin.bounds                 # (min_x, min_y, max_x, max_y)
tin.find_triangle(x, y)     # -> int index, or -1 if outside the hull
tin.interpolate(x, y)        # -> float, barycentric; raises ValueError outside the hull
tin.vertex_array()            # (n, 3) array -- a method, not a property
tin.elevation_array()          # (n,) array of vertex Z -- a method, not a property
```

## `Grid` — the raster footprint a `DTM` is sampled onto

```python
from topocore.terrain.grid import Grid

grid = Grid(min_x: float, min_y: float, max_x: float, max_y: float, resolution: float)
```

`Grid.bounds`/`.actual_max_x`/`.actual_max_y` reflect the *real*
generated extent, not the nominal one you passed in -- confirmed a
real, historical fix: when `resolution` doesn't evenly divide the
extent, the grid's own last row/column can extend past your nominal
`max_x`/`max_y`, and these properties correctly report the actual
extent rather than the requested one.

## `DTM` — a `TIN` sampled onto a `Grid`

```python
from topocore.terrain.dtm import DTM

dtm = DTM.from_tin(tin: TIN, grid: Grid, interpolator: BaseInterpolator)
```

For every grid cell inside the `TIN`'s own convex hull, samples the
interpolator; cells outside it are left as `NaN` -- this is the
expected, documented behavior for a rectangular grid over an
irregular real survey boundary, not a failure mode.

## Interpolation methods

```python
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.interpolation import TerrainInterpolator

interpolator = TerrainInterpolator(tin, method=InterpolationMethod.LINEAR, power=2.0)
```

4 real values, confirmed as the single source of truth (a real,
historical duplicate-enum dispatch bug -- passing a string like
`"linear"` used to silently fall through to `NEAREST` -- was fixed by
consolidating to this one definition and comparing by value, not
identity):

```text
InterpolationMethod.LINEAR        -- == BARYCENTRIC (same underlying math)
InterpolationMethod.BARYCENTRIC
InterpolationMethod.IDW           -- power only matters for this one
InterpolationMethod.NEAREST       -- nearest-vertex, no interpolation math
```

`power` (default `2.0`) only affects `IDW`; it is accepted but has no
effect for the other 3 methods.

## Contours

```python
from topocore.terrain.contours import ContourGenerator

contours = ContourGenerator(tin).generate(interval: float, base: float = 0.0)
```

Confirmed directly: consumes the `TIN` only, never the `DTM` --
`ContourGenerator` has no dependency on `DTM` at all, so building a
`DTM` first is never required for contours (see
[`workflows.md`](./workflows.md)'s own `extract_contours()`, which
requires `TIN` for exactly this reason). A contour level landing
exactly on vertex elevations (a flat pad, a graded slope's own toe)
is correctly produced as a real, closed line, not silently dropped --
a real, historical fix confirmed still correct.

## Slope, aspect, hillshade — per-triangle, not raster-based

```python
from topocore.terrain.slope import triangle_slope, SlopeCalculator
from topocore.terrain.aspect import triangle_aspect, AspectCalculator
from topocore.terrain.hillshade import triangle_hillshade, HillshadeCalculator

SlopeCalculator(tin).compute()       # degrees, [0, 90], one value per triangle
AspectCalculator(tin).compute()      # degrees, [0, 360), downslope direction
HillshadeCalculator(tin, azimuth=315.0, altitude=45.0).compute()  # [0, 255]
```

Computed directly from each TIN triangle's own exact plane, not a
resampled raster with a finite-difference approximation -- the same
approach professional TIN-based earthwork engines use. `aspect`
reports the downslope direction specifically (confirmed: not the
direction of ascent, an easy value to get backward). A flat triangle
returns `nan` for aspect, not an error.

## Where to go next

- [`workflows.md`](./workflows.md) -- `build_tin()`/`build_dtm()`/
  `extract_contours()`'s own full `Workflow` contract and artifact
  dependencies.
- [`analysis.md`](./analysis.md) -- statistics, profiles, and
  visibility computed from a `TIN`/`DTM`.
- [`surface.md`](./surface.md) -- comparing 2 surfaces and computing
  cut/fill volumes.
