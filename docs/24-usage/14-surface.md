# Surface Comparison & Cut/Fill

## `SurfaceComparison` — 2 elevation grids, same shape

```python
from topocore.analysis.comparison.surface import SurfaceComparison

comparison = SurfaceComparison(*, tolerance: float = 0.0)
result = comparison.compute(existing: FloatArray, proposed: FloatArray) -> SurfaceComparisonResult
```

`existing`/`proposed` are 2 elevation grids of the same shape (e.g.
2 `DTM`-derived arrays over the same `Grid`). `tolerance` must be
finite and non-negative (enforced at construction). Confirmed
directly: raises `VolumeError` for invalid parameters, not a
separate `ComparisonError` -- this was investigated explicitly during
this project's own audit and confirmed to be the real, intended
contract (a real, existing test asserts `VolumeError` specifically),
not an oversight.

## `SurfaceCutFill` — comparison + volume in one call

```python
from topocore.analysis.comparison.cut_fill import SurfaceCutFill

cutfill = SurfaceCutFill(*, cell_area: float, tolerance: float = 0.0)
comparison_result, volume_result = cutfill.compute(existing: FloatArray, proposed: FloatArray)
```

Composes `SurfaceComparison` (elevation differences) with
`CutFillVolume` (grid-cell volume integration) -- confirmed directly
by reading its own `__init__`, not 2 independently-written
implementations that happen to agree. `cell_area` must be finite and
strictly positive.

## `TINComparison` — 2 TINs, sampled onto a shared grid

```python
from topocore.analysis.comparison.tin_surface import TINComparison

comparison = TINComparison(*, resolution: float, tolerance: float = 0.0, max_grid_cells: int = 8_000_000)
result = comparison.compute(existing: TriangulatedSurface, proposed: TriangulatedSurface) -> SurfaceComparisonResult
```

Use this instead of `SurfaceComparison` when you have 2 `TIN`
surfaces rather than 2 already-gridded elevation arrays -- it samples
both onto a common grid at `resolution` internally before comparing.
`max_grid_cells` guards against an unintentionally huge grid from a
very fine `resolution` over a large extent (confirmed a real
validated integer bound, not merely documented).

## `VolumeAnalysis` — the general volume-computation manager

```python
from topocore.analysis.volume.manager import VolumeAnalysis

VolumeAnalysis(...).compute(*args, method: str | None = None, **kwargs) -> VolumeResult
```

Same dispatch shape as the 4 managers in [`analysis.md`](./analysis.md).
Confirmed real underlying calculators: `AverageEndAreaVolume`,
`CutFillVolume`, `GridVolume`, `PrismoidalVolume` -- each usable
directly when you already know which volume method you want.

## Where to go next

- [`terrain.md`](./terrain.md) -- building the `TIN`/`DTM` surfaces
  these classes compare.
- [`analysis.md`](./analysis.md) -- the other 4 analysis managers
  (distance, profile, visibility, statistics), sharing the same
  `Manager(...).compute(...)` dispatch pattern as `VolumeAnalysis`.
