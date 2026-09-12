# Volume

Earthwork volume calculations -- 3 genuinely different methods (a
single surface against a datum, a grid/array-based cut/fill between 2
surfaces, and corridor-style cross-section integration), all sharing
one `VolumeResult`.

```python
result.cut_volume    # float
result.fill_volume    # float
result.net_volume      # cut - fill
result.method            # a string label -- see the note on GridVolume/CutFillVolume below
result.valid_cells        # int | None
result.excluded_cells      # int | None
```

## `TINVolume` -- one surface, relative to a datum

```python
from topocore.analysis.volume import TINVolume

TINVolume(datum=0.0).compute(tin) -> VolumeResult
```

**This is not a comparison between 2 surfaces** -- it computes the
volume of material between a single TIN and a horizontal reference
plane. Verified directly on a perfectly flat TIN at `Z=5` over a
10x10 area with `datum=0.0`: `cut_volume=500.0` (5 units of height x
100 square units of area), `fill_volume=0.0` -- material sits entirely
above the datum, reported as `cut` (material that would need
removing to reach the datum).

## `GridVolume` and `CutFillVolume` -- the same formula, confirmed

```python
from topocore.analysis.volume import GridVolume, CutFillVolume

GridVolume(resolution=1.0).compute(existing, proposed) -> VolumeResult
CutFillVolume(cell_area=1.0).compute(existing, proposed) -> VolumeResult
```

**These 2 classes compute an identical result for identical inputs.**
Confirmed directly, including with a genuinely non-uniform surface
(`existing = [[0,0],[0,0]]`, `proposed = [[0,1],[2,4]]`): both return
`fill_volume=7.0` (matching the hand-computed sum `0+1+2+4`), and both
delegate to the exact same shared function
(`topocore.analysis._shared.volume.compute_cut_fill`) -- this is not
2 independent implementations that happen to agree. **The `method`
field (`"grid_volume"` vs. `"cut_fill"`) is metadata only** -- it does
not indicate any difference in the underlying calculation.

The real difference between the 2 classes is their own DTM-integration
convenience method, and it matters:

```python
GridVolume(resolution=1.0).compute_from_dtm(existing_dtm, proposed_dtm)
CutFillVolume(cell_area=1.0).compute_with_dtm(existing_dtm, proposed_dtm)
```

`GridVolume.compute_from_dtm()` **validates** that its own configured
`resolution` matches the DTMs' own actual resolution, raising
`VolumeError` on a mismatch rather than silently using either value.
A real, severe defect existed here previously (already fixed in the
code you have): it used to ignore the DTMs' own resolution entirely,
silently computing a volume off by a factor of `resolution` squared
(up to 4x wrong in a confirmed case) whenever the constructor's own
`resolution` didn't match. `CutFillVolume.compute_with_dtm()` takes a
different, equally valid approach: it never stores a persistent
resolution to potentially mismatch against -- it always derives a
fresh `cell_area` from the DTM's own resolution at call time, so this
specific failure mode cannot occur there by construction, at the cost
of never letting you assert your own expectation about what
resolution the DTMs should have.

`compute_cut_fill()` (the shared function both delegate to) excludes
NaN cells from both surfaces rather than failing outright -- a
deliberate choice: a `DTM.from_tin()` result routinely has NaN cells
outside its own source TIN's convex hull, and rejecting any grid
containing NaN would make cut/fill volume unusable on essentially any
real, irregularly-bounded terrain. `excluded_cells` in the result
tells you how much of the grid had no overlapping data, so a
near-total exclusion (mismatched or barely-overlapping surfaces) is
visible rather than silently returning a near-zero volume computed
from a handful of coincidentally-valid cells.

## `AverageEndAreaVolume` and `PrismoidalVolume` -- corridor cross-sections, not grids

```python
from topocore.analysis.volume import AverageEndAreaVolume, PrismoidalVolume

AverageEndAreaVolume(sections=[(station, area), ...]).compute() -> VolumeResult
PrismoidalVolume(sections=[(station, area), ...]).compute() -> VolumeResult
```

`sections` (the type alias `Section` is `tuple[float, float]` --
`(station, area)`, not a constructible class) come from your own
cross-section area computations (e.g.
[`profile.CrossSectionProfile`](./profile.md) results, reduced to an
area per station yourself) -- this class does not compute
cross-sectional areas itself.

Verified directly against hand-computed results:
`AverageEndAreaVolume` between stations `0` (area `10`) and `10`
(area `20`) gives `150.0` (the trapezoidal average, `(10+20)/2 * 10`).
`PrismoidalVolume` with an added midpoint (station `5`, area `18`)
gives `170.0`, matching the standard prismoidal formula exactly:
`L/6 * (A1 + 4*Am + A2) = 10/6 * (10 + 4*18 + 20) = 170.0`. The 2
methods genuinely differ numerically for the same end sections once a
midpoint area is involved -- `PrismoidalVolume` is not simply
`AverageEndAreaVolume` under another name.

**`PrismoidalVolume` requires an odd number of sections, at least 3
-- a real, enforced precondition, not a recommendation.** It
implements the composite Simpson's 1/3 rule, which mathematically
requires pairing each interval with a real, measured midpoint area
(not one synthesized from the endpoints -- a real, historical defect,
already fixed, where `PrismoidalVolume` used to be mathematically
equivalent to `AverageEndAreaVolume` for exactly this reason).
Verified directly, 2 distinct, clear rejections:

```python
>>> PrismoidalVolume(sections=[(0.0, 10.0), (10.0, 20.0)]).compute()          # 2 sections
VolumeError: Prismoidal volume requires at least three sections.
>>> PrismoidalVolume(sections=[(0,10.0),(5,15.0),(10,20.0),(15,25.0)]).compute()  # 4 sections -- even, but >= 3
VolumeError: Prismoidal volume requires an odd number of sections.
```

`AverageEndAreaVolume` has no such restriction -- any number of
sections >= 2 is accepted, since the trapezoidal rule pairs adjacent
sections directly rather than requiring a true midpoint.

## `VolumeAnalysis` -- the dispatcher

Selects between the methods above by name, matching the same generic
`*args`-dispatch pattern used elsewhere in this section. `VolumeMethod`
(the selector) is a real `StrEnum`, unlike several other submodules'
own dispatcher-selector types -- see [`contracts.md`](./contracts.md).
