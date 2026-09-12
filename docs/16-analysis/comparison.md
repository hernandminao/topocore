# Comparison

Spatial, per-cell comparison between 2 terrain surfaces -- where they
differ, not just by how much in total. This is the only submodule in
`topocore.analysis` with a confirmed dependency on another one
(`volume`).

## `SurfaceComparisonResult`

```python
result.difference       # proposed - existing, per cell (NaN at excluded cells)
result.cut_mask          # proposed below existing, beyond tolerance
result.fill_mask          # proposed above existing, beyond tolerance
result.unchanged_mask      # within tolerance
result.valid_cells          # int
result.excluded_cells        # int (NoData)
result.minimum_difference     # float
result.maximum_difference     # float
result.mean_difference         # float
result.cut_cells                # property -- count from cut_mask
result.fill_cells                 # property -- count from fill_mask
result.unchanged_cells              # property -- count from unchanged_mask
```

## `SurfaceComparison` -- spatial comparison only, grid-based

```python
from topocore.analysis.comparison import SurfaceComparison

SurfaceComparison(tolerance=0.0).compute(existing, proposed) -> SurfaceComparisonResult
```

## `TINComparison` -- the same comparison, from 2 triangulated surfaces

```python
from topocore.analysis.comparison import TINComparison

TINComparison(resolution=0.5, tolerance=0.0, max_grid_cells=8_000_000).compute(existing_tin, proposed_tin) -> SurfaceComparisonResult
```

Rasterizes both TINs at `resolution` before comparing -- `max_grid_cells`
exists specifically to bound how large that raster can get for a
large TIN at a fine resolution, rather than allocating an
arbitrarily large array.

## Both raise `VolumeError`, confirmed as the real project's own documented contract

```python
>>> SurfaceComparison(tolerance=-1.0)
VolumeError: Comparison tolerance cannot be negative.
>>> TINComparison(resolution=-1.0)
VolumeError: Resolution must be positive.
```

This was investigated carefully during this documentation effort's
own audit, including a serious look at reclassifying it as a defect:
`VolumeError`'s own docstring ("Raised when a volume calculation
fails") and `_shared.volume`'s own documented list of intended
consumers (grid volume, cut/fill analysis, terrain volume methods)
never explicitly list surface comparison, and
`comparison/surface.py` does reuse `_shared.volume`'s own
`validate_volume_arrays()` helper for its shape/NaN checks. Based on
that evidence alone, `VolumeError` looked like an unintended leak
from `volume`'s own domain.

**That conclusion was wrong, and the correction matters as a
methodology note for this whole audit**: this project's own real
test suite (not available in the sandbox this audit's own tooling
ran in, and not discovered until a fix based on the above reasoning
was already applied and then had to be reverted) includes
`tests/analysis/comparison/test_public_api.py::test_documented_error_paths_raise_only_volume_error`
-- a test whose own name states plainly that `VolumeError` is this
API's documented, intended contract. The lesson: absence of a
counter-example in a partial or incomplete test copy is not the same
as absence of one in the real project, and "the docstring doesn't
mention it" is not strong enough evidence to override an existing,
explicitly-named test contract once one is found.

## `SurfaceCutFill` -- comparison **and** volume, together

```python
from topocore.analysis.comparison import SurfaceCutFill

comparison_result, volume_result = SurfaceCutFill(cell_area=1.0, tolerance=0.0).compute(existing, proposed)
```

**Confirmed directly, not merely documented as intended**:
`SurfaceCutFill` imports `CutFillVolume` from
[`volume`](./volume.md) directly and delegates to it for the volume
half of its own return value. The `comparison_result` half is
field-for-field identical to what `SurfaceComparison` alone would
produce for the same inputs -- verified directly, comparing every
field (`mean_difference`, `cut_cells`, `fill_cells`, and the full
`difference` array via `np.array_equal`) between the two for the same
`existing`/`proposed` pair.

**Use `SurfaceComparison` alone if you only need the spatial
picture** (a difference map, cut/fill masks) **and `volume.CutFillVolume`
alone if you only need the total volumes** (see
[`volume.md`](./volume.md)) -- `SurfaceCutFill` exists specifically
for when you need both together, computed once, rather than calling
both separately and duplicating the underlying comparison work.
`SurfaceCutFill`'s own `cell_area` validation also raises
`VolumeError`, consistent with `SurfaceComparison`/`TINComparison`
above.

## What this submodule does not do

It never computes a volume on its own for a **single** surface
relative to a datum (no cut/fill between two surfaces involved) --
that is `volume.TINVolume`, a genuinely different operation covered
in [`volume.md`](./volume.md), not something `comparison` overlaps
with.
