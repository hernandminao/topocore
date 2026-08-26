"""
Edge-case and regression hardening suite for
topocore.analysis.comparison -- PR20.5.

Includes one real, severe bug found and fixed in this session:

TINComparison had no safety limit on the sampling grid it builds
from two TINs' shared domain. Confirmed directly (with a timeout):
resolution=0.001 over a 20x20m domain built a 20,000 x 20,000 grid
(4*10**8 cells), and given TINComparison's own documented
O(grid_points x triangle_count) sampling cost, this combination hung
indefinitely with no error. Fixed by adding the same max_grid_cells
safety guard already used by
topocore.processing.ground.pmf.PMFGroundClassifier for an analogous
risk (same default: 8_000_000).

The remaining edge cases below were audited and found to already
behave correctly -- no further bugs found; these are permanent
regressions documenting confirmed-correct behavior at the boundaries
of SurfaceComparison/TINComparison:

- Tolerance boundary is inclusive for "unchanged" (difference exactly
  == +-tolerance classifies as unchanged, not cut/fill) -- a
  mathematically sound choice (strict < / > for cut/fill).
- A barely-overlapping domain (narrower than resolution) can sample
  to an entirely-NaN grid, correctly rejected via the existing
  validate_volume_arrays check rather than crashing.
- A single-cell overlap (exactly one resolution-sized cell) works
  correctly.
- TIN.find_triangle() handles points on a shared edge between two
  triangles consistently (no ambiguous double-counting or spurious
  ValueError for points genuinely inside the combined hull).
- Asymmetric NoData (a triangular TIN, i.e. NaN on one side only,
  against a full square TIN with no NaN) still correctly identifies
  only the genuinely overlapping valid region.
- TINComparison's result is byte-for-byte identical (via
  np.array_equal with equal_nan=True) to manually sampling both TINs
  and calling SurfaceComparison directly -- confirming no divergent
  internal logic.
- SurfaceCutFill's volume output remains identical to CutFillVolume's
  own (pre-existing, already-audited) computation on the same mixed
  cut/fill/NoData array.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.comparison import (
    SurfaceComparison,
    SurfaceCutFill,
    TINComparison,
)
from topocore.analysis.exceptions import VolumeError
from topocore.analysis.volume.cut_fill import CutFillVolume
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


def _square(z: float, min_xy: float, max_xy: float) -> TIN:
    points = (
        Point3D(min_xy, min_xy, z),
        Point3D(max_xy, min_xy, z),
        Point3D(min_xy, max_xy, z),
        Point3D(max_xy, max_xy, z),
    )
    return TIN.from_points(points)


def _triangle(z: float, size: float = 20.0) -> TIN:
    points = (Point3D(0.0, 0.0, z), Point3D(size, 0.0, z), Point3D(0.0, size, z))
    return TIN.from_points(points)


# ----------------------------------------------------------------------
# The real bug: unbounded grid size from a small resolution.
# ----------------------------------------------------------------------


def test_tin_comparison_rejects_astronomically_large_grid() -> None:
    """The exact regression: before the fix, this hung indefinitely instead of raising."""
    existing = _square(110.0, 0.0, 20.0)
    proposed = _square(100.0, 0.0, 20.0)

    with pytest.raises(VolumeError, match="exceeding max_grid_cells"):
        TINComparison(resolution=0.001).compute(existing, proposed)


def test_tin_comparison_reasonable_resolution_still_works() -> None:
    existing = _square(110.0, 0.0, 20.0)
    proposed = _square(100.0, 0.0, 20.0)

    result = TINComparison(resolution=2.0).compute(existing, proposed)
    assert result.valid_cells == 100


@pytest.mark.parametrize("max_grid_cells", [0, -1, True, 1.5])
def test_tin_comparison_rejects_invalid_max_grid_cells(max_grid_cells: object) -> None:
    with pytest.raises(VolumeError, match="max_grid_cells"):
        TINComparison(resolution=1.0, max_grid_cells=max_grid_cells)  # type: ignore[arg-type]


def test_tin_comparison_max_grid_cells_is_configurable() -> None:
    """A caller can raise the limit for a genuinely large, intentional comparison."""
    existing = _square(110.0, 0.0, 4.0)
    proposed = _square(100.0, 0.0, 4.0)

    # 4x4 domain at resolution=0.1 -> 40x40 = 1600 cells, above a tiny custom limit.
    with pytest.raises(VolumeError, match="max_grid_cells"):
        TINComparison(resolution=0.1, max_grid_cells=100).compute(existing, proposed)

    # Same call succeeds once the limit is raised.
    result = TINComparison(resolution=0.1, max_grid_cells=10_000).compute(existing, proposed)
    assert result.valid_cells == 1600


# ----------------------------------------------------------------------
# Tolerance boundary -- inclusive for unchanged.
# ----------------------------------------------------------------------


def test_tolerance_boundary_is_inclusive_for_unchanged() -> None:
    comparison = SurfaceComparison(tolerance=0.05)
    result = comparison.compute(np.array([[100.0, 100.0]]), np.array([[100.05, 99.95]]))

    assert result.unchanged_cells == 2
    assert result.cut_cells == 0
    assert result.fill_cells == 0


def test_exactly_zero_difference_is_unchanged() -> None:
    comparison = SurfaceComparison(tolerance=0.0)
    result = comparison.compute(np.array([[100.0]]), np.array([[100.0]]))

    assert result.unchanged_cells == 1


# ----------------------------------------------------------------------
# Grid edge cases: barely-overlapping and single-cell overlap.
# ----------------------------------------------------------------------


def test_barely_overlapping_domain_narrower_than_resolution_rejected_cleanly() -> None:
    """
    Overlap width (0.1) is much smaller than resolution (2.0) -- the
    single sampled cell can miss the actual hull, correctly rejected
    by the existing all-NaN guard rather than crashing.
    """
    existing = _square(110.0, 0.0, 20.0)
    proposed = _square(100.0, 19.9, 39.9)

    with pytest.raises(VolumeError):
        TINComparison(resolution=2.0).compute(existing, proposed)


def test_single_cell_overlap() -> None:
    existing = _square(110.0, 0.0, 20.0)
    proposed = _square(100.0, 18.0, 38.0)  # overlap is exactly [18,20]x[18,20] -- one 2x2 cell

    result = TINComparison(resolution=2.0).compute(existing, proposed)
    assert result.valid_cells == 1


# ----------------------------------------------------------------------
# TIN edge: point on a shared triangle edge.
# ----------------------------------------------------------------------


def test_point_on_shared_triangle_edge_does_not_raise() -> None:
    """A square split into 2 triangles along a diagonal -- points near/on that diagonal must interpolate cleanly."""
    points = (
        Point3D(0.0, 0.0, 0.0),
        Point3D(10.0, 0.0, 10.0),
        Point3D(0.0, 10.0, 20.0),
        Point3D(10.0, 10.0, 30.0),
    )
    tin = TIN.from_points(points)

    for t in np.linspace(0.01, 9.99, 20):
        tin.interpolate(float(t), float(t))  # must not raise for any point strictly inside the bbox


# ----------------------------------------------------------------------
# Asymmetric NoData: TIN comparison.
# ----------------------------------------------------------------------


def test_asymmetric_nodata_triangular_vs_full_square() -> None:
    existing = _triangle(100.0)  # covers only half of its own 20x20 bbox
    proposed = _square(90.0, 0.0, 20.0)  # covers the full 20x20 bbox

    result = TINComparison(resolution=2.0).compute(existing, proposed)
    total = result.valid_cells + result.excluded_cells

    assert total == 100
    assert 0.35 < result.valid_cells / total < 0.65


# ----------------------------------------------------------------------
# Regression: TINComparison vs manual sampling + SurfaceComparison.
# ----------------------------------------------------------------------


def test_tin_comparison_matches_manual_sampling_exactly() -> None:
    existing = _triangle(100.0)
    proposed = _square(90.0, 0.0, 20.0)
    resolution = 2.0

    result = TINComparison(resolution=resolution).compute(existing, proposed)

    min_x = max(existing.bounds[0], proposed.bounds[0])
    min_y = max(existing.bounds[1], proposed.bounds[1])
    max_x = min(existing.bounds[2], proposed.bounds[2])
    max_y = min(existing.bounds[3], proposed.bounds[3])
    columns = max(1, int(np.ceil((max_x - min_x) / resolution)))
    rows = max(1, int(np.ceil((max_y - min_y) / resolution)))
    xs = min_x + (np.arange(columns) + 0.5) * resolution
    ys = min_y + (np.arange(rows) + 0.5) * resolution

    manual_existing = np.full((rows, columns), np.nan)
    manual_proposed = np.full((rows, columns), np.nan)
    for row in range(rows):
        for col in range(columns):
            try:
                manual_existing[row, col] = existing.interpolate(float(xs[col]), float(ys[row]))
            except ValueError:
                pass
            try:
                manual_proposed[row, col] = proposed.interpolate(float(xs[col]), float(ys[row]))
            except ValueError:
                pass

    manual_result = SurfaceComparison(tolerance=0.0).compute(manual_existing, manual_proposed)

    assert result.valid_cells == manual_result.valid_cells
    assert np.array_equal(result.difference, manual_result.difference, equal_nan=True)


# ----------------------------------------------------------------------
# Regression: SurfaceCutFill matches pre-existing CutFillVolume.
# ----------------------------------------------------------------------


def test_surface_cut_fill_matches_cut_fill_volume_on_mixed_nodata_array() -> None:
    existing = np.array([[100.0, 110.0, 105.0, np.nan]])
    proposed = np.array([[110.0, 100.0, 105.0, 108.0]])

    _comparison, volume = SurfaceCutFill(cell_area=2.0, tolerance=0.0).compute(existing, proposed)
    reference = CutFillVolume(cell_area=2.0).compute(existing, proposed)

    assert volume.cut_volume == pytest.approx(reference.cut_volume)
    assert volume.fill_volume == pytest.approx(reference.fill_volume)
    assert volume.net_volume == pytest.approx(reference.net_volume)


# ----------------------------------------------------------------------
# Invariants.
# ----------------------------------------------------------------------


def test_invariants_hold_for_gridded_comparison() -> None:
    comparison = SurfaceComparison(tolerance=0.0)
    result = comparison.compute(
        np.array([[100.0, 110.0, 105.0, np.nan]]),
        np.array([[110.0, 100.0, 105.0, 108.0]]),
    )

    total_cells = result.difference.size
    assert result.valid_cells + result.excluded_cells == total_cells
    assert result.cut_cells + result.fill_cells + result.unchanged_cells == result.valid_cells


def test_invariants_hold_for_tin_comparison() -> None:
    existing = _triangle(100.0)
    proposed = _square(90.0, 0.0, 20.0)

    result = TINComparison(resolution=2.0).compute(existing, proposed)

    total_cells = result.difference.size
    assert result.valid_cells + result.excluded_cells == total_cells
    assert result.cut_cells + result.fill_cells + result.unchanged_cells == result.valid_cells


# ----------------------------------------------------------------------
# NoData: both NaN, all valid.
# ----------------------------------------------------------------------


def test_both_nan_at_same_cell_excluded() -> None:
    comparison = SurfaceComparison()
    result = comparison.compute(np.array([[np.nan, 100.0]]), np.array([[np.nan, 105.0]]))

    assert result.valid_cells == 1
    assert result.excluded_cells == 1


def test_all_valid_no_exclusions() -> None:
    comparison = SurfaceComparison()
    result = comparison.compute(
        np.array([[100.0, 105.0, 98.0]]),
        np.array([[102.0, 103.0, 99.0]]),
    )

    assert result.valid_cells == 3
    assert result.excluded_cells == 0
