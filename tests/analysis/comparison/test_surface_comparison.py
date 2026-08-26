"""
Regression suite for topocore.analysis.comparison -- PR20.2, new
feature implementation.

New module (SurfaceComparison, SurfaceComparisonResult, SurfaceCutFill)
implemented per the PR20 roadmap, distinct from the pre-existing
analysis.volume module: comparison answers "what changed" (per-cell
cut/fill/unchanged classification), volume answers "how much
material that represents" (aggregate cut/fill/net volumes) --
SurfaceCutFill combines both without either duplicating the other's
math. Validation is delegated to the same
analysis._shared.volume.validate_volume_arrays already used by
CutFillVolume/GridVolume, so surfaces accepted by one are accepted
by the other.

Verified with the exact scenarios from the design document: cut-only,
fill-only, mixed, and NoData cases, plus a real edge case the
document's own validate_volume_arrays reuse doesn't cover on its
own: existing and proposed can each have SOME valid data (so neither
is "entirely NaN", the only case validate_volume_arrays rejects) while
sharing NO overlapping valid cells at all -- confirmed this raises a
clean VolumeError rather than crashing on np.min/np.max of an empty
array.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.comparison import (
    SurfaceComparison,
    SurfaceComparisonResult,
    SurfaceCutFill,
)
from topocore.analysis.exceptions import VolumeError

# ----------------------------------------------------------------------
# SurfaceComparison -- the exact scenarios from the design document.
# ----------------------------------------------------------------------


def test_cut_only_scenario() -> None:
    """existing=110, proposed=100 -> difference=-10 -> cut."""
    comparison = SurfaceComparison()
    result = comparison.compute(np.array([[110.0]]), np.array([[100.0]]))

    assert result.difference[0, 0] == pytest.approx(-10.0)
    assert result.cut_cells == 1
    assert result.fill_cells == 0


def test_fill_only_scenario() -> None:
    """existing=100, proposed=110 -> difference=+10 -> fill."""
    comparison = SurfaceComparison()
    result = comparison.compute(np.array([[100.0]]), np.array([[110.0]]))

    assert result.difference[0, 0] == pytest.approx(10.0)
    assert result.cut_cells == 0
    assert result.fill_cells == 1


def test_mixed_scenario() -> None:
    """existing=[100,110], proposed=[110,100] -> one cut cell, one fill cell."""
    comparison = SurfaceComparison()
    result = comparison.compute(np.array([[100.0, 110.0]]), np.array([[110.0, 100.0]]))

    assert result.cut_cells == 1
    assert result.fill_cells == 1


def test_nodata_scenario() -> None:
    """existing=[100,NaN], proposed=[110,105] -> 1 valid cell, 1 excluded."""
    comparison = SurfaceComparison()
    result = comparison.compute(np.array([[100.0, np.nan]]), np.array([[110.0, 105.0]]))

    assert result.valid_cells == 1
    assert result.excluded_cells == 1
    assert result.difference[0, 0] == pytest.approx(10.0)
    assert np.isnan(result.difference[0, 1])


def test_tolerance_classifies_small_differences_as_unchanged() -> None:
    comparison = SurfaceComparison(tolerance=0.05)
    result = comparison.compute(np.array([[100.0]]), np.array([[100.02]]))

    assert result.unchanged_cells == 1
    assert result.cut_cells == 0
    assert result.fill_cells == 0


def test_tolerance_zero_classifies_any_difference() -> None:
    comparison = SurfaceComparison(tolerance=0.0)
    result = comparison.compute(np.array([[100.0]]), np.array([[100.01]]))

    assert result.fill_cells == 1
    assert result.unchanged_cells == 0


def test_summary_statistics() -> None:
    comparison = SurfaceComparison()
    result = comparison.compute(np.array([[100.0, 110.0, 120.0]]), np.array([[105.0, 105.0, 115.0]]))
    # differences: +5, -5, -5
    assert result.minimum_difference == pytest.approx(-5.0)
    assert result.maximum_difference == pytest.approx(5.0)
    assert result.mean_difference == pytest.approx(-5.0 / 3.0)


def test_call_matches_compute() -> None:
    comparison = SurfaceComparison()
    existing = np.array([[100.0]])
    proposed = np.array([[110.0]])
    assert comparison(existing, proposed).fill_cells == comparison.compute(existing, proposed).fill_cells


def test_tolerance_property() -> None:
    assert SurfaceComparison(tolerance=0.5).tolerance == pytest.approx(0.5)


# ----------------------------------------------------------------------
# SurfaceComparison -- validation, including the empty-overlap edge case.
# ----------------------------------------------------------------------


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(VolumeError, match="cannot be negative"):
        SurfaceComparison(tolerance=-1.0)


def test_rejects_nonfinite_tolerance() -> None:
    with pytest.raises(VolumeError, match="finite"):
        SurfaceComparison(tolerance=float("inf"))


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(VolumeError, match="Shape mismatch"):
        SurfaceComparison().compute(np.array([[1.0, 2.0]]), np.array([[1.0]]))


def test_rejects_empty_arrays() -> None:
    with pytest.raises(VolumeError, match="no elevation values"):
        SurfaceComparison().compute(np.empty((0, 0)), np.empty((0, 0)))


def test_rejects_infinite_values() -> None:
    with pytest.raises(VolumeError, match="infinite"):
        SurfaceComparison().compute(np.array([[np.inf]]), np.array([[1.0]]))


def test_rejects_no_overlapping_valid_cells() -> None:
    """
    The decisive edge case: neither array is entirely NaN (so
    validate_volume_arrays's own checks don't catch this), but their
    valid cells never overlap -- must still raise a clean VolumeError,
    not crash on np.min/np.max of an empty array.
    """
    existing = np.array([[np.nan, 1.0]])
    proposed = np.array([[1.0, np.nan]])

    with pytest.raises(VolumeError, match="no valid overlapping cells"):
        SurfaceComparison().compute(existing, proposed)


# ----------------------------------------------------------------------
# SurfaceCutFill -- combined comparison + volume.
# ----------------------------------------------------------------------


def test_surface_cut_fill_combines_comparison_and_volume() -> None:
    combined = SurfaceCutFill(cell_area=1.0, tolerance=0.0)
    comparison, volume = combined.compute(np.array([[100.0, 110.0]]), np.array([[110.0, 100.0]]))

    assert isinstance(comparison, SurfaceComparisonResult)
    assert comparison.cut_cells == 1
    assert comparison.fill_cells == 1
    assert volume.cut_volume == pytest.approx(10.0)
    assert volume.fill_volume == pytest.approx(10.0)
    assert volume.net_volume == pytest.approx(0.0)


def test_surface_cut_fill_call_matches_compute() -> None:
    combined = SurfaceCutFill(cell_area=1.0)
    existing = np.array([[110.0]])
    proposed = np.array([[100.0]])
    comparison_call, volume_call = combined(existing, proposed)
    comparison_compute, volume_compute = combined.compute(existing, proposed)
    assert comparison_call.cut_cells == comparison_compute.cut_cells
    assert volume_call.cut_volume == volume_compute.cut_volume


def test_surface_cut_fill_properties() -> None:
    combined = SurfaceCutFill(cell_area=4.0, tolerance=0.1)
    assert combined.tolerance == pytest.approx(0.1)
    assert combined.cell_area == pytest.approx(4.0)


def test_surface_cut_fill_rejects_nonpositive_cell_area() -> None:
    with pytest.raises(VolumeError, match="positive"):
        SurfaceCutFill(cell_area=0.0)


def test_surface_cut_fill_rejects_nonfinite_cell_area() -> None:
    with pytest.raises(VolumeError, match="finite"):
        SurfaceCutFill(cell_area=float("nan"))


# ----------------------------------------------------------------------
# SurfaceComparisonResult -- derived properties.
# ----------------------------------------------------------------------


def test_result_cell_count_properties_match_masks() -> None:
    comparison = SurfaceComparison()
    result = comparison.compute(
        np.array([[100.0, 110.0, 105.0]]),
        np.array([[110.0, 100.0, 105.0]]),
    )

    assert result.cut_cells == int(np.count_nonzero(result.cut_mask))
    assert result.fill_cells == int(np.count_nonzero(result.fill_mask))
    assert result.unchanged_cells == int(np.count_nonzero(result.unchanged_mask))
    assert result.cut_cells + result.fill_cells + result.unchanged_cells == result.valid_cells
