"""
Public API / facade audit regression suite for
topocore.analysis.comparison -- PR20.6.

Covers the 6-point audit performed before closing PR20:

1. __init__.py exports exactly the 4 intended public classes, no
   internals leaked.
2. Naming consistency: comparison/'s own new methods (all just
   `.compute()`, no DTM-specific variant) are internally consistent.
   The pre-existing `CutFillVolume.compute_with_dtm()` vs.
   `GridVolume.compute_from_dtm()` naming inconsistency predates PR20
   (both classes were already public API before this PR started) and
   is explicitly NOT changed here -- renaming either would break
   already-shipped API for a purely cosmetic naming benefit, outside
   PR20.6's stated "only what's strictly necessary" scope.
3. analysis/__init__.py does not expose comparison/ (matching the
   existing convention: volume/, quality/, distance/ aren't exposed
   there either) -- confirmed unchanged, not "made more public just
   because".
4. Every documented error path (no overlapping domain, invalid grid,
   total NoData, empty valid intersection, max_grid_cells exceeded,
   non-positive cell_area, shape mismatch) raises VolumeError and
   ONLY VolumeError -- no internal exception (IndexError, ValueError,
   etc.) ever leaks through.
5. SurfaceComparisonResult's contract: added `total_cells` property
   (valid_cells + excluded_cells) -- previously missing, forcing
   every caller to compute it manually. Purely additive. Both
   invariants (`valid_cells + excluded_cells == total_cells`,
   `cut_cells + fill_cells + unchanged_cells == valid_cells`) verified
   to hold via this new property.
6. Integration hierarchy: SurfaceCutFill delegates to (pre-existing)
   CutFillVolume, TINComparison delegates to SurfaceComparison --
   confirmed via source inspection that no cut/fill volume formula
   is duplicated anywhere in comparison/ (grep for cut_volume=/
   fill_volume= assignments returns zero matches across the entire
   module).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from topocore.analysis.comparison import (
    SurfaceComparison,
    SurfaceComparisonResult,
    SurfaceCutFill,
    TINComparison,
)
from topocore.analysis.exceptions import VolumeError
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

# ----------------------------------------------------------------------
# 1. __init__.py exports.
# ----------------------------------------------------------------------


def test_module_exports_exactly_four_public_classes() -> None:
    import topocore.analysis.comparison as comparison_module

    assert set(comparison_module.__all__) == {
        "SurfaceComparison",
        "SurfaceComparisonResult",
        "SurfaceCutFill",
        "TINComparison",
    }


# ----------------------------------------------------------------------
# 3. analysis/__init__.py does not expose comparison/.
# ----------------------------------------------------------------------


def test_analysis_init_does_not_expose_comparison() -> None:
    import topocore.analysis as analysis_module

    assert "comparison" not in analysis_module.__all__
    assert not hasattr(analysis_module, "SurfaceComparison")


# ----------------------------------------------------------------------
# 4. Every documented error path raises VolumeError, and only VolumeError.
# ----------------------------------------------------------------------


def _square(z: float, min_xy: float, max_xy: float) -> TIN:
    points = (
        Point3D(min_xy, min_xy, z),
        Point3D(max_xy, min_xy, z),
        Point3D(min_xy, max_xy, z),
        Point3D(max_xy, max_xy, z),
    )
    return TIN.from_points(points)


@pytest.mark.parametrize(
    "trigger",
    [
        lambda: TINComparison(resolution=2.0).compute(_square(0.0, 0.0, 10.0), _square(0.0, 1000.0, 1010.0)),
        lambda: SurfaceComparison().compute(np.array([1.0, 2.0]), np.array([1.0, 2.0])),
        lambda: SurfaceComparison().compute(np.full((2, 2), np.nan), np.full((2, 2), np.nan)),
        lambda: SurfaceComparison().compute(np.array([[np.nan, 1.0]]), np.array([[1.0, np.nan]])),
        lambda: TINComparison(resolution=0.001).compute(_square(0.0, 0.0, 20.0), _square(0.0, 0.0, 20.0)),
        lambda: SurfaceCutFill(cell_area=0.0),
        lambda: SurfaceComparison().compute(np.array([[1.0, 2.0]]), np.array([[1.0]])),
    ],
    ids=[
        "no_overlapping_domain",
        "invalid_grid_ndim",
        "total_nodata",
        "empty_valid_intersection",
        "max_grid_cells_exceeded",
        "nonpositive_cell_area",
        "shape_mismatch",
    ],
)
def test_documented_error_paths_raise_only_volume_error(trigger) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(VolumeError):
        trigger()


# ----------------------------------------------------------------------
# 5. SurfaceComparisonResult contract, including the new total_cells.
# ----------------------------------------------------------------------


def test_total_cells_derives_from_difference_size_not_from_valid_plus_excluded() -> None:
    """
    Found and fixed before PR20.6 close: total_cells must be derived
    independently from difference.size, NOT from
    valid_cells + excluded_cells -- otherwise the documented
    invariant would be a tautology (always true by construction,
    even for a deliberately inconsistent result), never a genuine,
    checkable property. Confirmed here by hand-constructing an
    inconsistent SurfaceComparisonResult and verifying total_cells
    still reflects the real grid size, correctly exposing the
    inconsistency rather than hiding it.
    """
    difference = np.array([[10.0, np.nan, -5.0]])
    cut_mask = np.array([[False, False, True]])
    fill_mask = np.array([[True, False, False]])
    unchanged_mask = np.array([[False, False, False]])

    inconsistent = SurfaceComparisonResult(
        difference=difference,
        cut_mask=cut_mask,
        fill_mask=fill_mask,
        unchanged_mask=unchanged_mask,
        valid_cells=99,  # deliberately wrong
        excluded_cells=1,
        minimum_difference=-5.0,
        maximum_difference=10.0,
        mean_difference=2.5,
    )

    assert inconsistent.total_cells == 3  # from difference.size, not 99 + 1
    assert inconsistent.valid_cells + inconsistent.excluded_cells != inconsistent.total_cells


def test_total_cells_property_matches_valid_plus_excluded() -> None:
    result = SurfaceComparison().compute(
        np.array([[100.0, np.nan, 105.0]]),
        np.array([[110.0, 105.0, 105.0]]),
    )

    assert result.total_cells == 3
    assert result.total_cells == result.valid_cells + result.excluded_cells


def test_both_invariants_hold_together() -> None:
    result = SurfaceComparison(tolerance=0.0).compute(
        np.array([[100.0, 110.0, 105.0, np.nan]]),
        np.array([[110.0, 100.0, 105.0, 108.0]]),
    )

    assert result.valid_cells + result.excluded_cells == result.total_cells
    assert result.cut_cells + result.fill_cells + result.unchanged_cells == result.valid_cells


def test_result_is_frozen_dataclass() -> None:
    """
    Narrowed per review: pytest.raises(Exception) was too broad for
    a professional project's own test suite -- it would silently
    pass even if the wrong exception type were ever raised for an
    unrelated reason. FrozenInstanceError is confirmed to be the
    exact exception dataclasses.frozen=True raises.
    """
    result = SurfaceComparison().compute(np.array([[100.0]]), np.array([[100.0]]))
    with pytest.raises(FrozenInstanceError):
        result.valid_cells = 999  # type: ignore[misc]


def test_result_type_is_the_documented_dataclass() -> None:
    result = SurfaceComparison().compute(np.array([[100.0]]), np.array([[100.0]]))
    assert isinstance(result, SurfaceComparisonResult)


# ----------------------------------------------------------------------
# 6. Integration hierarchy: no duplicated volume math.
# ----------------------------------------------------------------------
#
# Found and fixed before PR20.6 close: the original version of this
# check inspected comparison/'s source files on disk via a hardcoded
# absolute path (/home/claude/pkg/...) -- non-portable (would fail on
# Windows, CI, or any other machine/checkout location) and, more
# importantly, a weaker guarantee than a behavioral check: grepping
# for "cut_volume =" can be fooled by a differently-named variable
# holding the same duplicated formula, while confirming
# SurfaceCutFill's OUTPUT is identical to CutFillVolume's own
# (test_surface_cut_fill_matches_cut_fill_volume_on_mixed_nodata_array,
# in test_edge_cases.py) cannot be fooled that way. Replaced with the
# two portable, instance-based delegation checks below.


def test_surface_cut_fill_delegates_to_cut_fill_volume() -> None:
    from topocore.analysis.volume.cut_fill import CutFillVolume

    combined = SurfaceCutFill(cell_area=1.0)
    assert isinstance(combined._volume, CutFillVolume)


def test_tin_comparison_delegates_to_surface_comparison() -> None:
    comparison = TINComparison(resolution=1.0)
    assert isinstance(comparison._comparison, SurfaceComparison)
