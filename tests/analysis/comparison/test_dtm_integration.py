"""
Integration regression suite for topocore.analysis.comparison against
real topocore.terrain.dtm.DTM objects -- PR20.3.

Verifies the new comparison/ module works correctly end to end
against the SAME real DTM/Grid/Raster objects the pre-existing,
already-audited analysis.volume module uses -- not just raw NumPy
arrays -- and that both modules agree with each other exactly on
overlapping concerns (valid_cells/excluded_cells, cut volume), since
SurfaceCutFill's whole design premise is that it doesn't duplicate
CutFillVolume's math.

Includes the realistic case that originally motivated
validate_volume_arrays's own NaN-tolerance design in PR19: a DTM
with NaN cells outside its source data's convex hull (e.g. from
DTM.from_tin()). Confirms comparison and volume independently arrive
at the identical valid/excluded cell counts even when the NaN cells
come from DIFFERENT corners of the two surfaces (existing's NaN and
proposed's NaN don't overlap), and that the resulting cut volume
correctly reflects only the genuinely overlapping valid cells.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.comparison import SurfaceComparison, SurfaceCutFill
from topocore.analysis.volume.cut_fill import CutFillVolume
from topocore.terrain.dtm import DTM
from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster


def _make_dtm(grid: Grid, value: float, nan_at: tuple[int, int] | None = None) -> DTM:
    values = np.full((grid.rows, grid.columns), value)
    if nan_at is not None:
        values[nan_at] = np.nan
    raster = Raster(grid=grid, values=values)
    return DTM(tin=None, grid=grid, raster=raster)  # type: ignore[arg-type]


def test_surface_comparison_works_on_real_dtm_elevations() -> None:
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    existing_dtm = _make_dtm(grid, 110.0)
    proposed_dtm = _make_dtm(grid, 100.0)

    result = SurfaceComparison().compute(existing_dtm.elevations, proposed_dtm.elevations)

    assert result.cut_cells == grid.rows * grid.columns
    assert result.fill_cells == 0
    assert result.mean_difference == pytest.approx(-10.0)


def test_surface_cut_fill_matches_existing_cut_fill_volume_on_same_dtm() -> None:
    """
    The decisive cross-check: SurfaceCutFill (new) and
    CutFillVolume.compute_with_dtm() (pre-existing, already audited)
    must agree exactly on the same real DTM pair, confirming
    SurfaceCutFill genuinely delegates rather than reimplementing.
    """
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    existing_dtm = _make_dtm(grid, 110.0)
    proposed_dtm = _make_dtm(grid, 100.0)

    existing_result = CutFillVolume(cell_area=1.0).compute_with_dtm(existing_dtm, proposed_dtm)

    combined = SurfaceCutFill(cell_area=grid.resolution**2, tolerance=0.0)
    _comparison, volume = combined.compute(existing_dtm.elevations, proposed_dtm.elevations)

    assert volume.cut_volume == pytest.approx(existing_result.cut_volume)
    assert volume.fill_volume == pytest.approx(existing_result.fill_volume)
    assert volume.net_volume == pytest.approx(existing_result.net_volume)


def test_dtm_with_nodata_outside_convex_hull_from_different_corners() -> None:
    """
    The realistic PR19 scenario: NaN cells from DTM.from_tin()
    extending past the source data's convex hull. Here existing's
    and proposed's NaN cells are at DIFFERENT corners (don't
    overlap), confirming comparison and volume both correctly
    exclude the UNION of NaN cells, agreeing with each other exactly.
    """
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    existing_dtm = _make_dtm(grid, 110.0, nan_at=(0, 0))
    proposed_dtm = _make_dtm(grid, 100.0, nan_at=(-1, -1))

    total_cells = grid.rows * grid.columns

    combined = SurfaceCutFill(cell_area=grid.resolution**2, tolerance=0.0)
    comparison, volume = combined.compute(existing_dtm.elevations, proposed_dtm.elevations)

    assert comparison.valid_cells == total_cells - 2
    assert comparison.excluded_cells == 2
    assert comparison.valid_cells == volume.valid_cells
    assert comparison.excluded_cells == volume.excluded_cells
    assert comparison.valid_cells + comparison.excluded_cells == total_cells

    expected_cut = (total_cells - 2) * grid.resolution**2 * 10.0
    assert volume.cut_volume == pytest.approx(expected_cut)


def test_dtm_grid_geometry_used_consistently() -> None:
    """Confirms the cell_area derived from a real Grid's resolution matches what compute_with_dtm() derives."""
    grid = Grid(min_x=0, min_y=0, max_x=20, max_y=20, resolution=5.0)
    existing_dtm = _make_dtm(grid, 50.0)
    proposed_dtm = _make_dtm(grid, 45.0)

    combined = SurfaceCutFill(cell_area=grid.resolution**2, tolerance=0.0)
    _comparison, volume = combined.compute(existing_dtm.elevations, proposed_dtm.elevations)

    existing_result = CutFillVolume(cell_area=1.0).compute_with_dtm(existing_dtm, proposed_dtm)
    assert volume.cut_volume == pytest.approx(existing_result.cut_volume)
