"""
Regression suite for topocore.analysis.volume.cut_fill.CutFillVolume
and topocore.analysis.volume.grid_volume.GridVolume -- PR19.

Includes the real bug found and fixed in this session (see
topocore.analysis._shared.test_volume for the full description):
both classes now correctly handle DTMs with legitimate NaN cells
(e.g. outside a triangulated surface's convex hull) instead of
rejecting the entire computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.volume.cut_fill import CutFillVolume
from topocore.analysis.volume.grid_volume import GridVolume
from topocore.geometry.point3d import Point3D
from topocore.terrain.dtm import DTM
from topocore.terrain.grid import Grid
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.tin import TIN


@pytest.fixture
def dtm_with_nan_edges() -> tuple[np.ndarray, np.ndarray]:
    """
    A real DTM (via DTM.from_tin(), the same code path already
    audited and fixed elsewhere in this session) whose grid legitimately
    extends beyond the source TIN's triangular convex hull -- NaN at
    the corners, not an error condition.
    """
    points = (Point3D(0, 0, 100.0), Point3D(10, 0, 102.0), Point3D(5, 10, 105.0))
    tin = TIN.from_points(points)
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=2.0)
    existing = DTM.from_tin(tin, grid, LinearInterpolator(tin))
    proposed_elevations = existing.elevations - 1.0
    return existing.elevations, proposed_elevations


# ----------------------------------------------------------------------
# CutFillVolume
# ----------------------------------------------------------------------


def test_cut_fill_handles_real_dtm_with_nan_edges(
    dtm_with_nan_edges: tuple[np.ndarray, np.ndarray],
) -> None:
    existing, proposed = dtm_with_nan_edges
    n_valid = int(np.isfinite(existing).sum())

    result = CutFillVolume(cell_area=4.0).compute(existing, proposed)  # must not raise

    assert result.valid_cells == n_valid
    assert result.excluded_cells == existing.size - n_valid
    assert result.cut_volume == pytest.approx(n_valid * 1.0 * 4.0)


def test_cut_fill_still_rejects_all_nan() -> None:
    with pytest.raises(VolumeError):
        CutFillVolume(cell_area=1.0).compute(np.full((3, 3), np.nan), np.zeros((3, 3)))


def test_cut_fill_still_rejects_infinite() -> None:
    bad = np.array([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(VolumeError):
        CutFillVolume(cell_area=1.0).compute(bad, np.zeros((2, 2)))


def test_cut_fill_no_nan_reports_zero_excluded() -> None:
    existing = np.full((5, 5), 10.0)
    proposed = np.full((5, 5), 9.0)
    result = CutFillVolume(cell_area=1.0).compute(existing, proposed)

    assert result.valid_cells == 25
    assert result.excluded_cells == 0


def test_cut_fill_rejects_nonpositive_cell_area() -> None:
    with pytest.raises(VolumeError):
        CutFillVolume(cell_area=0.0)


# ----------------------------------------------------------------------
# GridVolume -- shares the same underlying compute_cut_fill().
# ----------------------------------------------------------------------


def test_grid_volume_handles_real_dtm_with_nan_edges(
    dtm_with_nan_edges: tuple[np.ndarray, np.ndarray],
) -> None:
    existing, proposed = dtm_with_nan_edges
    n_valid = int(np.isfinite(existing).sum())

    result = GridVolume(resolution=2.0).compute(existing, proposed)  # must not raise

    assert result.valid_cells == n_valid
    assert result.excluded_cells == existing.size - n_valid


def test_grid_volume_still_rejects_all_nan() -> None:
    with pytest.raises(VolumeError):
        GridVolume(resolution=1.0).compute(np.full((3, 3), np.nan), np.zeros((3, 3)))
