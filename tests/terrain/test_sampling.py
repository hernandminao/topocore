"""
Regression suite for topocore.terrain.sampling -- PR19.

bilinear() verified against an exact linear plane (z = x + 2y) at
non-grid-aligned points -- bilinear interpolation must reproduce a
truly linear surface exactly, not approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster
from topocore.terrain.sampling import RasterSampler


@pytest.fixture
def plane_raster() -> Raster:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=3.0, max_y=3.0, resolution=1.0)
    values = np.zeros(grid.shape)
    for row in range(grid.rows):
        for col in range(grid.columns):
            x, y = grid.coordinate(row, col)
            values[row, col] = x + 2.0 * y
    return Raster(grid=grid, values=values)


def test_bilinear_matches_exact_linear_plane(plane_raster: Raster) -> None:
    assert RasterSampler.bilinear(plane_raster, 1.5, 2.25) == pytest.approx(1.5 + 2.0 * 2.25)


def test_bilinear_matches_exact_linear_plane_second_point(plane_raster: Raster) -> None:
    assert RasterSampler.bilinear(plane_raster, 0.3, 0.7) == pytest.approx(0.3 + 2.0 * 0.7)


def test_bilinear_at_exact_grid_point_matches_cell_value(plane_raster: Raster) -> None:
    assert RasterSampler.bilinear(plane_raster, 1.0, 1.0) == pytest.approx(1.0 + 2.0 * 1.0)


def test_bilinear_returns_nan_when_a_neighbor_is_nodata() -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=2.0, max_y=2.0, resolution=1.0)
    values = np.zeros(grid.shape)
    values[1, 1] = float("nan")
    raster = Raster(grid=grid, values=values)

    result = RasterSampler.bilinear(raster, 0.5, 0.5)
    assert np.isnan(result)


def test_bilinear_raises_outside_grid(plane_raster: Raster) -> None:
    with pytest.raises(ValueError, match="outside raster"):
        RasterSampler.bilinear(plane_raster, 100.0, 100.0)


def test_nearest_snaps_to_closest_grid_point(plane_raster: Raster) -> None:
    result = RasterSampler.nearest(plane_raster, 1.4, 1.4)
    assert result == pytest.approx(1.0 + 2.0 * 1.0)  # rounds to (1,1)


def test_nearest_raises_outside_grid(plane_raster: Raster) -> None:
    with pytest.raises(ValueError, match="outside raster"):
        RasterSampler.nearest(plane_raster, -100.0, -100.0)


def test_cell_matches_nearest_value(plane_raster: Raster) -> None:
    cell = RasterSampler.cell(plane_raster, 1.4, 1.4)
    assert cell.z == RasterSampler.nearest(plane_raster, 1.4, 1.4)


def test_index_matches_grid_row_column(plane_raster: Raster) -> None:
    row, column = RasterSampler.index(plane_raster.grid, 1.4, 1.4)
    assert (row, column) == (plane_raster.grid.row(1.4), plane_raster.grid.column(1.4))


def test_index_raises_outside_grid(plane_raster: Raster) -> None:
    with pytest.raises(ValueError, match="outside grid"):
        RasterSampler.index(plane_raster.grid, 100.0, 100.0)


def test_coordinate_matches_grid_coordinate(plane_raster: Raster) -> None:
    assert RasterSampler.coordinate(plane_raster.grid, 1, 1) == plane_raster.grid.coordinate(1, 1)
