"""Regression tests for topocore.terrain.raster."""

from __future__ import annotations

import numpy as np
import pytest

from topocore.terrain.cell import Cell
from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster


@pytest.fixture
def grid() -> Grid:
    return Grid(
        min_x=0.0,
        min_y=0.0,
        max_x=2.0,
        max_y=2.0,
        resolution=1.0,
    )


@pytest.fixture
def raster(grid: Grid) -> Raster:
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    return Raster(
        grid=grid,
        values=values,
    )


def test_raster_valid_and_nodata_counts(
    raster: Raster,
) -> None:
    assert raster.is_nodata(1, 1) is True
    assert raster.is_nodata(0, 0) is False

    assert raster.valid_count == 8
    assert raster.nodata_count == 1
    assert raster.size == 9


def test_raster_shape_dimensions_and_resolution(
    raster: Raster,
) -> None:
    assert raster.shape == (3, 3)
    assert raster.rows == 3
    assert raster.columns == 3
    assert raster.size == 9
    assert raster.resolution == pytest.approx(1.0)


def test_raster_bounds_and_extent(
    raster: Raster,
) -> None:
    assert raster.bounds == raster.grid.bounds
    assert raster.extent == raster.bounds


def test_raster_cell_size(
    raster: Raster,
) -> None:
    assert raster.cell_size == pytest.approx(raster.resolution)


def test_raster_statistics(
    raster: Raster,
) -> None:
    assert raster.minimum == pytest.approx(1.0)
    assert raster.maximum == pytest.approx(9.0)

    expected_mean = np.nanmean(raster.values)
    expected_std = np.nanstd(raster.values)

    assert raster.mean == pytest.approx(expected_mean)
    assert raster.std == pytest.approx(expected_std)


def test_raster_value_and_getitem(
    raster: Raster,
) -> None:
    assert raster.value(0, 0) == pytest.approx(1.0)
    assert raster.value(2, 2) == pytest.approx(9.0)

    assert raster[0, 0] == pytest.approx(1.0)
    assert raster[2, 2] == pytest.approx(9.0)

    assert np.isnan(raster[1, 1])


def test_raster_contains(
    raster: Raster,
) -> None:
    assert raster.contains(0.0, 0.0) is True
    assert raster.contains(1.0, 1.0) is True
    assert raster.contains(2.0, 2.0) is True

    assert raster.contains(-1.0, 1.0) is False
    assert raster.contains(1.0, 3.0) is False


def test_raster_elevation(
    raster: Raster,
) -> None:
    assert raster.elevation(0.0, 0.0) == pytest.approx(1.0)
    assert raster.elevation(1.0, 0.0) == pytest.approx(2.0)
    assert raster.elevation(2.0, 2.0) == pytest.approx(9.0)


def test_raster_cell_returns_cell(
    raster: Raster,
) -> None:
    cell = raster.cell(1, 1)

    assert isinstance(cell, Cell)
    assert cell.row == 1
    assert cell.column == 1
    assert cell.x == pytest.approx(1.0)
    assert cell.y == pytest.approx(1.0)
    assert np.isnan(cell.z)


def test_raster_array_returns_defensive_copy(
    raster: Raster,
) -> None:
    result = raster.array()

    assert result is not raster.values

    np.testing.assert_equal(result, raster.values)

    result[0, 0] = 999.0

    assert raster.value(0, 0) == pytest.approx(1.0)


def test_raster_copy_is_independent(
    raster: Raster,
) -> None:
    copied = raster.copy()

    assert copied is not raster
    assert copied.grid is raster.grid
    assert copied.values is not raster.values

    np.testing.assert_equal(copied.values, raster.values)

    copied.values[0, 0] = 999.0

    assert raster.value(0, 0) == pytest.approx(1.0)
    assert copied.value(0, 0) == pytest.approx(999.0)


def test_raster_fill_replaces_nodata(
    raster: Raster,
) -> None:
    filled = raster.fill(5.0)

    assert filled is not raster
    assert filled.valid_count == 9
    assert filled.nodata_count == 0

    assert filled.value(1, 1) == pytest.approx(5.0)

    # Original remains unchanged.
    assert raster.is_nodata(1, 1) is True


def test_raster_mask_replaces_false_cells_with_nodata(
    raster: Raster,
) -> None:
    mask = np.ones(
        raster.shape,
        dtype=np.bool_,
    )

    mask[0, 0] = False
    mask[2, 2] = False

    masked = raster.mask(mask)

    assert masked.nodata_count == 3
    assert masked.valid_count == 6

    assert masked.is_nodata(0, 0) is True
    assert masked.is_nodata(2, 2) is True

    # Existing NoData remains NoData.
    assert masked.is_nodata(1, 1) is True

    # Original remains unchanged.
    assert raster.is_nodata(0, 0) is False
    assert raster.is_nodata(2, 2) is False


def test_raster_mask_rejects_wrong_shape(
    raster: Raster,
) -> None:
    mask = np.ones(
        (2, 2),
        dtype=np.bool_,
    )

    with pytest.raises(
        ValueError,
        match="Mask dimensions do not match raster",
    ):
        raster.mask(mask)


def test_raster_valid_cells_excludes_nodata(
    raster: Raster,
) -> None:
    cells = list(raster.valid_cells())

    assert len(cells) == raster.valid_count
    assert all(isinstance(cell, Cell) for cell in cells)

    coordinates = {(cell.row, cell.column) for cell in cells}

    assert (1, 1) not in coordinates
    assert (0, 0) in coordinates
    assert (2, 2) in coordinates


def test_raster_iteration_returns_all_cells(
    raster: Raster,
) -> None:
    cells = list(raster)

    assert len(cells) == raster.size
    assert all(isinstance(cell, Cell) for cell in cells)

    indices = [(cell.row, cell.column) for cell in cells]

    assert indices[0] == (0, 0)
    assert indices[-1] == (2, 2)

    assert (1, 1) in indices


def test_raster_window_center(
    raster: Raster,
) -> None:
    window = raster.window(
        row=1,
        column=1,
    )

    np.testing.assert_equal(
        window,
        raster.values,
    )


def test_raster_window_at_corner(
    raster: Raster,
) -> None:
    window = raster.window(
        row=0,
        column=0,
    )

    np.testing.assert_equal(
        window,
        np.array(
            [
                [1.0, 2.0],
                [4.0, np.nan],
            ],
            dtype=np.float64,
        ),
    )


def test_raster_window_custom_radius(
    raster: Raster,
) -> None:
    window = raster.window(
        row=1,
        column=1,
        radius=0,
    )

    assert window.shape == (1, 1)
    assert np.isnan(window[0, 0])


def test_raster_window_returns_copy(
    raster: Raster,
) -> None:
    window = raster.window(
        row=1,
        column=1,
    )

    window[0, 0] = 999.0

    assert raster.value(0, 0) == pytest.approx(1.0)


def test_raster_transform(
    raster: Raster,
) -> None:
    transform = raster.transform

    assert transform[0] == pytest.approx(raster.grid.min_x)
    assert transform[1] == pytest.approx(raster.grid.resolution)
    assert transform[2] == pytest.approx(0.0)

    assert transform[3] == pytest.approx(
        raster.grid.actual_max_y,
    )
    assert transform[4] == pytest.approx(0.0)
    assert transform[5] == pytest.approx(
        -raster.grid.resolution,
    )


def test_raster_statistics_dictionary(
    raster: Raster,
) -> None:
    statistics = raster.statistics()

    assert statistics["rows"] == raster.rows
    assert statistics["columns"] == raster.columns
    assert statistics["cells"] == raster.size
    assert statistics["valid"] == raster.valid_count
    assert statistics["nodata"] == raster.nodata_count
    assert statistics["minimum"] == pytest.approx(raster.minimum)
    assert statistics["maximum"] == pytest.approx(raster.maximum)
    assert statistics["mean"] == pytest.approx(raster.mean)
    assert statistics["std"] == pytest.approx(raster.std)


def test_raster_constructor_rejects_wrong_dimensions(
    grid: Grid,
) -> None:
    values = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Raster dimensions do not match grid",
    ):
        Raster(
            grid=grid,
            values=values,
        )


def test_raster_nodata_is_nan(
    raster: Raster,
) -> None:
    assert np.isnan(raster.value(1, 1))
    assert raster.is_nodata(1, 1) is True


def test_raster_nodata_value_is_preserved_by_copy_and_fill(
    raster: Raster,
) -> None:
    copied = raster.copy()
    filled = raster.fill(100.0)

    assert np.isnan(copied.nodata)
    assert np.isnan(filled.nodata)
    assert np.isnan(raster.nodata)
