"""
Regression suite for a real bug found during PR19 (not a coverage
gap, a genuine inconsistency in already-shipped Terrain code):
Grid.columns/rows correctly round UP (math.ceil) to guarantee full
coverage of the requested extent when resolution doesn't evenly
divide width/height -- a deliberate, correct choice, kept as-is.

But the REST of the API (bounds/contains()/Raster.transform) kept
describing the ORIGINAL nominal max_x/max_y, not the grid's actual
generated extent -- so a grid's own last column/row could fail its
own contains() check, and Raster.transform (used for GDAL/GeoTIFF
export) could misalign exported rasters whenever the resolution
doesn't evenly divide the requested extent.

Fixed: Grid.bounds/contains()/Raster.transform now describe the
actual generated extent (min -> last generated point), via new
Grid.actual_max_x/actual_max_y properties. columns/rows/coordinate()
themselves were correct and are unchanged.
"""

from __future__ import annotations

import pytest

from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster

# ----------------------------------------------------------------------
# Case 1: exact multiple (nominal extent == actual extent).
# ----------------------------------------------------------------------


def test_exact_multiple_bounds_match_nominal_extent() -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=2.0)

    assert grid.bounds == (0.0, 0.0, 10.0, 10.0)
    assert grid.actual_max_x == pytest.approx(10.0)
    assert grid.actual_max_y == pytest.approx(10.0)


def test_exact_multiple_contains_nominal_max_exactly() -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=2.0)
    assert grid.contains(10.0, 10.0) is True


# ----------------------------------------------------------------------
# Case 2: non-exact multiple -- actual extent exceeds nominal.
# ----------------------------------------------------------------------


def test_non_exact_multiple_bounds_reflect_actual_extent() -> None:
    """
    width=10, resolution=3 -> columns=5 (ceil(10/3)+1), last column's
    own x = 0 + 4*3 = 12.0, exceeding the nominal max_x=10.0.
    bounds must report 12.0, not 10.0.
    """
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)

    assert grid.bounds == (0.0, 0.0, 12.0, 12.0)
    assert grid.actual_max_x == pytest.approx(12.0)
    assert grid.actual_max_y == pytest.approx(12.0)


def test_non_exact_multiple_last_generated_point_is_contained() -> None:
    """
    The exact regression this fix targets: a grid's own last
    generated column/row must satisfy its own contains() check.
    """
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)

    last_x = grid.x(grid.columns - 1)
    last_y = grid.y(grid.rows - 1)

    assert grid.contains(last_x, last_y) is True


def test_non_exact_multiple_rejects_beyond_actual_extent() -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)

    assert grid.contains(12.0001, 0.0) is False


def test_columns_rows_and_coordinate_unaffected_by_the_fix() -> None:
    """
    The rounding-up behavior itself (columns/rows/coordinate()) is
    correct and deliberately unchanged -- only the metadata
    describing the extent was wrong.
    """
    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)

    assert grid.columns == 5
    assert grid.rows == 5
    assert grid.x(4) == pytest.approx(12.0)
    assert grid.y(4) == pytest.approx(12.0)


# ----------------------------------------------------------------------
# Raster.transform (GDAL export) -- must use the actual extent's
# top edge, not the nominal one, or exports misalign.
# ----------------------------------------------------------------------


def test_raster_transform_uses_actual_max_y_for_non_exact_grid() -> None:
    import numpy as np

    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)
    raster = Raster(grid=grid, values=np.full(grid.shape, 5.0))

    origin_x, pixel_width, _, origin_y, _, pixel_height = raster.transform

    assert origin_x == pytest.approx(0.0)
    assert pixel_width == pytest.approx(3.0)
    assert origin_y == pytest.approx(12.0)  # actual top edge, not nominal 10.0
    assert pixel_height == pytest.approx(-3.0)


def test_raster_transform_matches_nominal_for_exact_grid() -> None:
    import numpy as np

    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=2.0)
    raster = Raster(grid=grid, values=np.full(grid.shape, 5.0))

    _, _, _, origin_y, _, _ = raster.transform

    assert origin_y == pytest.approx(10.0)


# ----------------------------------------------------------------------
# Raster.bounds/extent delegate to grid.bounds -- confirm the fix
# propagates through, not just tested at the Grid level.
# ----------------------------------------------------------------------


def test_raster_bounds_reflect_actual_extent() -> None:
    import numpy as np

    grid = Grid(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, resolution=3.0)
    raster = Raster(grid=grid, values=np.full(grid.shape, 5.0))

    assert raster.bounds == (0.0, 0.0, 12.0, 12.0)
    assert raster.extent == raster.bounds
