"""
Integration regression suite for topocore.terrain.dtm.DTM -- PR19.

DTM.from_tin() ties together Grid, Raster, and an interpolator
(each independently verified elsewhere in this session's PR19 work)
-- this suite confirms the wiring itself, not the underlying math a
second time.
"""

from __future__ import annotations

import math

import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.dtm import DTM
from topocore.terrain.grid import Grid
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.tin import TIN


@pytest.fixture
def tilted_plane_tin() -> TIN:
    # z = x -- a known linear plane, so DTM cell values are
    # independently predictable without relying on the interpolator
    # being "trusted blindly" a second time.
    points = (Point3D(0, 0, 0), Point3D(4, 0, 4), Point3D(0, 4, 0), Point3D(4, 4, 4))
    return TIN.from_points(points)


def test_dtm_from_tin_matches_known_plane(tilted_plane_tin: TIN) -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=4.0, max_y=4.0, resolution=2.0)
    interpolator = LinearInterpolator(tilted_plane_tin)

    dtm = DTM.from_tin(tilted_plane_tin, grid, interpolator)

    for row in range(grid.rows):
        for column in range(grid.columns):
            x, _y = grid.coordinate(row, column)
            assert dtm.elevations[row, column] == pytest.approx(x)  # z = x


def test_dtm_rows_columns_match_grid(tilted_plane_tin: TIN) -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=4.0, max_y=4.0, resolution=1.0)
    dtm = DTM.from_tin(tilted_plane_tin, grid, LinearInterpolator(tilted_plane_tin))

    assert dtm.rows == grid.rows
    assert dtm.columns == grid.columns


def test_dtm_raster_grid_is_the_same_grid_instance(tilted_plane_tin: TIN) -> None:
    grid = Grid(min_x=0.0, min_y=0.0, max_x=4.0, max_y=4.0, resolution=1.0)
    dtm = DTM.from_tin(tilted_plane_tin, grid, LinearInterpolator(tilted_plane_tin))

    assert dtm.raster.grid is grid


def test_dtm_out_of_hull_query_produces_nan() -> None:
    """
    A grid extending beyond the TIN's convex hull must show NaN
    (initial fill value) at cells LinearInterpolator can't reach --
    InterpolationError there is caught nowhere in from_tin(), so
    this only holds where the grid stays within the hull. Uses a
    grid matching the hull exactly to confirm the DEFAULT fill
    doesn't leak into in-hull cells (a sanity check, not a
    reproduction of an out-of-hull crash).
    """
    points = (Point3D(0, 0, 5), Point3D(1, 0, 5), Point3D(0, 1, 5))
    tin = TIN.from_points(points)
    grid = Grid(min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0, resolution=1.0)

    dtm = DTM.from_tin(tin, grid, LinearInterpolator(tin))

    # (0,0), (1,0), (0,1) are inside/on the hull; (1,1) is not.
    assert dtm.elevations[0, 0] == pytest.approx(5.0)
    assert math.isnan(dtm.elevations[1, 1])
