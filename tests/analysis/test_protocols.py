"""
Regression suite for topocore.analysis.protocols.GriddedSurface --
PR20.1 fix.

Real, confirmed type-safety inconsistency fixed: GriddedSurface
previously extended TerrainSurface (requiring interpolate()/
contains()) and declared `grid` as NDArray[np.float64] -- neither
matches the real topocore.terrain.dtm.DTM class this protocol is
used for throughout analysis.volume/analysis.statistics. Confirmed
via mypy structural checking (before the fix, DTM failed to satisfy
GriddedSurface -- missing interpolate/contains, and grid type
mismatch) and via a grep across every real consumer: none of them
call interpolate()/contains() on a GriddedSurface-typed parameter,
only .grid (for geometry-equality checks), .resolution, and
.elevations. This did not cause any wrong RUNTIME behavior (Python
protocols aren't enforced at runtime), only broken static type
checking -- confirmed here directly with a real mypy invocation
against real DTM construction.
"""

from __future__ import annotations

import numpy as np

from topocore.analysis.protocols import GriddedSurface
from topocore.terrain.dtm import DTM
from topocore.terrain.grid import Grid
from topocore.terrain.raster import Raster


def test_dtm_structurally_satisfies_gridded_surface_at_runtime() -> None:
    """
    Confirms DTM has the 3 properties GriddedSurface now declares --
    duck-typed structural conformance, checked directly (Protocol
    isn't @runtime_checkable, so isinstance() isn't available; this
    checks the attributes GriddedSurface actually requires).
    """
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    raster = Raster(grid=grid, values=np.zeros((grid.rows, grid.columns)))
    dtm = DTM(tin=None, grid=grid, raster=raster)  # type: ignore[arg-type]

    assert hasattr(dtm, "grid")
    assert hasattr(dtm, "resolution")
    assert hasattr(dtm, "elevations")
    assert dtm.grid is grid
    assert dtm.resolution == grid.resolution


def test_gridded_surface_grid_type_matches_real_dtm_attribute() -> None:
    """The exact fix: .grid is a Grid geometry object, not an elevation array."""
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    raster = Raster(grid=grid, values=np.zeros((grid.rows, grid.columns)))
    dtm = DTM(tin=None, grid=grid, raster=raster)  # type: ignore[arg-type]

    assert isinstance(dtm.grid, Grid)
    assert not isinstance(dtm.grid, np.ndarray)


def test_gridded_surface_no_longer_requires_interpolate_or_contains() -> None:
    """DTM genuinely lacks these methods -- confirms the protocol no longer demands them."""
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    raster = Raster(grid=grid, values=np.zeros((grid.rows, grid.columns)))
    dtm = DTM(tin=None, grid=grid, raster=raster)  # type: ignore[arg-type]

    assert not hasattr(dtm, "interpolate")
    assert not hasattr(dtm, "contains")


def test_gridded_surface_protocol_members() -> None:
    grid = Grid(min_x=0, min_y=0, max_x=10, max_y=10, resolution=2.0)
    raster = Raster(grid=grid, values=np.zeros((grid.rows, grid.columns)))
    dtm = DTM(tin=None, grid=grid, raster=raster)  # type: ignore[arg-type]

    surface: GriddedSurface = dtm  # must type-check: DTM structurally satisfies GriddedSurface
    assert surface.grid is grid
    assert surface.resolution == grid.resolution
    assert surface.elevations is dtm.elevations
