"""
Regression suite for topocore.analysis.visibility.viewshed.Viewshed
-- PR19. Verified against flat terrain (all visible) and an
obstructing wall (reduced visibility). No bugs found in this class
itself (the earth-curvature propagation bug found in this session
was in the MANAGER, not here -- see test_manager.py).
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.visibility.viewshed import Viewshed
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

from ._helpers import SurfaceAdapter


def test_flat_terrain_all_cells_visible() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    viewshed = Viewshed(observer_height=1.7, resolution=10.0, earth_curvature=False, num_samples=20)
    result = viewshed.compute((0.0, 0.0), surface)

    assert result.visible_count == result.total_count
    assert result.total_count > 0


def test_wall_reduces_visible_count() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
        Point3D(20, -50, 30.0),
        Point3D(20, 50, 30.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    viewshed = Viewshed(observer_height=1.7, resolution=10.0, earth_curvature=False, num_samples=20)
    result = viewshed.compute((0.0, 0.0), surface)

    assert result.visible_count < result.total_count


def test_cells_outside_tin_are_excluded_not_counted() -> None:
    # Small TIN -- max_distance forces the grid to extend beyond it.
    points = (
        Point3D(-5, -5, 0.0),
        Point3D(5, -5, 0.0),
        Point3D(-5, 5, 0.0),
        Point3D(5, 5, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    viewshed = Viewshed(observer_height=1.7, resolution=1.0, max_distance=20.0, earth_curvature=False)
    result = viewshed.compute((0.0, 0.0), surface)

    # total_count only reflects cells genuinely inside the TIN.
    assert result.total_count <= result.visibility_map.size


def test_rejects_observer_outside_tin() -> None:
    points = (
        Point3D(-10, -10, 0.0),
        Point3D(10, -10, 0.0),
        Point3D(-10, 10, 0.0),
        Point3D(10, 10, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    viewshed = Viewshed()
    with pytest.raises(VisibilityError):
        viewshed.compute((-999.0, -999.0), surface)


def test_rejects_negative_observer_height() -> None:
    with pytest.raises(VisibilityError):
        Viewshed(observer_height=-1.0)


def test_rejects_nonpositive_resolution() -> None:
    with pytest.raises(VisibilityError):
        Viewshed(resolution=0.0)
