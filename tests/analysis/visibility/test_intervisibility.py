"""
Regression suite for topocore.analysis.visibility.intervisibility.
Intervisibility -- PR19. Verified matrix symmetry, exact obstacle
geometry (including a diagonal path independently confirmed to cross
the obstacle's actual position -- an initial "unexpected" result
during the audit that turned out to be geometrically correct, not a
bug), and validation. No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import VisibilityError
from topocore.analysis.visibility.intervisibility import Intervisibility
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

from ._helpers import SurfaceAdapter


def test_flat_terrain_all_pairs_visible() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    iv = Intervisibility(observer_height=1.7, earth_curvature=False, num_samples=50)
    pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    result = iv.compute(pts, surface)

    assert result.visible_pairs == result.total_pairs == 3


def test_matrix_is_symmetric() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
        Point3D(5, -50, 30.0),
        Point3D(5, 50, 30.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    iv = Intervisibility(observer_height=1.7, earth_curvature=False, num_samples=50)
    pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    result = iv.compute(pts, surface)

    assert (result.visibility_matrix == result.visibility_matrix.T).all()


def test_wall_blocks_pairs_whose_path_crosses_it() -> None:
    """
    A wall at x=5 blocks: (0,0)-(10,0) directly (crosses x=5, y=0),
    and (10,0)-(0,10) too -- independently confirmed the diagonal
    path crosses x=5 at y=5, within the wall's y-span. Only
    (0,0)-(0,10) (staying at x=0 the whole time) remains visible.
    """
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
        Point3D(5, -50, 30.0),
        Point3D(5, 50, 30.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    iv = Intervisibility(observer_height=1.7, earth_curvature=False, num_samples=50)
    pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    result = iv.compute(pts, surface)

    assert result.visibility_matrix[0, 1] == False
    assert result.visibility_matrix[1, 2] == False
    assert result.visibility_matrix[0, 2] == True


def test_rejects_fewer_than_two_points() -> None:
    points = (
        Point3D(-10, -10, 0.0),
        Point3D(10, -10, 0.0),
        Point3D(-10, 10, 0.0),
        Point3D(10, 10, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    with pytest.raises(VisibilityError):
        Intervisibility().compute([(0.0, 0.0)], surface)


def test_rejects_point_outside_tin() -> None:
    points = (
        Point3D(-10, -10, 0.0),
        Point3D(10, -10, 0.0),
        Point3D(-10, 10, 0.0),
        Point3D(10, 10, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    with pytest.raises(VisibilityError):
        Intervisibility().compute([(0.0, 0.0), (-999.0, -999.0)], surface)


def test_visibility_network_matches_matrix() -> None:
    points = (
        Point3D(-50, -50, 0.0),
        Point3D(50, -50, 0.0),
        Point3D(-50, 50, 0.0),
        Point3D(50, 50, 0.0),
    )
    surface = SurfaceAdapter(TIN.from_points(points))

    iv = Intervisibility(observer_height=1.7, earth_curvature=False, num_samples=50)
    pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    network = iv.visibility_network(pts, surface)

    assert set(network) == {(0, 1), (0, 2), (1, 2)}
