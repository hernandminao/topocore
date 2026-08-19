"""
Regression suite for topocore.analysis.profile.longitudinal.
LongitudinalProfile -- PR19.

Verified against a known tilted plane (z=x) and known distances
(3-4-5 diagonal). Station generation verified with exact-division,
non-exact-division, and floating-point-precision-risk cases
(interval=0.1 over axis=10.0). No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import ProfileError
from topocore.analysis.profile.longitudinal import LongitudinalProfile
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

from ._helpers import SurfaceAdapter


@pytest.fixture
def tilted_plane_surface() -> SurfaceAdapter:
    points = (
        Point3D(-10, -10, -10.0),
        Point3D(20, -10, 20.0),
        Point3D(-10, 20, -10.0),
        Point3D(20, 20, 20.0),
    )
    return SurfaceAdapter(TIN.from_points(points))


def test_profile_matches_known_tilted_plane(
    tilted_plane_surface: SurfaceAdapter,
) -> None:
    profile = LongitudinalProfile(interval=2.0)
    result = profile.generate((0.0, 0.0), (10.0, 0.0), tilted_plane_surface)

    assert result.axis_length == pytest.approx(10.0)
    for point in result.points:
        assert point.z == pytest.approx(point.x, abs=1e-6)  # z=x plane
        assert point.offset == 0.0


def test_diagonal_axis_length_known_3_4_5(
    tilted_plane_surface: SurfaceAdapter,
) -> None:
    result = LongitudinalProfile(interval=1.0).generate((0.0, 0.0), (3.0, 4.0), tilted_plane_surface)
    assert result.axis_length == pytest.approx(5.0)


def test_generate_stations_exact_division() -> None:
    profile = LongitudinalProfile(interval=2.0)
    stations = profile._generate_stations(10.0)
    assert stations == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_generate_stations_non_exact_division() -> None:
    profile = LongitudinalProfile(interval=3.0)
    stations = profile._generate_stations(10.0)
    assert stations == [0.0, 3.0, 6.0, 9.0, 10.0]


def test_generate_stations_floating_point_precision() -> None:
    """
    interval=0.1 over axis=10.0 is a classic floating-point division
    risk (10.0 // 0.1 can be 99 instead of 100 due to representation
    error) -- verified this does NOT happen here.
    """
    profile = LongitudinalProfile(interval=0.1)
    stations = profile._generate_stations(10.0)
    assert len(stations) == 101
    assert stations[0] == pytest.approx(0.0)
    assert stations[-1] == pytest.approx(10.0)


def test_generate_stations_interval_exceeds_axis() -> None:
    profile = LongitudinalProfile(interval=100.0)
    stations = profile._generate_stations(10.0)
    assert stations == [0.0, 10.0]


def test_rejects_identical_origin_and_target(
    tilted_plane_surface: SurfaceAdapter,
) -> None:
    profile = LongitudinalProfile()
    with pytest.raises(ProfileError):
        profile.generate((5.0, 5.0), (5.0, 5.0), tilted_plane_surface)


def test_rejects_nonpositive_interval() -> None:
    with pytest.raises(ProfileError):
        LongitudinalProfile(interval=0.0)


def test_rejects_nan_coordinate(tilted_plane_surface: SurfaceAdapter) -> None:
    profile = LongitudinalProfile()
    with pytest.raises(ProfileError):
        profile.generate((float("nan"), 0.0), (10.0, 0.0), tilted_plane_surface)
