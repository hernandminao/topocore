"""
Regression suite for GeodesicCalculator -- PR19, module Geodesy.

Verifies the pyproj.Geod return-value unpacking order in each method
(a real risk: Geod.inv() returns (az1, az2, dist), easy to mis-order)
against known, independently-derivable geodesic geometry -- not
mocks, not assumed from the method names alone.
"""

from __future__ import annotations

import pytest

from topocore.geodesy.crs import CRS
from topocore.geodesy.geodesic import GeodesicCalculator


@pytest.fixture
def wgs84_calc() -> GeodesicCalculator:
    return GeodesicCalculator(CRS.from_epsg(4326))


# ----------------------------------------------------------------------
# distance(): along the equator, exact analytic value = a * delta_lon_rad
# (equator has radius exactly equal to the semi-major axis).
# ----------------------------------------------------------------------


def test_distance_along_equator_matches_analytic_value(
    wgs84_calc: GeodesicCalculator,
) -> None:
    import math

    semi_major_axis = 6378137.0  # WGS84
    delta_lon_deg = 1.0

    distance = wgs84_calc.distance(0.0, 0.0, delta_lon_deg, 0.0)
    expected = semi_major_axis * math.radians(delta_lon_deg)

    assert distance == pytest.approx(expected, rel=1e-9)


def test_distance_zero_between_identical_points(wgs84_calc: GeodesicCalculator) -> None:
    assert wgs84_calc.distance(-74.5, 40.3, -74.5, 40.3) == pytest.approx(0.0, abs=1e-6)


def test_distance_is_symmetric(wgs84_calc: GeodesicCalculator) -> None:
    forward = wgs84_calc.distance(-74.5, 40.3, 139.7, 35.7)
    backward = wgs84_calc.distance(139.7, 35.7, -74.5, 40.3)
    assert forward == pytest.approx(backward)


# ----------------------------------------------------------------------
# azimuth(): confirms it returns the FORWARD azimuth (index 0 of
# Geod.inv()'s return), not the back azimuth or distance.
# ----------------------------------------------------------------------


def test_azimuth_due_east_along_equator_is_90(wgs84_calc: GeodesicCalculator) -> None:
    azimuth = wgs84_calc.azimuth(0.0, 0.0, 10.0, 0.0)
    assert azimuth == pytest.approx(90.0, abs=1e-6)


def test_azimuth_due_north_is_0(wgs84_calc: GeodesicCalculator) -> None:
    azimuth = wgs84_calc.azimuth(0.0, 0.0, 0.0, 10.0)
    assert azimuth == pytest.approx(0.0, abs=1e-6)


def test_azimuth_due_south_is_180(wgs84_calc: GeodesicCalculator) -> None:
    azimuth = wgs84_calc.azimuth(0.0, 10.0, 0.0, 0.0)
    assert abs(azimuth) == pytest.approx(180.0, abs=1e-6)


# ----------------------------------------------------------------------
# forward()/inverse(): round-trip consistency with known values.
# ----------------------------------------------------------------------


def test_forward_then_inverse_round_trips(wgs84_calc: GeodesicCalculator) -> None:
    lon2, lat2, _ = wgs84_calc.forward(-74.5, 40.3, 45.0, 50000.0)
    forward_az, _, distance = wgs84_calc.inverse(-74.5, 40.3, lon2, lat2)

    assert forward_az == pytest.approx(45.0, abs=1e-6)
    assert distance == pytest.approx(50000.0, abs=1e-3)


def test_forward_due_east_100km_matches_analytic_longitude_shift(
    wgs84_calc: GeodesicCalculator,
) -> None:
    import math

    semi_major_axis = 6378137.0
    lon2, lat2, back_azimuth = wgs84_calc.forward(0.0, 0.0, 90.0, 100000.0)

    expected_lon2 = math.degrees(100000.0 / semi_major_axis)
    assert lon2 == pytest.approx(expected_lon2, abs=1e-6)
    assert lat2 == pytest.approx(0.0, abs=1e-9)
    assert back_azimuth == pytest.approx(-90.0, abs=1e-6)


def test_inverse_returns_az1_az2_distance_in_that_order(
    wgs84_calc: GeodesicCalculator,
) -> None:
    """
    Direct check of the tuple unpacking order documented in
    inverse()'s docstring (forward azimuth, back azimuth, distance).
    """
    az1, az2, distance = wgs84_calc.inverse(0.0, 0.0, 10.0, 0.0)

    assert az1 == pytest.approx(90.0, abs=1e-6)  # due east
    assert abs(az2) == pytest.approx(90.0, abs=1e-6)  # back azimuth, same line
    assert distance > 0.0


# ----------------------------------------------------------------------
# polygon_area(): known square near the equator.
# ----------------------------------------------------------------------


def test_polygon_area_one_degree_square_near_equator(
    wgs84_calc: GeodesicCalculator,
) -> None:
    lons = [0.0, 1.0, 1.0, 0.0]
    lats = [0.0, 0.0, 1.0, 1.0]

    area = wgs84_calc.polygon_area(lons, lats)

    # ~111km x 111km at the equator -> ~1.23e10 m^2, verified
    # independently against pyproj's own polygon_area_perimeter
    # during this session's audit (see HANDOFF notes).
    assert area == pytest.approx(12308778361.47, rel=1e-6)


def test_polygon_area_is_always_positive(wgs84_calc: GeodesicCalculator) -> None:
    """
    Vertex winding order (CW vs CCW) must not produce a negative
    area -- the method explicitly takes abs().
    """
    lons_ccw = [0.0, 1.0, 1.0, 0.0]
    lats_ccw = [0.0, 0.0, 1.0, 1.0]
    lons_cw = list(reversed(lons_ccw))
    lats_cw = list(reversed(lats_ccw))

    area_ccw = wgs84_calc.polygon_area(lons_ccw, lats_ccw)
    area_cw = wgs84_calc.polygon_area(lons_cw, lats_cw)

    assert area_ccw > 0.0
    assert area_cw > 0.0
    assert area_ccw == pytest.approx(area_cw)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_construction_uses_crs_ellipsoid() -> None:
    """
    Different ellipsoids should produce (slightly) different
    distances for the same lon/lat inputs -- confirms the CRS's own
    ellipsoid is actually used, not a hardcoded default.
    """
    wgs84 = GeodesicCalculator(CRS.from_epsg(4326))
    nad83 = GeodesicCalculator(CRS.from_epsg(4269))  # GRS80 ellipsoid, slightly different

    d1 = wgs84.distance(-74.5, 40.3, 139.7, 35.7)
    d2 = nad83.distance(-74.5, 40.3, 139.7, 35.7)

    # Not required to differ by a specific amount, but WGS84 and
    # GRS80 have (very slightly) different flattening, so an exact
    # equality would suggest the ellipsoid isn't actually being used.
    assert d1 != d2
    assert d1 == pytest.approx(d2, rel=1e-6)  # still close -- ellipsoids are nearly identical
